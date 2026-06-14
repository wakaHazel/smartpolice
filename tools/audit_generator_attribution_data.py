from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "backend" / "data" / "smartpolice.db"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "docs" / "generator_data_gap_report.md"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

GENERATOR_TASK = "vision_generator_attribution"
REAL_LABEL = "real"

PUBLIC_DATASET_OPTIONS: list[dict[str, str]] = [
    {
        "name": "Synthbuster",
        "fit": "高",
        "use": "多生成器归因评测和补源；每个生成器约 1K 张，覆盖 DALL-E 2/3、Midjourney v5、SD 1.x/2/XL、Firefly、Glide。",
        "labels": "dall-e-3, midjourney, stable-diffusion/sd21/sdxl, glide/unknown, firefly/unknown",
        "caution": "原始 real RAISE-1K 需单独下载；先作为 source-holdout，再决定是否入训练。",
        "url": "https://www.veraai.eu/posts/dataset-synthbuster-towards-detection-of-diffusion-model-generated-images",
    },
    {
        "name": "GenImage",
        "fit": "中",
        "use": "大规模跨生成器检测和扰动评测；覆盖 Midjourney、Stable Diffusion、ADM、GLIDE、Wukong、VQDM、BigGAN。",
        "labels": "midjourney, stable-diffusion, adm/unknown, glide/unknown, wukong/unknown, vqdm/unknown, biggan/unknown",
        "caution": "很多类别不是我们当前强归因标签；优先做 detection/source-holdout，不直接把 unknown 类混入强归因。",
        "url": "https://github.com/GenImage-Dataset/GenImage",
    },
    {
        "name": "MS COCOAI / CT2",
        "fit": "高",
        "use": "专门补 SD3、SDXL、SD2.1、DALL-E 3、Midjourney 等多类归因来源，适合缓解来源耦合。",
        "labels": "sd3, sdxl, sd21, dall-e-3, midjourney",
        "caution": "竞赛平台数据需确认下载权限和许可；先导入为 benchmark_role=external_holdout。",
        "url": "https://codalab.lisn.upsaclay.fr/competitions/20331",
    },
    {
        "name": "AIGCDetectBenchmark",
        "fit": "中",
        "use": "复用其统一训练/评测口径和 CNNSpot/FreDect/DIRE/UnivFD/PatchCraft baseline；更适合二分类和鲁棒性对照。",
        "labels": "binary/generated first; generator labels depend on downloaded test split",
        "caution": "公开方法多为 real/fake 检测，不等价于多模型归因成绩。",
        "url": "https://github.com/Ekko-zn/AIGCDetectBenchmark",
    },
    {
        "name": "UnivFD",
        "fit": "中",
        "use": "用 CLIP 特征 + 线性/近邻的思路增强跨生成器泛化；适合作为我们 binary gate 或 embedding baseline。",
        "labels": "method baseline, not a labeled attribution dataset",
        "caution": "不能直接解决 generator label 不均衡，但能减少只学压缩/尺寸等数据集伪特征。",
        "url": "https://github.com/WisconsinAIVision/UniversalFakeDetect",
    },
    {
        "name": "Synthbuster baseline",
        "fit": "中",
        "use": "Fourier-domain artifact 检测；可作为频域鲁棒检测 baseline 和特征设计参考。",
        "labels": "method baseline, binary synthetic detector",
        "caution": "检测 fake 不等于判断具体生成器；适合第一层初筛或辅助特征。",
        "url": "https://github.com/qbammey/synthbuster",
    },
]

LOCAL_BASELINE_OPTIONS: list[dict[str, str]] = [
    {
        "name": "AIGCDetectBenchmark",
        "path": "external_baselines/AIGCDetectBenchmark",
        "status": "code-local",
        "use": "统一评测 CNNSpot/FreDect/DIRE/UnivFD/PatchCraft；扰动参数覆盖 blur/jpeg/resize。",
        "limit": "主要是 real/fake 检测，不是具体生成器归因。",
    },
    {
        "name": "UniversalFakeDetect",
        "path": "external_baselines/UniversalFakeDetect",
        "status": "code-local",
        "use": "CLIP ViT embedding + 线性头；适合提升跨生成器 fake/real 泛化，减少尺寸/压缩伪特征。",
        "limit": "仍需我们的多源标签数据才能做多类归因。",
    },
    {
        "name": "Synthbuster",
        "path": "external_baselines/synthbuster",
        "status": "code-local",
        "use": "Fourier-domain artifact 检测和 Zenodo 多生成器数据；适合补 DALL-E/MJ/SD 来源。",
        "limit": "Fourier 检测是二分类辅助，不等于直接判断具体生成器。",
    },
]

LABEL_DATASET_RECOMMENDATIONS: dict[str, list[str]] = {
    "dall-e-3": ["Synthbuster", "MS COCOAI / CT2"],
    "midjourney": ["Synthbuster", "GenImage", "MS COCOAI / CT2"],
    "sd21": ["Synthbuster", "MS COCOAI / CT2"],
    "sd3": ["MS COCOAI / CT2"],
    "sdxl": ["Synthbuster", "MS COCOAI / CT2", "AIGCDetectBenchmark"],
    "stable-diffusion": ["Synthbuster", "GenImage"],
    "flux": ["B-Free new generators", "Qwen/Qwen-Image-Bench", "Rapidata/bananamark"],
    "gpt-image2": ["Scam-AI/gpt-image-2", "Qwen/Qwen-Image-Bench", "user/local GPT-image2 pool"],
    "gpt-image1": ["Qwen/Qwen-Image-Bench", "DeepSafe-benchmark"],
    "gpt-image1.5": ["Qwen/Qwen-Image-Bench", "user/local GPT-image1.5 pool"],
    "nano-banana": ["Qwen/Qwen-Image-Bench", "Rapidata/bananamark"],
    "seedream-4": ["Qwen/Qwen-Image-Bench", "Rapidata/bananamark"],
}


@dataclass(frozen=True)
class LabelGap:
    label: str
    total: int
    source_count: int
    largest_source_count: int
    largest_source_share: float
    effective_source_count: float
    status: str
    recommended_new_sources: int
    recommended_samples: int
    top_sources: list[tuple[str, int]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit generator-attribution data balance and write a source-coverage gap report. "
            "This does not train or activate any model."
        )
    )
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="Path to smartpolice.db.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Markdown report path.")
    parser.add_argument("--json-output", default="", help="Optional JSON report path.")
    parser.add_argument("--target-sources", type=int, default=3, help="Desired independent sources per generator.")
    parser.add_argument(
        "--target-per-source",
        type=int,
        default=300,
        help=(
            "Planning target per source for multi-generator attribution. "
            "This is not a cap; GPT-image2-focused training can use far more samples."
        ),
    )
    parser.add_argument(
        "--max-dominant-share",
        type=float,
        default=0.60,
        help="Flag labels whose largest source share is above this value.",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    json_output = Path(args.json_output).expanduser().resolve() if args.json_output else None
    rows = load_source_rows(db_path)
    gaps = build_gaps(
        rows,
        target_sources=args.target_sources,
        target_per_source=args.target_per_source,
        max_dominant_share=args.max_dominant_share,
    )
    payload = build_payload(
        gaps,
        rows,
        db_path=db_path,
        target_sources=args.target_sources,
        target_per_source=args.target_per_source,
        max_dominant_share=args.max_dominant_share,
    )
    write_markdown(output_path, payload)
    if json_output:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote generator attribution data gap report: {output_path}")


def load_source_rows(db_path: Path) -> list[dict[str, Any]]:
    if not db_path.exists():
        raise SystemExit(f"database not found: {db_path}")
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT label, dataset_name, source, COUNT(*)
            FROM external_training_samples
            WHERE task_type = ? AND image_available = 1
            GROUP BY label, dataset_name, source
            ORDER BY label, COUNT(*) DESC
            """,
            (GENERATOR_TASK,),
        ).fetchall()
    return [
        {
            "label": str(label),
            "dataset_name": str(dataset_name),
            "source": str(source),
            "source_key": f"{dataset_name}:{source}",
            "count": int(count),
        }
        for label, dataset_name, source, count in rows
    ]


def build_gaps(
    rows: list[dict[str, Any]],
    *,
    target_sources: int,
    target_per_source: int,
    max_dominant_share: float,
) -> list[LabelGap]:
    by_label: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_label[str(row["label"])][str(row["source_key"])] += int(row["count"])

    gaps: list[LabelGap] = []
    for label, source_counts in sorted(by_label.items()):
        total = sum(source_counts.values())
        if total <= 0:
            continue
        top_sources = source_counts.most_common()
        largest = top_sources[0][1] if top_sources else 0
        largest_share = largest / total
        effective_sources = inverse_simpson(source_counts)
        recommended_new_sources = max(0, target_sources - len(source_counts))
        recommended_samples = recommended_new_sources * target_per_source
        if label == REAL_LABEL:
            status = "negative-pool"
            recommended_new_sources = 0
            recommended_samples = 0
        elif len(source_counts) < 2:
            status = "critical-single-source"
        elif len(source_counts) < target_sources:
            status = "needs-more-sources"
        elif largest_share > max_dominant_share:
            status = "source-dominated"
        else:
            status = "usable"
        gaps.append(
            LabelGap(
                label=label,
                total=total,
                source_count=len(source_counts),
                largest_source_count=largest,
                largest_source_share=largest_share,
                effective_source_count=effective_sources,
                status=status,
                recommended_new_sources=recommended_new_sources,
                recommended_samples=recommended_samples,
                top_sources=top_sources[:4],
            )
        )
    return gaps


def inverse_simpson(source_counts: Counter[str]) -> float:
    total = sum(source_counts.values())
    if total <= 0:
        return 0.0
    concentration = sum((count / total) ** 2 for count in source_counts.values())
    if concentration <= 0:
        return 0.0
    return 1.0 / concentration


def build_payload(
    gaps: list[LabelGap],
    rows: list[dict[str, Any]],
    *,
    db_path: Path,
    target_sources: int,
    target_per_source: int,
    max_dominant_share: float,
) -> dict[str, Any]:
    total = sum(gap.total for gap in gaps)
    generator_gaps = [gap for gap in gaps if gap.label != REAL_LABEL]
    weak = [
        gap
        for gap in generator_gaps
        if gap.status in {"critical-single-source", "needs-more-sources", "source-dominated"}
    ]
    status_counts = Counter(gap.status for gap in gaps)
    return {
        "db_path": str(db_path),
        "task_type": GENERATOR_TASK,
        "total_image_samples": total,
        "label_count": len(gaps),
        "generator_label_count": len(generator_gaps),
        "status_distribution": dict(sorted(status_counts.items())),
        "target_sources": target_sources,
        "target_per_source": target_per_source,
        "max_dominant_share": max_dominant_share,
        "labels": [gap_to_dict(gap) for gap in gaps],
        "weak_labels": [gap_to_dict(gap) for gap in weak],
        "public_dataset_options": PUBLIC_DATASET_OPTIONS,
        "local_baseline_options": local_baseline_statuses(),
        "label_dataset_recommendations": LABEL_DATASET_RECOMMENDATIONS,
        "source_rows": rows,
        "strategy": {
            "gpt_image2_focus": (
                "不要用每类 200 张限制 GPT-image2 专项；该轨道应尽量使用可追溯的有效 GPT-image2，"
                "再用来源互留报告 Qwen/Scam-AI/本地池之间的召回和精度。"
            ),
            "multi_generator_attribution": (
                "多模型归因不是按总量堆图，而是先让每个强归因标签覆盖至少 3 个独立来源；"
                "训练采样时对单一来源设软上限/权重，是为了避免把 dataset_source 当成 generator 指纹。"
            ),
            "baseline_reuse": (
                "UnivFD/AIGCDetectBenchmark/Synthbuster 等可作为二分类或频域/CLIP 特征 baseline；"
                "不能把它们的 real/fake 指标直接写成具体生成器归因指标。"
            ),
        },
    }


def gap_to_dict(gap: LabelGap) -> dict[str, Any]:
    return {
        "label": gap.label,
        "total": gap.total,
        "source_count": gap.source_count,
        "largest_source_count": gap.largest_source_count,
        "largest_source_share": round(gap.largest_source_share, 4),
        "effective_source_count": round(gap.effective_source_count, 2),
        "status": gap.status,
        "recommended_new_sources": gap.recommended_new_sources,
        "recommended_samples": gap.recommended_samples,
        "recommended_datasets": LABEL_DATASET_RECOMMENDATIONS.get(gap.label, []),
        "top_sources": [{"source": source, "count": count} for source, count in gap.top_sources],
    }


def local_baseline_statuses() -> list[dict[str, str]]:
    statuses: list[dict[str, str]] = []
    for option in LOCAL_BASELINE_OPTIONS:
        relative_path = Path(option["path"])
        absolute_path = PROJECT_ROOT / relative_path
        status = option["status"] if absolute_path.exists() else "not-local"
        statuses.append({**option, "absolute_path": str(absolute_path), "status": status})
    return statuses


def write_markdown(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# 多生成器归因数据缺口报告",
        "",
        "本报告只审计 `vision_generator_attribution` 当前外部图片池，不训练、不激活模型。",
        "",
        "## 结论",
        "",
        (
            f"- 当前可用图片样本 `{payload['total_image_samples']}` 张，"
            f"生成器标签 `{payload['generator_label_count']}` 个。"
        ),
        (
            "- 多模型归因的主要瓶颈是来源覆盖和来源耦合，不是简单的总图片数。"
        ),
        (
            "- `max-per-label`/`max-per-class` 只能作为下载安全阀；"
            "GPT-image2 专项不应被 200 张这类冒烟上限限制。"
        ),
        (
            "- 多模型归因训练应先补到每个强归因标签至少 "
            f"`{payload['target_sources']}` 个独立来源，目标约 "
            f"`{payload['target_per_source']}` 张/来源；这是规划目标，不是硬上限。"
        ),
        "",
        "## 标签覆盖表",
        "",
        "| 标签 | 样本数 | 来源数 | 有效来源数 | 最大来源占比 | 状态 | 建议新增来源 | 建议新增样本 | 推荐补源 |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for gap in payload["labels"]:
        recommendations = ", ".join(gap["recommended_datasets"]) or "-"
        lines.append(
            "| {label} | {total} | {source_count} | {effective_source_count:.2f} | {share:.1%} | {status} | "
            "{recommended_new_sources} | {recommended_samples} | {recommendations} |".format(
                label=gap["label"],
                total=gap["total"],
                source_count=gap["source_count"],
                effective_source_count=gap["effective_source_count"],
                share=gap["largest_source_share"],
                status=gap["status"],
                recommended_new_sources=gap["recommended_new_sources"],
                recommended_samples=gap["recommended_samples"],
                recommendations=recommendations,
            )
        )
    lines.extend(
        [
            "",
            "## 主要问题标签",
            "",
            "| 标签 | 问题 | Top 来源 |",
            "| --- | --- | --- |",
        ]
    )
    for gap in payload["weak_labels"]:
        top_sources = "<br>".join(f"{item['source']} = {item['count']}" for item in gap["top_sources"])
        lines.append(f"| {gap['label']} | {gap['status']} | {top_sources} |")

    lines.extend(
        [
            "",
            "## 公开数据和 baseline 选择",
            "",
            "| 名称 | 适配度 | 用法 | 标签帮助 | 注意事项 | 链接 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for option in payload["public_dataset_options"]:
        lines.append(
            "| {name} | {fit} | {use} | {labels} | {caution} | {url} |".format(
                name=option["name"],
                fit=option["fit"],
                use=option["use"],
                labels=option["labels"],
                caution=option["caution"],
                url=option["url"],
            )
        )
    lines.extend(
        [
            "",
            "## 本地 baseline 落地状态",
            "",
            "| baseline | 本地路径 | 当前状态 | 可借鉴点 | 不解决的问题 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for option in payload["local_baseline_options"]:
        lines.append(
            "| {name} | `{absolute_path}` | {status} | {use} | {limit} |".format(
                name=option["name"],
                absolute_path=option["absolute_path"],
                status=option["status"],
                use=option["use"],
                limit=option["limit"],
            )
        )
    lines.extend(
        [
            "",
            "## 下一步执行顺序",
            "",
            "1. 多生成器归因先收束为 `mainstream_five_attribution`：GPT-image2、Nano Banana、豆包/Seedream、Stable Diffusion、Midjourney。",
            "2. 优先补五类主流标签的跨 dataset_source 覆盖，暂不继续扩展 DALL-E、Flux、Imagen、Firefly 等长尾归因。",
            "3. GPT-image2、Nano Banana、Seedream 尽量使用全部有效样本，并做来源互留；Stable Diffusion 系列合并评估。",
            "4. 新数据先以 source-holdout/benchmark 角色导入，指标稳定后再进入训练；不把 clean 高分写成跨来源泛化。",
            "5. 现成 baseline 先复用 UnivFD/AIGCDetectBenchmark/Synthbuster 的思路和评测，不把二分类 baseline 分数冒充多类归因。",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
