from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_JSON = ROOT / "output" / "audits" / "platform_transcode_artifacts_latest.json"
DEFAULT_CANDIDATE_JSON = ROOT / "output" / "audits" / "platform_like_augmentation_candidate_latest.json"
DEFAULT_OUTPUT_JSON = ROOT / "output" / "audits" / "platform_transcode_enhancement_comparison_latest.json"
DEFAULT_OUTPUT_MD = ROOT / "output" / "audits" / "platform_transcode_enhancement_comparison_latest.md"
DEFAULT_OUTPUT_CSV = ROOT / "output" / "audits" / "platform_transcode_enhancement_comparison_latest.csv"
DEFAULT_DOC_MD = ROOT / "docs" / "platform_transcode_enhancement_comparison.md"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a before/after comparison report for platform transcode analysis-driven augmentation."
    )
    parser.add_argument("--artifacts-json", default=str(DEFAULT_ARTIFACTS_JSON))
    parser.add_argument("--candidate-json", default=str(DEFAULT_CANDIDATE_JSON))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_OUTPUT_MD))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--doc-md", default=str(DEFAULT_DOC_MD))
    args = parser.parse_args()

    artifacts = load_json(Path(args.artifacts_json))
    candidate = load_json(Path(args.candidate_json))
    payload = build_payload(artifacts, candidate)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_markdown(payload)
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown, encoding="utf-8")
    doc_md = Path(args.doc_md)
    doc_md.parent.mkdir(parents=True, exist_ok=True)
    doc_md.write_text(markdown, encoding="utf-8")
    write_csv(Path(args.output_csv), payload["metric_comparison"])

    print(json.dumps(payload["headline"], ensure_ascii=False, indent=2))
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    print(f"wrote {Path(args.output_csv)}")
    print(f"wrote {doc_md}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"input not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_payload(artifacts: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    summaries = candidate.get("condition_summaries", {})
    active = summaries.get("active", {})
    enhanced = summaries.get("candidate", {})
    rows = []
    for condition in ("clean", "weibo_download", "xhs_download"):
        before = active.get(condition, {})
        after = enhanced.get(condition, {})
        if not before or not after:
            continue
        rows.append(
            {
                "condition": condition,
                "sample_count": after.get("sample_count", before.get("sample_count")),
                "before_model": candidate.get("active_model_id_before"),
                "after_model": candidate.get("candidate_id"),
                "before_gpt_image2_recall": num(before.get("gpt_image2_recall")),
                "after_gpt_image2_recall": num(after.get("gpt_image2_recall")),
                "delta_gpt_image2_recall": delta(after.get("gpt_image2_recall"), before.get("gpt_image2_recall")),
                "before_real_fpr": num(before.get("real_false_positive_rate")),
                "after_real_fpr": num(after.get("real_false_positive_rate")),
                "delta_real_fpr": delta(after.get("real_false_positive_rate"), before.get("real_false_positive_rate")),
                "before_binary_macro_f1": num(before.get("binary_macro_f1")),
                "after_binary_macro_f1": num(after.get("binary_macro_f1")),
                "delta_binary_macro_f1": delta(after.get("binary_macro_f1"), before.get("binary_macro_f1")),
                "before_gpt_auc": num(before.get("gpt_image2_auc")),
                "after_gpt_auc": num(after.get("gpt_image2_auc")),
                "delta_gpt_auc": delta(after.get("gpt_image2_auc"), before.get("gpt_image2_auc")),
            }
        )

    condition_summary = artifacts.get("condition_summary", {})
    weibo = compact_artifact(condition_summary.get("weibo_download", {}))
    xhs = compact_artifact(condition_summary.get("xhs_download", {}))
    screenshot = compact_artifact(condition_summary.get("weibo_screenshot", {}))
    weibo_row = next((row for row in rows if row["condition"] == "weibo_download"), {})
    headline = {
        "candidate_id": candidate.get("candidate_id"),
        "active_model_id": candidate.get("active_model_id_before"),
        "active_unchanged": candidate.get("active_unchanged"),
        "weibo_download_recall_before": weibo_row.get("before_gpt_image2_recall"),
        "weibo_download_recall_after": weibo_row.get("after_gpt_image2_recall"),
        "weibo_download_recall_delta": weibo_row.get("delta_gpt_image2_recall"),
        "weibo_download_fpr_before": weibo_row.get("before_real_fpr"),
        "weibo_download_fpr_after": weibo_row.get("after_real_fpr"),
        "official_operating_point": candidate.get("official_operating_point", {}).get("method"),
    }
    return {
        "id": f"platform-transcode-enhancement-comparison-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        "created_at": datetime.now(UTC).isoformat(),
        "headline": headline,
        "research_claim": (
            "Real platform returned samples are used to infer observable download/transcode artifacts; "
            "the inferred perturbation is synthesized on the larger external pool, and the real returned "
            "download set is reserved for evaluation."
        ),
        "artifact_observations": {
            "weibo_download": {
                **weibo,
                "interpretation": "dimension-preserving JPEG re-encoding; suitable for synthetic platform-like augmentation",
            },
            "xhs_download": {
                **xhs,
                "interpretation": "mostly clean-equivalent in this creator-backend collection; useful as a stability check",
            },
            "weibo_screenshot": {
                **screenshot,
                "interpretation": "low-resolution PNG thumbnail/screenshot chain; parked as a failure boundary rather than main objective",
            },
        },
        "enhancement_design": {
            "training_uses_real_platform_returned_files": False,
            "training_pool": "larger external GPT-image2/real/other-generated pool",
            "synthetic_augmentation_conditions": candidate.get("training_policy", {}).get("augmentation_conditions", []),
            "platform_eval_conditions": ["clean", "weibo_download", "xhs_download"],
            "operating_point": candidate.get("official_operating_point", {}),
        },
        "metric_comparison": rows,
        "limitations": [
            "The paired platform set is small, so the result is a bounded black-box test rather than a platform-wide guarantee.",
            "The method infers observable artifacts from returned samples; it does not claim official Weibo/XHS codec rules.",
            "Screenshot propagation is intentionally excluded from the main objective because the recovered files are low-resolution rendered thumbnails and xhs_screenshot is unavailable.",
            "The candidate remains component-only; active model is not automatically replaced.",
        ],
    }


def compact_artifact(summary: dict[str, Any]) -> dict[str, Any]:
    paired = summary.get("paired") if isinstance(summary, dict) else {}
    if not isinstance(paired, dict):
        paired = {}
    return {
        "count": summary.get("count", 0),
        "formats": summary.get("format_distribution", {}),
        "same_sha256": paired.get("same_sha256", 0),
        "same_dimensions": paired.get("same_dimensions", 0),
        "median_byte_ratio": nested(paired, "byte_ratio", "median"),
        "median_area_ratio": nested(paired, "area_ratio", "median"),
        "jpeg_qtable_changed": paired.get("jpeg_qtable_changed", 0),
    }


def nested(mapping: dict[str, Any], key: str, subkey: str) -> float:
    value = mapping.get(key)
    if isinstance(value, dict) and isinstance(value.get(subkey), int | float):
        return round(float(value[subkey]), 3)
    return 0.0


def num(value: object) -> float | None:
    if isinstance(value, int | float):
        return round(float(value), 3)
    return None


def delta(after: object, before: object) -> float | None:
    if isinstance(after, int | float) and isinstance(before, int | float):
        return round(float(after) - float(before), 3)
    return None


def render_markdown(payload: dict[str, Any]) -> str:
    headline = payload["headline"]
    weibo_row = next((row for row in payload["metric_comparison"] if row["condition"] == "weibo_download"), {})
    xhs_row = next((row for row in payload["metric_comparison"] if row["condition"] == "xhs_download"), {})
    weibo_before = float(weibo_row.get("before_gpt_image2_recall") or 0.0)
    weibo_after = float(weibo_row.get("after_gpt_image2_recall") or 0.0)
    weibo_fpr_after = float(weibo_row.get("after_real_fpr") or 0.0)
    xhs_after = float(xhs_row.get("after_gpt_image2_recall") or 0.0)
    xhs_fpr_after = float(xhs_row.get("after_real_fpr") or 0.0)
    lines = [
        "# Platform Transcode Enhancement Comparison",
        "",
        f"- Created at: `{payload['created_at']}`",
        f"- Active model: `{headline['active_model_id']}`",
        f"- Enhancement candidate: `{headline['candidate_id']}`",
        f"- Active unchanged: `{headline['active_unchanged']}`",
        f"- Operating point: `{headline['official_operating_point']}`",
        "",
        "## Why This Matters",
        "",
        payload["research_claim"],
        "",
        "## Platform Artifact Findings",
        "",
        "| Condition | N | Formats | Same SHA | Same Dim | Median Byte Ratio | Median Area Ratio | JPEG QTable Changed | Interpretation |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for condition in ("weibo_download", "xhs_download", "weibo_screenshot"):
        row = payload["artifact_observations"][condition]
        lines.append(
            "| {condition} | {count} | {formats} | {same_sha} | {same_dim} | {byte_ratio:.3f} | {area_ratio:.3f} | {qtable} | {interp} |".format(
                condition=condition,
                count=row["count"],
                formats=json.dumps(row["formats"], ensure_ascii=False),
                same_sha=row["same_sha256"],
                same_dim=row["same_dimensions"],
                byte_ratio=float(row["median_byte_ratio"]),
                area_ratio=float(row["median_area_ratio"]),
                qtable=row["jpeg_qtable_changed"],
                interp=row["interpretation"],
            )
        )
    lines.extend(
        [
            "",
            "## Before vs After",
            "",
            "| Condition | N | Recall Before | Recall After | Recall Delta | FPR Before | FPR After | F1 Before | F1 After | AUC Before | AUC After |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["metric_comparison"]:
        lines.append(
            "| {condition} | {n} | {rb:.3f} | {ra:.3f} | {rd:+.3f} | {fb:.3f} | {fa:.3f} | {f1b:.3f} | {f1a:.3f} | {ab:.3f} | {aa:.3f} |".format(
                condition=row["condition"],
                n=int(row["sample_count"] or 0),
                rb=float(row["before_gpt_image2_recall"] or 0.0),
                ra=float(row["after_gpt_image2_recall"] or 0.0),
                rd=float(row["delta_gpt_image2_recall"] or 0.0),
                fb=float(row["before_real_fpr"] or 0.0),
                fa=float(row["after_real_fpr"] or 0.0),
                f1b=float(row["before_binary_macro_f1"] or 0.0),
                f1a=float(row["after_binary_macro_f1"] or 0.0),
                ab=float(row["before_gpt_auc"] or 0.0),
                aa=float(row["after_gpt_auc"] or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Academic Takeaway",
            "",
            "- The improvement is not only from adding more data; it comes from using the platform returned set to identify the actual download/transcode artifact family, then synthesizing that perturbation on the larger pool.",
            f"- Weibo download is the clearest example: dimension-preserving JPEG re-encoding was observed, and the enhanced candidate improves GPT-image2 recall from `{weibo_before:.3f}` to `{weibo_after:.3f}` on the held-out paired Weibo download test while keeping real FPR at `{weibo_fpr_after:.3f}` in this set.",
            f"- XHS download acts as a stability check because this collection is mostly clean-equivalent; the enhanced candidate reaches `{xhs_after:.3f}` recall there with real FPR `{xhs_fpr_after:.3f}`.",
            "- Screenshot chains remain outside the main claim and should be written as a limitation/future-work branch.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["limitations"])
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "sample_count",
        "before_model",
        "after_model",
        "before_gpt_image2_recall",
        "after_gpt_image2_recall",
        "delta_gpt_image2_recall",
        "before_real_fpr",
        "after_real_fpr",
        "delta_real_fpr",
        "before_binary_macro_f1",
        "after_binary_macro_f1",
        "delta_binary_macro_f1",
        "before_gpt_auc",
        "after_gpt_auc",
        "delta_gpt_auc",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


if __name__ == "__main__":
    main()
