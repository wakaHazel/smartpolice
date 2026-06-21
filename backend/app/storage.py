from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
import re
import shutil
import sqlite3
from pathlib import Path
from uuid import uuid4

from app.models import (
    AgentMetrics,
    AgentModelRoute,
    AgentRunRecord,
    CaseAsset,
    CaseCreateRequest,
    CaseEvidenceBundle,
    CaseLabelRequest,
    CaseSample,
    ExternalDatasetSourceSummary,
    ExternalTrainingSample,
    FeatureCacheRecord,
    FusionTrainingRunResult,
    FullAnalysis,
    ImageForensicsResult,
    KnowledgeSearchResult,
    ModelInvocationAudit,
    LocalVisionCalibrationRunResult,
    RealCaseAnalysisResult,
    TamperForensicsResult,
    TrainingDataStatus,
    TrainingTaskStatus,
    TrainingRunResult,
    UsageCount,
    VisionTrainingRunRecord,
    VisionTrainingRunResult,
    WebEvidenceSnapshot,
)
from app.sample_data import DEMO_CASES, TAMPER_DEMO_CASES

DB_PATH = Path(
    os.getenv(
        "SMARTPOLICE_DB_PATH",
        str(Path(__file__).resolve().parents[1] / "data" / "smartpolice.db"),
    )
)
SUPPORTED_PROVIDERS = {"LocalReview", "LocalReport", "LocalVision", "DeepSeek", "MiniMax"}

KNOWLEDGE_SEED = [
    {
        "id": "policy-ai-labeling",
        "title": "AI生成合成内容标识与公共安全谣言治理",
        "source": "项目规则模板",
        "category": "政策法规",
        "content": (
            "生成式人工智能、深度合成和AI生成合成内容标识相关要求强调，"
            "不得利用生成合成内容制作、复制、发布虚假有害信息。公共安全类疑似谣言"
            "应保留原帖、图片、链接、传播节点和模型辅助研判记录，最终结论由人工复核。"
        ),
    },
    {
        "id": "workflow-public-safety-rumor",
        "title": "公共安全谣言闭环处置流程",
        "source": "项目处置规范",
        "category": "处置流程",
        "content": (
            "公共安全谣言处置遵循发现风险、固定证据、核验来源、构建证据链、"
            "风险分级、平台协查、属地联动、公开澄清和复盘评估的闭环。较高及紧急风险"
            "应优先核查线下扰动、群体对立和应急资源挤占风险。"
        ),
    },
    {
        "id": "review-boundary",
        "title": "模型辅助研判边界",
        "source": "项目报告材料",
        "category": "人工复核",
        "content": (
            "大模型只生成辅助研判草稿，不生成最终执法结论。风险理由必须引用证据链或"
            "知识库依据；证据不足时应输出需补充核查，避免把谣言性质或模型来源写成最终确认。"
        ),
    },
    {
        "id": "evidence-chain-template",
        "title": "多模态证据链结构",
        "source": "项目报告材料",
        "category": "证据链",
        "content": (
            "证据链应覆盖内容证据、图像证据、来源证据、传播证据、权威依据和模型分析结论，"
            "并说明支撑、反驳、补充、矛盾或需核查关系。报告中应保留证据编号、来源和置信度。"
        ),
    },
    {
        "id": "report-template",
        "title": "研判报告结构模板",
        "source": "项目报告材料",
        "category": "报告模板",
        "content": (
            "报告建议包含事件概况、核心主张、多模态分析、生成模型来源归因、证据链摘要、"
            "风险评估、风险推演、处置建议和人工复核声明。措辞应审慎，区分疑似、待核查和已核实。"
        ),
    },
]


def initialize_database() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS case_samples (
                id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS case_deletions (
                id TEXT PRIMARY KEY,
                deleted_at TEXT NOT NULL
            )
            """
        )
        tamper_demo_ids = {item.id for item in TAMPER_DEMO_CASES}
        for case in [*DEMO_CASES, *TAMPER_DEMO_CASES]:
            deleted = connection.execute(
                "SELECT 1 FROM case_deletions WHERE id = ?",
                (case.id,),
            ).fetchone()
            if deleted is not None:
                continue
            connection.execute(
                """
                INSERT OR IGNORE INTO case_samples (id, payload)
                VALUES (?, ?)
                """,
                (case.id, case.model_dump_json()),
            )
            if case.id in tamper_demo_ids:
                connection.execute(
                    """
                    UPDATE case_samples
                    SET payload = ?
                    WHERE id = ?
                    """,
                    (case.model_dump_json(), case.id),
                )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_runs (
                id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                risk_score INTEGER NOT NULL,
                model_routes TEXT NOT NULL,
                skill_names TEXT NOT NULL,
                estimated_cost_units INTEGER NOT NULL,
                primary_strategy TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS training_runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                artifact TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS local_vision_training_runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                artifact TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS external_training_samples (
                id TEXT PRIMARY KEY,
                dataset_name TEXT NOT NULL,
                source TEXT NOT NULL,
                source_url TEXT,
                task_type TEXT NOT NULL DEFAULT 'text_risk',
                split TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                image_path TEXT,
                image_url TEXT,
                image_sha256 TEXT,
                image_available INTEGER NOT NULL DEFAULT 0,
                label TEXT NOT NULL,
                risk_score INTEGER NOT NULL,
                scenario TEXT NOT NULL,
                raw_payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        _ensure_columns(
            connection,
            "external_training_samples",
            {
                "task_type": "TEXT NOT NULL DEFAULT 'text_risk'",
                "image_path": "TEXT",
                "image_url": "TEXT",
                "image_sha256": "TEXT",
                "image_available": "INTEGER NOT NULL DEFAULT 0",
            },
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_cache (
                id TEXT PRIMARY KEY,
                cache_key TEXT NOT NULL UNIQUE,
                extractor_version TEXT NOT NULL,
                modality TEXT NOT NULL,
                sha256 TEXT,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS vision_training_runs (
                id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                artifact TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS fusion_training_runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                artifact TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_invocations (
                id TEXT PRIMARY KEY,
                case_id TEXT,
                provider TEXT NOT NULL,
                role TEXT NOT NULL,
                model TEXT NOT NULL,
                status TEXT NOT NULL,
                request_payload TEXT NOT NULL,
                response_text TEXT,
                error TEXT,
                latency_ms INTEGER NOT NULL,
                token_usage TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                category TEXT NOT NULL,
                content TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS case_assets (
                id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                content_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                width INTEGER,
                height INTEGER,
                sha256 TEXT NOT NULL,
                storage_path TEXT NOT NULL,
                preview_url TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS image_forensics_runs (
                case_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tamper_forensics_runs (
                case_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS real_analysis_runs (
                case_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS web_snapshots (
                id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                requested_url TEXT NOT NULL,
                final_url TEXT NOT NULL,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                html_path TEXT NOT NULL,
                text_path TEXT NOT NULL,
                screenshot_path TEXT,
                screenshot_url TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evidence_items (
                id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                supports TEXT NOT NULL,
                artifact_id TEXT,
                source_url TEXT,
                sha256 TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts
            USING fts5(id UNINDEXED, title, source, category, content)
            """
        )
        knowledge_count = connection.execute(
            "SELECT COUNT(*) FROM knowledge_documents"
        ).fetchone()[0]
        if int(knowledge_count) == 0:
            connection.executemany(
                """
                INSERT INTO knowledge_documents (id, title, source, category, content)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        item["id"],
                        item["title"],
                        item["source"],
                        item["category"],
                        item["content"],
                    )
                    for item in KNOWLEDGE_SEED
                ],
            )
        _seed_generation_demo_assets(connection)
        _seed_tamper_demo_assets(connection)
        _rebuild_knowledge_fts(connection)
        connection.commit()


def create_case_sample(payload: CaseCreateRequest) -> CaseSample:
    initialize_database()
    case_id = _case_id(payload)
    case = CaseSample(
        id=case_id,
        title=payload.title.strip(),
        scenario=payload.scenario.strip(),
        platform=payload.platform.strip(),
        publish_time=payload.publish_time.strip(),
        source_url=payload.source_url.strip() or "本地录入样本",
        content=payload.content.strip(),
        image_description=payload.image_description.strip(),
        spread=payload.spread,
        manual_label=payload.manual_label.strip() or "待人工复核",
        manual_risk_score=payload.manual_risk_score,
        tags=[tag.strip() for tag in payload.tags if tag.strip()],
        sensitivity_notes=payload.sensitivity_notes.strip(),
        review_note=payload.review_note.strip(),
        created_by_user=True,
    )
    with sqlite3.connect(DB_PATH) as connection:
        exists = connection.execute(
            "SELECT 1 FROM case_samples WHERE id = ?",
            (case.id,),
        ).fetchone()
        if exists is not None:
            raise ValueError(f"Case already exists: {case.id}")
        connection.execute(
            "INSERT INTO case_samples (id, payload) VALUES (?, ?)",
            (case.id, case.model_dump_json()),
        )
        connection.execute("DELETE FROM case_deletions WHERE id = ?", (case.id,))
        connection.commit()
    return case


def update_case_label(case_id: str, payload: CaseLabelRequest) -> CaseSample:
    case = load_case_sample(case_id)
    updated = case.model_copy(
        update={
            "manual_risk_score": payload.manual_risk_score,
            "manual_label": payload.manual_label.strip() or case.manual_label,
            "review_note": payload.review_note.strip(),
        }
    )
    save_case_sample(updated)
    delete_real_analysis_result(case_id)
    return updated


def save_case_sample(case: CaseSample) -> None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO case_samples (id, payload)
            VALUES (?, ?)
            """,
            (case.id, case.model_dump_json()),
        )
        connection.commit()


def list_case_samples() -> list[CaseSample]:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            "SELECT payload FROM case_samples ORDER BY id"
        ).fetchall()
    return [CaseSample.model_validate(json.loads(row[0])) for row in rows]


def load_case_sample(case_id: str) -> CaseSample:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute(
            "SELECT payload FROM case_samples WHERE id = ?",
            (case_id,),
        ).fetchone()
    if row is None:
        raise KeyError(case_id)
    return CaseSample.model_validate(json.loads(row[0]))


def delete_case_sample(case_id: str) -> CaseSample:
    initialize_database()
    data_root = DB_PATH.parent.resolve()
    upload_root = data_root / "uploads"
    snapshot_root = data_root / "snapshots"
    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute(
            "SELECT payload FROM case_samples WHERE id = ?",
            (case_id,),
        ).fetchone()
        if row is None:
            raise KeyError(case_id)
        deleted_case = CaseSample.model_validate(json.loads(row[0]))
        deleted_at = datetime.now(UTC).isoformat()
        connection.execute(
            """
            INSERT OR REPLACE INTO case_deletions (id, deleted_at)
            VALUES (?, ?)
            """,
            (case_id, deleted_at),
        )
        connection.execute(
            """
            DELETE FROM knowledge_documents
            WHERE id IN (
                SELECT 'snapshot-' || id FROM web_snapshots WHERE case_id = ?
            )
            """,
            (case_id,),
        )
        connection.execute(
            """
            DELETE FROM knowledge_documents
            WHERE id IN (
                SELECT 'evidence-' || id FROM evidence_items WHERE case_id = ?
            )
            """,
            (case_id,),
        )
        for table in (
            "agent_runs",
            "llm_invocations",
            "case_assets",
            "image_forensics_runs",
            "tamper_forensics_runs",
            "real_analysis_runs",
            "web_snapshots",
            "evidence_items",
            "case_samples",
        ):
            connection.execute(f"DELETE FROM {table} WHERE case_id = ?" if table != "case_samples" else "DELETE FROM case_samples WHERE id = ?", (case_id,))
        _rebuild_knowledge_fts(connection)
        connection.commit()
    _remove_case_directory(upload_root, case_id)
    _remove_case_directory(snapshot_root, case_id)
    return deleted_case


def filter_new_external_training_samples(samples: list[ExternalTrainingSample]) -> list[ExternalTrainingSample]:
    initialize_database()
    if not samples:
        return []
    filtered: list[ExternalTrainingSample] = []
    seen_keys: set[tuple[str, str]] = set()
    with sqlite3.connect(DB_PATH) as connection:
        existing_keys = {
            (str(row[0]), str(row[1]))
            for row in connection.execute(
                """
                SELECT task_type, image_sha256
                FROM external_training_samples
                WHERE image_sha256 IS NOT NULL AND image_sha256 != ''
                """
            ).fetchall()
        }
        for sample in samples:
            sha = (sample.image_sha256 or "").strip()
            if sha:
                key = (sample.task_type, sha)
                if key in existing_keys or key in seen_keys:
                    continue
                seen_keys.add(key)
            filtered.append(sample)
    return filtered


def save_external_training_samples(samples: list[ExternalTrainingSample]) -> int:
    initialize_database()
    filtered = filter_new_external_training_samples(samples)
    if not filtered:
        return 0
    with sqlite3.connect(DB_PATH) as connection:
        connection.executemany(
            """
            INSERT OR REPLACE INTO external_training_samples (
                id,
                dataset_name,
                source,
                source_url,
                task_type,
                split,
                title,
                content,
                image_path,
                image_url,
                image_sha256,
                image_available,
                label,
                risk_score,
                scenario,
                raw_payload,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    sample.id,
                    sample.dataset_name,
                    sample.source,
                    sample.source_url,
                    sample.task_type,
                    sample.split,
                    sample.title,
                    sample.content,
                    sample.image_path,
                    sample.image_url,
                    sample.image_sha256,
                    1 if sample.image_available else 0,
                    sample.label,
                    sample.risk_score,
                    sample.scenario,
                    json.dumps(sample.raw_payload, ensure_ascii=False),
                    sample.created_at,
                )
                for sample in filtered
            ],
        )
        connection.commit()
    return len(filtered)


def delete_external_training_samples_by_source_patterns(
    *,
    task_type: str,
    patterns: list[str],
) -> int:
    initialize_database()
    cleaned_patterns = [pattern.strip().lower() for pattern in patterns if pattern.strip()]
    if not cleaned_patterns:
        return 0
    clauses = []
    params: list[object] = []
    for pattern in cleaned_patterns:
        like_pattern = f"%{pattern}%"
        clauses.append(
            """
            (
                lower(dataset_name) LIKE ?
                OR lower(source) LIKE ?
                OR lower(coalesce(source_url, '')) LIKE ?
                OR lower(label) LIKE ?
            )
            """
        )
        params.extend([like_pattern, like_pattern, like_pattern, like_pattern])
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute(
            f"""
            DELETE FROM external_training_samples
            WHERE task_type = ? AND ({' OR '.join(clauses)})
            """,
            (task_type, *params),
        )
        connection.commit()
        return int(cursor.rowcount if cursor.rowcount is not None else 0)


def list_external_training_samples(
    limit: int = 10000,
    dataset_name: str | None = None,
    task_type: str | None = None,
) -> list[ExternalTrainingSample]:
    initialize_database()
    safe_limit = max(1, min(limit, 50000))
    with sqlite3.connect(DB_PATH) as connection:
        clauses: list[str] = []
        params: list[object] = []
        if dataset_name:
            clauses.append("dataset_name = ?")
            params.append(dataset_name)
        if task_type:
            clauses.append("task_type = ?")
            params.append(task_type)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = connection.execute(
            f"""
            SELECT
                id,
                dataset_name,
                source,
                source_url,
                task_type,
                split,
                title,
                content,
                image_path,
                image_url,
                image_sha256,
                image_available,
                label,
                risk_score,
                scenario,
                raw_payload,
                created_at
            FROM external_training_samples
            {where}
            ORDER BY created_at DESC, id
            LIMIT ?
            """,
            (*params, safe_limit),
        ).fetchall()
    return [_external_training_sample_from_row(row) for row in rows]


def count_external_training_samples(task_type: str | None = None) -> int:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        if task_type:
            row = connection.execute(
                "SELECT COUNT(*) FROM external_training_samples WHERE task_type = ?",
                (task_type,),
            ).fetchone()
        else:
            row = connection.execute(
                "SELECT COUNT(*) FROM external_training_samples"
            ).fetchone()
    return int(row[0]) if row else 0


def list_labeled_user_case_samples() -> list[CaseSample]:
    return [
        case
        for case in list_case_samples()
        if case.created_by_user and case.manual_risk_score is not None
    ]


def get_external_training_sample_count() -> int:
    return count_external_training_samples()


def get_training_data_status() -> TrainingDataStatus:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        external_count = int(
            connection.execute("SELECT COUNT(*) FROM external_training_samples").fetchone()[0]
        )
        source_rows = connection.execute(
            """
            SELECT
                dataset_name,
                source,
                MAX(source_url),
                task_type,
                COUNT(*),
                SUM(image_available),
                MAX(created_at)
            FROM external_training_samples
            GROUP BY dataset_name, source, task_type
            ORDER BY COUNT(*) DESC, dataset_name
            """
        ).fetchall()
        label_rows = connection.execute(
            """
            SELECT dataset_name, source, task_type, label, COUNT(*)
            FROM external_training_samples
            GROUP BY dataset_name, source, task_type, label
            """
        ).fetchall()
        task_rows = connection.execute(
            """
            SELECT task_type, COUNT(*), SUM(image_available)
            FROM external_training_samples
            GROUP BY task_type
            """
        ).fetchall()
        task_label_rows = connection.execute(
            """
            SELECT task_type, label, COUNT(*)
            FROM external_training_samples
            GROUP BY task_type, label
            """
        ).fetchall()
    labeled_user_count = len(list_labeled_user_case_samples())
    label_distributions: dict[tuple[str, str, str], dict[str, int]] = {}
    for row in label_rows:
        key = (str(row[0]), str(row[1]), str(row[2]))
        label_distributions.setdefault(key, {})[str(row[3])] = int(row[4])
    sources = [
        ExternalDatasetSourceSummary(
            dataset_name=str(row[0]),
            source=str(row[1]),
            source_url=str(row[2]) if row[2] is not None else None,
            task_type=str(row[3]),
            sample_count=int(row[4]),
            image_available_count=int(row[5] or 0),
            label_distribution=label_distributions.get((str(row[0]), str(row[1]), str(row[3])), {}),
            latest_import_at=str(row[6]) if row[6] is not None else None,
        )
        for row in source_rows
    ]
    task_counts = {str(row[0]): (int(row[1]), int(row[2] or 0)) for row in task_rows}
    task_label_distributions: dict[str, dict[str, int]] = {}
    for row in task_label_rows:
        task_label_distributions.setdefault(str(row[0]), {})[str(row[1])] = int(row[2])
    tasks = [
        _training_task_status(
            task_type=task_type,
            sample_count=task_counts.get(task_type, (0, 0))[0],
            image_available_count=task_counts.get(task_type, (0, 0))[1],
            label_distribution=task_label_distributions.get(task_type, {}),
            sources=[source for source in sources if source.task_type == task_type],
        )
        for task_type in _known_task_types()
    ]
    text_sample_count = task_counts.get("text_risk", (0, 0))[0]
    eligible = text_sample_count + labeled_user_count
    return TrainingDataStatus(
        external_sample_count=external_count,
        labeled_user_case_count=labeled_user_count,
        eligible_sample_count=eligible,
        demo_case_count=len(DEMO_CASES),
        training_ready=eligible >= 4,
        sources=sources,
        tasks=tasks,
        recommended_huggingface_datasets=_recommended_huggingface_datasets(),
        note=(
            "训练集来自外部数据集导入和人工标注用户案例；内置四方向样例只用于展示/评测，不参与训练。"
            if eligible >= 4
            else "请先导入 HuggingFace/本地外部数据集，或录入并人工标注至少 4 条用户案例；内置四方向样例不参与训练。"
        ),
    )


def save_training_run(result: TrainingRunResult, artifact: dict[str, object]) -> None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("UPDATE training_runs SET is_active = 0")
        connection.execute(
            """
            INSERT INTO training_runs (id, created_at, payload, artifact, is_active)
            VALUES (?, ?, ?, ?, 1)
            """,
            (
                result.id,
                result.created_at,
                result.model_dump_json(),
                json.dumps(artifact, ensure_ascii=False),
            ),
        )
        connection.commit()


def get_active_training_artifact() -> dict[str, object] | None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            """
            SELECT artifact FROM training_runs
            WHERE is_active = 1
            ORDER BY created_at DESC
            """
        ).fetchall()
    for row in rows:
        loaded = json.loads(row[0])
        if isinstance(loaded, dict) and _valid_training_artifact(loaded):
            return loaded
    return None


def get_latest_training_run() -> TrainingRunResult | None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            """
            SELECT payload, artifact FROM training_runs
            ORDER BY created_at DESC
            """
        ).fetchall()
    for row in rows:
        artifact = json.loads(row[1])
        if isinstance(artifact, dict) and _valid_training_artifact(artifact):
            return TrainingRunResult.model_validate(json.loads(row[0]))
    return None


def list_training_runs(limit: int = 10) -> list[TrainingRunResult]:
    initialize_database()
    safe_limit = max(1, min(limit, 100))
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            """
            SELECT payload, artifact FROM training_runs
            ORDER BY created_at DESC
            """
        ).fetchall()
    results: list[TrainingRunResult] = []
    for row in rows:
        artifact = json.loads(row[1])
        if not isinstance(artifact, dict) or not _valid_training_artifact(artifact):
            continue
        results.append(TrainingRunResult.model_validate(json.loads(row[0])))
        if len(results) >= safe_limit:
            break
    return results


def save_feature_cache(record: FeatureCacheRecord) -> None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO feature_cache (
                id,
                cache_key,
                extractor_version,
                modality,
                sha256,
                payload,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.cache_key,
                record.extractor_version,
                record.modality,
                record.sha256,
                json.dumps(record.payload, ensure_ascii=False),
                record.created_at,
            ),
        )
        connection.commit()


def get_feature_cache(cache_key: str) -> FeatureCacheRecord | None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute(
            """
            SELECT id, cache_key, extractor_version, modality, sha256, payload, created_at
            FROM feature_cache
            WHERE cache_key = ?
            """,
            (cache_key,),
        ).fetchone()
    if row is None:
        return None
    return FeatureCacheRecord(
        id=str(row[0]),
        cache_key=str(row[1]),
        extractor_version=str(row[2]),
        modality=str(row[3]),
        sha256=str(row[4]) if row[4] is not None else None,
        payload=dict(json.loads(str(row[5]))),
        created_at=str(row[6]),
    )


def save_vision_training_run(
    result: VisionTrainingRunResult,
    artifact: dict[str, object],
    *,
    activate: bool = True,
) -> None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        if activate:
            connection.execute(
                "UPDATE vision_training_runs SET is_active = 0 WHERE task_type = ?",
                (result.task_type,),
            )
        connection.execute(
            """
            INSERT INTO vision_training_runs (id, task_type, created_at, payload, artifact, is_active)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                result.id,
                result.task_type,
                result.created_at,
                result.model_dump_json(),
                json.dumps(artifact, ensure_ascii=False),
                1 if activate else 0,
            ),
        )
        connection.commit()


def get_active_vision_training_artifact(task_type: str) -> dict[str, object] | None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute(
            """
            SELECT artifact FROM vision_training_runs
            WHERE task_type = ? AND is_active = 1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (task_type,),
        ).fetchone()
    if row is None:
        return None
    loaded = json.loads(row[0])
    return loaded if isinstance(loaded, dict) else None


def get_vision_training_artifact_by_id(
    task_type: str,
    run_id: str,
) -> dict[str, object] | None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute(
            """
            SELECT artifact FROM vision_training_runs
            WHERE task_type = ? AND id = ?
            LIMIT 1
            """,
            (task_type, run_id),
        ).fetchone()
    if row is None:
        return None
    loaded = json.loads(row[0])
    return loaded if isinstance(loaded, dict) else None


def get_active_vision_training_run(task_type: str) -> VisionTrainingRunResult | None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute(
            """
            SELECT payload FROM vision_training_runs
            WHERE task_type = ? AND is_active = 1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (task_type,),
        ).fetchone()
    if row is None:
        return None
    return VisionTrainingRunResult.model_validate(json.loads(row[0]))


def get_latest_vision_training_run(task_type: str) -> VisionTrainingRunResult | None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute(
            """
            SELECT payload FROM vision_training_runs
            WHERE task_type = ?
            ORDER BY is_active DESC, created_at DESC
            LIMIT 1
            """,
            (task_type,),
        ).fetchone()
    if row is None:
        return None
    return VisionTrainingRunResult.model_validate(json.loads(row[0]))


def get_latest_vision_candidate_run(task_type: str) -> VisionTrainingRunResult | None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute(
            """
            SELECT payload FROM vision_training_runs
            WHERE task_type = ? AND is_active = 0
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (task_type,),
        ).fetchone()
    if row is None:
        return None
    return VisionTrainingRunResult.model_validate(json.loads(row[0]))


def list_vision_training_runs(
    task_type: str,
    limit: int = 10,
) -> list[VisionTrainingRunRecord]:
    initialize_database()
    safe_limit = max(1, min(limit, 100))
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            """
            SELECT payload, is_active FROM vision_training_runs
            WHERE task_type = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (task_type, safe_limit),
        ).fetchall()
    return [
        VisionTrainingRunRecord(
            run=VisionTrainingRunResult.model_validate(json.loads(str(row[0]))),
            is_active=bool(row[1]),
        )
        for row in rows
    ]


def activate_vision_training_run(
    task_type: str,
    run_id: str,
) -> tuple[VisionTrainingRunResult | None, str | None]:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        current = connection.execute(
            """
            SELECT id FROM vision_training_runs
            WHERE task_type = ? AND is_active = 1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (task_type,),
        ).fetchone()
        row = connection.execute(
            """
            SELECT payload FROM vision_training_runs
            WHERE task_type = ? AND id = ?
            LIMIT 1
            """,
            (task_type, run_id),
        ).fetchone()
        if row is None:
            return None, str(current[0]) if current is not None else None
        result = VisionTrainingRunResult.model_validate(json.loads(str(row[0])))
        result.status = "active_trained"
        connection.execute(
            "UPDATE vision_training_runs SET is_active = 0 WHERE task_type = ?",
            (task_type,),
        )
        connection.execute(
            """
            UPDATE vision_training_runs
            SET is_active = 1, payload = ?
            WHERE task_type = ? AND id = ?
            """,
            (result.model_dump_json(), task_type, run_id),
        )
        connection.commit()
    return result, str(current[0]) if current is not None else None


def update_vision_training_run_payload(result: VisionTrainingRunResult) -> None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            UPDATE vision_training_runs
            SET payload = ?
            WHERE task_type = ? AND id = ?
            """,
            (result.model_dump_json(), result.task_type, result.id),
        )
        connection.commit()


def save_fusion_training_run(
    result: FusionTrainingRunResult,
    artifact: dict[str, object],
) -> None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("UPDATE fusion_training_runs SET is_active = 0")
        connection.execute(
            """
            INSERT INTO fusion_training_runs (id, created_at, payload, artifact, is_active)
            VALUES (?, ?, ?, ?, 1)
            """,
            (
                result.id,
                result.created_at,
                result.model_dump_json(),
                json.dumps(artifact, ensure_ascii=False),
            ),
        )
        connection.commit()


def get_active_fusion_training_artifact() -> dict[str, object] | None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute(
            """
            SELECT artifact FROM fusion_training_runs
            WHERE is_active = 1
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        return None
    loaded = json.loads(row[0])
    return loaded if isinstance(loaded, dict) else None


def get_latest_fusion_training_run() -> FusionTrainingRunResult | None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute(
            """
            SELECT payload FROM fusion_training_runs
            ORDER BY is_active DESC, created_at DESC
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        return None
    return FusionTrainingRunResult.model_validate(json.loads(row[0]))


def save_local_vision_training_run(
    result: LocalVisionCalibrationRunResult,
    artifact: dict[str, object],
) -> None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("UPDATE local_vision_training_runs SET is_active = 0")
        connection.execute(
            """
            INSERT INTO local_vision_training_runs (id, created_at, payload, artifact, is_active)
            VALUES (?, ?, ?, ?, 1)
            """,
            (
                result.id,
                result.created_at,
                result.model_dump_json(),
                json.dumps(artifact, ensure_ascii=False),
            ),
        )
        connection.commit()


def get_active_local_vision_training_artifact() -> dict[str, object] | None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute(
            """
            SELECT artifact FROM local_vision_training_runs
            WHERE is_active = 1
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        return None
    loaded = json.loads(row[0])
    return loaded if isinstance(loaded, dict) else None


def get_latest_local_vision_training_run() -> LocalVisionCalibrationRunResult | None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute(
            """
            SELECT payload FROM local_vision_training_runs
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        return None
    return LocalVisionCalibrationRunResult.model_validate(json.loads(row[0]))


def list_local_vision_training_runs(
    limit: int = 10,
) -> list[LocalVisionCalibrationRunResult]:
    initialize_database()
    safe_limit = max(1, min(limit, 100))
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            """
            SELECT payload FROM local_vision_training_runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()
    return [
        LocalVisionCalibrationRunResult.model_validate(json.loads(row[0]))
        for row in rows
    ]


def record_agent_run(full_analysis: FullAnalysis) -> AgentRunRecord:
    initialize_database()
    agent = full_analysis.agent
    skill_names = [skill.name for skill in agent.recommended_skills]
    cost_units = sum(_route_cost_units(route) for route in agent.model_routes)
    record = AgentRunRecord(
        id=str(uuid4()),
        case_id=full_analysis.case.id,
        created_at=datetime.now(UTC).isoformat(),
        risk_level=full_analysis.risk.level,
        risk_score=full_analysis.risk.score,
        model_routes=agent.model_routes,
        skill_names=skill_names,
        estimated_cost_units=cost_units,
        primary_strategy=agent.primary_strategy,
    )
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO agent_runs (
                id,
                case_id,
                created_at,
                risk_level,
                risk_score,
                model_routes,
                skill_names,
                estimated_cost_units,
                primary_strategy
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.case_id,
                record.created_at,
                record.risk_level.value,
                record.risk_score,
                json.dumps(
                    [route.model_dump(mode="json") for route in record.model_routes],
                    ensure_ascii=False,
                ),
                json.dumps(record.skill_names, ensure_ascii=False),
                record.estimated_cost_units,
                record.primary_strategy,
            ),
        )
        connection.commit()
    return record


def list_agent_runs(limit: int = 10) -> list[AgentRunRecord]:
    initialize_database()
    safe_limit = max(1, min(limit, 100))
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                case_id,
                created_at,
                risk_level,
                risk_score,
                model_routes,
                skill_names,
                estimated_cost_units,
                primary_strategy
            FROM agent_runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()
    return [_supported_provider_run(_agent_run_from_row(row)) for row in rows]


def get_agent_metrics() -> AgentMetrics:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                case_id,
                created_at,
                risk_level,
                risk_score,
                model_routes,
                skill_names,
                estimated_cost_units,
                primary_strategy
            FROM agent_runs
            ORDER BY created_at DESC
            LIMIT 200
            """
        ).fetchall()
    runs = [_agent_run_from_row(row) for row in rows]
    total = len(runs)
    provider_counter: Counter[str] = Counter()
    skill_counter: Counter[str] = Counter()
    for run in runs:
        provider_counter.update(
            route.provider
            for route in run.model_routes
            if route.provider in SUPPORTED_PROVIDERS
        )
        skill_counter.update(run.skill_names)
    average_cost = (
        round(sum(run.estimated_cost_units for run in runs) / total, 2)
        if total
        else 0.0
    )
    return AgentMetrics(
        total_runs=total,
        average_cost_units=average_cost,
        high_risk_runs=sum(1 for run in runs if run.risk_level.value in {"较高", "紧急"}),
        provider_usage=[
            UsageCount(name=name, count=count)
            for name, count in provider_counter.most_common()
        ],
        skill_usage=[
            UsageCount(name=name, count=count)
            for name, count in skill_counter.most_common()
        ],
        recent_runs=[_supported_provider_run(run) for run in runs[:5]],
    )


def record_llm_invocation(audit: ModelInvocationAudit) -> None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO llm_invocations (
                id,
                case_id,
                provider,
                role,
                model,
                status,
                request_payload,
                response_text,
                error,
                latency_ms,
                token_usage,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                audit.id,
                audit.case_id,
                audit.provider,
                audit.role,
                audit.model,
                audit.status,
                json.dumps(audit.request_payload, ensure_ascii=False),
                audit.response_text,
                audit.error,
                audit.latency_ms,
                json.dumps(audit.token_usage, ensure_ascii=False),
                audit.created_at,
            ),
        )
        connection.commit()


def list_llm_invocations(
    case_id: str | None = None,
    limit: int = 20,
) -> list[ModelInvocationAudit]:
    initialize_database()
    safe_limit = max(1, min(limit, 100))
    with sqlite3.connect(DB_PATH) as connection:
        if case_id:
            rows = connection.execute(
                """
                SELECT
                    id,
                    case_id,
                    provider,
                    role,
                    model,
                    status,
                    request_payload,
                    response_text,
                    error,
                    latency_ms,
                    token_usage,
                    created_at
                FROM llm_invocations
                WHERE case_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (case_id, safe_limit),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT
                    id,
                    case_id,
                    provider,
                    role,
                    model,
                    status,
                    request_payload,
                    response_text,
                    error,
                    latency_ms,
                    token_usage,
                    created_at
                FROM llm_invocations
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
    return [_llm_invocation_from_row(row) for row in rows]


def save_case_asset(asset: CaseAsset) -> None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO case_assets (
                id,
                case_id,
                filename,
                content_type,
                size_bytes,
                width,
                height,
                sha256,
                storage_path,
                preview_url,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asset.id,
                asset.case_id,
                asset.filename,
                asset.content_type,
                asset.size_bytes,
                asset.width,
                asset.height,
                asset.sha256,
                asset.storage_path,
                asset.preview_url,
                asset.created_at,
            ),
        )
        connection.commit()


def list_case_assets(case_id: str) -> list[CaseAsset]:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                case_id,
                filename,
                content_type,
                size_bytes,
                width,
                height,
                sha256,
                storage_path,
                preview_url,
                created_at
            FROM case_assets
            WHERE case_id = ?
            ORDER BY created_at DESC
            """,
            (case_id,),
        ).fetchall()
    return [_case_asset_from_row(row) for row in rows]


def load_case_asset(asset_id: str) -> CaseAsset:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute(
            """
            SELECT
                id,
                case_id,
                filename,
                content_type,
                size_bytes,
                width,
                height,
                sha256,
                storage_path,
                preview_url,
                created_at
            FROM case_assets
            WHERE id = ?
            """,
            (asset_id,),
        ).fetchone()
    if row is None:
        raise KeyError(asset_id)
    return _case_asset_from_row(row)


def save_image_forensics_result(result: ImageForensicsResult) -> None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO image_forensics_runs (case_id, created_at, payload)
            VALUES (?, ?, ?)
            """,
            (
                result.case_id,
                datetime.now(UTC).isoformat(),
                result.model_dump_json(),
            ),
        )
        connection.commit()


def load_image_forensics_result(case_id: str) -> ImageForensicsResult | None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute(
            """
            SELECT payload
            FROM image_forensics_runs
            WHERE case_id = ?
            """,
            (case_id,),
        ).fetchone()
    if row is None:
        return None
    return ImageForensicsResult.model_validate(json.loads(str(row[0])))


def delete_image_forensics_result(case_id: str) -> None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("DELETE FROM image_forensics_runs WHERE case_id = ?", (case_id,))
        connection.commit()


def save_tamper_forensics_result(result: TamperForensicsResult) -> None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO tamper_forensics_runs (case_id, created_at, payload)
            VALUES (?, ?, ?)
            """,
            (
                result.case_id,
                datetime.now(UTC).isoformat(),
                result.model_dump_json(),
            ),
        )
        connection.commit()


def load_tamper_forensics_result(case_id: str) -> TamperForensicsResult | None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute(
            """
            SELECT payload
            FROM tamper_forensics_runs
            WHERE case_id = ?
            """,
            (case_id,),
        ).fetchone()
    if row is None:
        return None
    return TamperForensicsResult.model_validate(json.loads(str(row[0])))


def delete_tamper_forensics_result(case_id: str) -> None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("DELETE FROM tamper_forensics_runs WHERE case_id = ?", (case_id,))
        connection.commit()


def save_real_analysis_result(result: RealCaseAnalysisResult) -> None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO real_analysis_runs (case_id, created_at, payload)
            VALUES (?, ?, ?)
            """,
            (
                result.case.id,
                datetime.now(UTC).isoformat(),
                result.model_dump_json(),
            ),
        )
        connection.commit()


def load_real_analysis_result(case_id: str) -> RealCaseAnalysisResult | None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        row = connection.execute(
            """
            SELECT payload
            FROM real_analysis_runs
            WHERE case_id = ?
            """,
            (case_id,),
        ).fetchone()
    if row is None:
        return None
    return RealCaseAnalysisResult.model_validate(json.loads(str(row[0])))


def delete_real_analysis_result(case_id: str) -> None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("DELETE FROM real_analysis_runs WHERE case_id = ?", (case_id,))
        connection.commit()


def save_web_snapshot(snapshot: WebEvidenceSnapshot) -> None:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO web_snapshots (
                id,
                case_id,
                requested_url,
                final_url,
                title,
                text,
                sha256,
                status,
                error,
                html_path,
                text_path,
                screenshot_path,
                screenshot_url,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.id,
                snapshot.case_id,
                snapshot.requested_url,
                snapshot.final_url,
                snapshot.title,
                snapshot.text,
                snapshot.sha256,
                snapshot.status,
                snapshot.error,
                snapshot.html_path,
                snapshot.text_path,
                snapshot.screenshot_path,
                snapshot.screenshot_url,
                snapshot.created_at,
            ),
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO knowledge_documents (id, title, source, category, content)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                f"snapshot-{snapshot.id}",
                snapshot.title or snapshot.final_url,
                snapshot.final_url,
                "URL取证快照",
                snapshot.text,
            ),
        )
        _rebuild_knowledge_fts(connection)
        connection.commit()


def list_web_snapshots(case_id: str) -> list[WebEvidenceSnapshot]:
    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                case_id,
                requested_url,
                final_url,
                title,
                text,
                sha256,
                status,
                error,
                html_path,
                text_path,
                screenshot_path,
                screenshot_url,
                created_at
            FROM web_snapshots
            WHERE case_id = ?
            ORDER BY created_at DESC
            """,
            (case_id,),
        ).fetchall()
    return [_web_snapshot_from_row(row) for row in rows]


def save_evidence_items(case_id: str, items: list["EvidenceItem"]) -> None:
    from app.models import EvidenceItem

    initialize_database()
    created_at = datetime.now(UTC).isoformat()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("DELETE FROM evidence_items WHERE case_id = ?", (case_id,))
        connection.executemany(
            """
            INSERT INTO evidence_items (
                id,
                case_id,
                type,
                title,
                content,
                confidence,
                source,
                supports,
                artifact_id,
                source_url,
                sha256,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item.id,
                    case_id,
                    item.type.value,
                    item.title,
                    item.content,
                    item.confidence,
                    item.source,
                    item.supports,
                    item.artifact_id,
                    item.source_url,
                    item.sha256,
                    item.created_at or created_at,
                )
                for item in items
            ],
        )
        for item in items:
            connection.execute(
                """
                INSERT OR REPLACE INTO knowledge_documents (id, title, source, category, content)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    f"evidence-{item.id}",
                    item.title,
                    item.source_url or item.source,
                    item.type.value,
                    item.content,
                ),
            )
        _rebuild_knowledge_fts(connection)
        connection.commit()


def list_evidence_items(case_id: str) -> list["EvidenceItem"]:
    from app.models import EvidenceItem, EvidenceType

    initialize_database()
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                type,
                title,
                content,
                confidence,
                source,
                supports,
                artifact_id,
                source_url,
                sha256,
                created_at
            FROM evidence_items
            WHERE case_id = ?
            ORDER BY created_at DESC, id
            """,
            (case_id,),
        ).fetchall()
    return [
        EvidenceItem(
            id=str(row[0]),
            type=EvidenceType(str(row[1])),
            title=str(row[2]),
            content=str(row[3]),
            confidence=float(row[4]),
            source=str(row[5]),
            supports=str(row[6]),
            artifact_id=str(row[7]) if row[7] is not None else None,
            source_url=str(row[8]) if row[8] is not None else None,
            sha256=str(row[9]) if row[9] is not None else None,
            created_at=str(row[10]) if row[10] is not None else None,
        )
        for row in rows
    ]


def get_case_evidence_bundle(case_id: str) -> CaseEvidenceBundle:
    return CaseEvidenceBundle(
        case_id=case_id,
        assets=list_case_assets(case_id),
        snapshots=list_web_snapshots(case_id),
        evidence_items=list_evidence_items(case_id),
    )


def search_knowledge(query: str, limit: int = 5) -> list[KnowledgeSearchResult]:
    initialize_database()
    safe_limit = max(1, min(limit, 20))
    with sqlite3.connect(DB_PATH) as connection:
        fts_query = _fts_query(query)
        rows: list[tuple[object, ...]] = []
        if fts_query:
            try:
                rows = connection.execute(
                    """
                    SELECT
                        id,
                        title,
                        source,
                        category,
                        content,
                        bm25(knowledge_fts) AS rank
                    FROM knowledge_fts
                    WHERE knowledge_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, safe_limit),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
        if rows:
            return [
                KnowledgeSearchResult(
                    id=str(row[0]),
                    title=str(row[1]),
                    source=str(row[2]),
                    category=str(row[3]),
                    content=_snippet(str(row[4])),
                    score=round(1.0 / (1.0 + abs(float(row[5]))), 4),
                    evidence_id=_strip_prefix(str(row[0]), "evidence-"),
                    source_url=str(row[2]) if str(row[2]).startswith(("http://", "https://")) else None,
                )
                for row in rows
            ]
        fallback_rows = connection.execute(
            """
            SELECT id, title, source, category, content
            FROM knowledge_documents
            """
        ).fetchall()
    terms = _search_terms(query)
    scored: list[KnowledgeSearchResult] = []
    for row in fallback_rows:
        text = f"{row[1]} {row[2]} {row[3]} {row[4]}".lower()
        score = _knowledge_score(text, terms)
        if score > 0:
            scored.append(
                KnowledgeSearchResult(
                    id=str(row[0]),
                    title=str(row[1]),
                    source=str(row[2]),
                    category=str(row[3]),
                    content=_snippet(str(row[4])),
                    score=round(score, 4),
                    evidence_id=_strip_prefix(str(row[0]), "evidence-"),
                    source_url=str(row[2]) if str(row[2]).startswith(("http://", "https://")) else None,
                )
            )
    if not scored and fallback_rows:
        scored = [
            KnowledgeSearchResult(
                id=str(row[0]),
                title=str(row[1]),
                source=str(row[2]),
                category=str(row[3]),
                content=_snippet(str(row[4])),
                score=0.01,
                evidence_id=_strip_prefix(str(row[0]), "evidence-"),
                source_url=str(row[2]) if str(row[2]).startswith(("http://", "https://")) else None,
            )
            for row in fallback_rows
        ]
    return sorted(scored, key=lambda item: item.score, reverse=True)[:safe_limit]


def _seed_tamper_demo_assets(connection: sqlite3.Connection) -> None:
    try:
        from PIL import Image
    except ImportError:
        return
    data_root = Path(
        os.getenv(
            "SMARTPOLICE_DATA_ROOT",
            str(Path(__file__).resolve().parents[1] / "data"),
        )
    )
    bundled_source_root = Path(__file__).resolve().parent / "demo_assets" / "tamper"
    source_root = data_root / "tamper_demo_sources"
    specs = {
        "tamper-demo-order-after-sale-001": {
            "asset_id": "asset-tamper-demo-order",
            "filename": "tamper-demo-order-after-sale.png",
            "source_path": bundled_source_root / "tamper-demo-order-after-sale.png",
            "legacy_filenames": ["tamper-demo-order-after-sale.jpg", "tamper-demo-disaster-material.jpg"],
        },
        "tamper-demo-bank-transfer-001": {
            "asset_id": "asset-tamper-demo-bank",
            "filename": "tamper-demo-bank-transfer.jpg",
            "source_stem": "tamper-demo-bank-transfer",
            "legacy_filenames": ["tamper-demo-bank-transfer.png", "tamper-demo-old-image-screenshot.jpg"],
        },
        "tamper-demo-medical-complaint-001": {
            "asset_id": "asset-tamper-demo-medical",
            "filename": "tamper-demo-medical-complaint.jpg",
            "source_path": bundled_source_root / "tamper-demo-medical-complaint.jpg",
            "legacy_filenames": ["tamper-demo-medical-complaint.png", "tamper-demo-public-order-screenshot.png"],
        },
    }
    for case in TAMPER_DEMO_CASES:
        spec = specs[case.id]
        deleted = connection.execute(
            "SELECT 1 FROM case_deletions WHERE id = ?",
            (case.id,),
        ).fetchone()
        if deleted is not None:
            continue
        seeded_case = connection.execute(
            "SELECT 1 FROM case_samples WHERE id = ?",
            (case.id,),
        ).fetchone()
        if seeded_case is None:
            continue
        case_dir = data_root / "uploads" / case.id
        case_dir.mkdir(parents=True, exist_ok=True)
        for legacy_name in spec.get("legacy_filenames", []):
            legacy_path = case_dir / str(legacy_name)
            if legacy_path.exists() and legacy_path.is_file():
                legacy_path.unlink()
        source_path = Path(str(spec["source_path"])) if spec.get("source_path") else _tamper_demo_source_path(source_root, str(spec["source_stem"]))
        if source_path is None:
            connection.execute(
                "DELETE FROM case_assets WHERE id = ? OR case_id = ?",
                (spec["asset_id"], case.id),
            )
            connection.execute(
                "DELETE FROM tamper_forensics_runs WHERE case_id = ?",
                (case.id,),
            )
            continue
        image_path = case_dir / str(spec["filename"])
        try:
            source_raw = source_path.read_bytes()
            source_digest = sha256(source_raw).hexdigest()
            existing_asset = connection.execute(
                """
                SELECT sha256, storage_path
                FROM case_assets
                WHERE id = ? AND case_id = ?
                """,
                (spec["asset_id"], case.id),
            ).fetchone()
            if (
                existing_asset is not None
                and str(existing_asset[0]) == source_digest
                and Path(str(existing_asset[1])).is_file()
                and Path(str(existing_asset[1])).resolve() == image_path.resolve()
            ):
                continue
            image_path.write_bytes(source_raw)
            raw = source_raw
            with Image.open(image_path) as image:
                width, height = image.size
                content_type = _content_type_for_path(image_path)
        except OSError:
            connection.execute(
                "DELETE FROM case_assets WHERE id = ? OR case_id = ?",
                (spec["asset_id"], case.id),
            )
            connection.execute(
                "DELETE FROM tamper_forensics_runs WHERE case_id = ?",
                (case.id,),
            )
            continue
        digest = sha256(raw).hexdigest()
        created_at = datetime.now(UTC).isoformat()
        relative_path = image_path.resolve().relative_to(data_root.resolve())
        connection.execute(
            """
            INSERT OR REPLACE INTO case_assets (
                id,
                case_id,
                filename,
                content_type,
                size_bytes,
                width,
                height,
                sha256,
                storage_path,
                preview_url,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                spec["asset_id"],
                case.id,
                spec["filename"],
                content_type,
                len(raw),
                width,
                height,
                digest,
                str(image_path),
                f"/evidence/files/{str(relative_path).replace(chr(92), '/')}",
                created_at,
            ),
        )
        connection.execute(
            "DELETE FROM tamper_forensics_runs WHERE case_id = ?",
            (case.id,),
        )


def _seed_generation_demo_assets(connection: sqlite3.Connection) -> None:
    try:
        from PIL import Image
    except ImportError:
        return
    data_root = Path(
        os.getenv(
            "SMARTPOLICE_DATA_ROOT",
            str(Path(__file__).resolve().parents[1] / "data"),
        )
    )
    source_root = Path(__file__).resolve().parents[1] / "demo_assets"
    specs = {
        "demo-doubao-collapse-disaster-001": {
            "asset_id": "asset-demo-nano-banana-collapse",
            "filename": "nano-banana-tunnel-collapse-social.png",
        },
        "demo-gptimage-station-police-conflict-001": {
            "asset_id": "asset-demo-gptimage-station-conflict",
            "filename": "gptimage-station-police-conflict-original.jpg",
        },
    }
    _remove_generation_demo_seeded_real_asset(connection, data_root)
    cases_by_id = {case.id: case for case in DEMO_CASES}
    for case_id, spec in specs.items():
        deleted = connection.execute(
            "SELECT 1 FROM case_deletions WHERE id = ?",
            (case_id,),
        ).fetchone()
        if deleted is not None:
            continue
        case = cases_by_id.get(case_id)
        if case is not None:
            connection.execute(
                """
                UPDATE case_samples
                SET payload = ?
                WHERE id = ?
                """,
                (case.model_dump_json(), case_id),
            )
        seeded_case = connection.execute(
            "SELECT 1 FROM case_samples WHERE id = ?",
            (case_id,),
        ).fetchone()
        if seeded_case is None:
            continue
        source_path = source_root / str(spec["filename"])
        if not source_path.is_file():
            continue
        case_dir = data_root / "uploads" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        _remove_extra_generation_demo_assets(
            connection,
            case_id=case_id,
            keep_asset_id=str(spec["asset_id"]),
            keep_filename=str(spec["filename"]),
            case_dir=case_dir,
        )
        image_path = case_dir / str(spec["filename"])
        try:
            source_raw = source_path.read_bytes()
            source_digest = sha256(source_raw).hexdigest()
            existing_asset = connection.execute(
                """
                SELECT sha256, storage_path
                FROM case_assets
                WHERE id = ? AND case_id = ?
                """,
                (spec["asset_id"], case_id),
            ).fetchone()
            if (
                existing_asset is not None
                and str(existing_asset[0]) == source_digest
                and Path(str(existing_asset[1])).is_file()
                and Path(str(existing_asset[1])).resolve() == image_path.resolve()
            ):
                continue
            image_path.write_bytes(source_raw)
            with Image.open(image_path) as image:
                width, height = image.size
                content_type = _content_type_for_path(image_path)
        except OSError:
            connection.execute(
                "DELETE FROM case_assets WHERE id = ? OR case_id = ?",
                (spec["asset_id"], case_id),
            )
            connection.execute(
                "DELETE FROM image_forensics_runs WHERE case_id = ?",
                (case_id,),
            )
            continue
        created_at = datetime.now(UTC).isoformat()
        relative_path = image_path.resolve().relative_to(data_root.resolve())
        connection.execute(
            """
            INSERT OR REPLACE INTO case_assets (
                id,
                case_id,
                filename,
                content_type,
                size_bytes,
                width,
                height,
                sha256,
                storage_path,
                preview_url,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                spec["asset_id"],
                case_id,
                spec["filename"],
                content_type,
                len(source_raw),
                width,
                height,
                source_digest,
                str(image_path),
                f"/evidence/files/{str(relative_path).replace(chr(92), '/')}",
                created_at,
            ),
        )
        connection.execute(
            "DELETE FROM image_forensics_runs WHERE case_id = ?",
            (case_id,),
        )


def _remove_generation_demo_seeded_real_asset(connection: sqlite3.Connection, data_root: Path) -> None:
    case_id = "demo-real-beijing-road-street-001"
    filename = "real-sichuan-earthquake-rescue.jpg"
    connection.execute(
        """
        DELETE FROM case_assets
        WHERE case_id = ?
          AND (id = ? OR filename = ?)
        """,
        (case_id, "asset-demo-real-sichuan-earthquake-rescue", filename),
    )
    image_path = data_root / "uploads" / case_id / filename
    if image_path.is_file():
        try:
            image_path.unlink()
        except OSError:
            pass
    connection.execute(
        "DELETE FROM image_forensics_runs WHERE case_id = ?",
        (case_id,),
    )


def _remove_extra_generation_demo_assets(
    connection: sqlite3.Connection,
    *,
    case_id: str,
    keep_asset_id: str,
    keep_filename: str,
    case_dir: Path,
) -> None:
    rows = connection.execute(
        """
        SELECT id, filename, storage_path
        FROM case_assets
        WHERE case_id = ?
          AND (id != ? OR filename != ?)
        """,
        (case_id, keep_asset_id, keep_filename),
    ).fetchall()
    for _, filename, storage_path in rows:
        for candidate in {case_dir / str(filename), Path(str(storage_path))}:
            if candidate.is_file():
                try:
                    candidate.unlink()
                except OSError:
                    pass
    connection.execute(
        """
        DELETE FROM case_assets
        WHERE case_id = ?
          AND (id != ? OR filename != ?)
        """,
        (case_id, keep_asset_id, keep_filename),
    )


def _tamper_demo_source_path(source_root: Path, stem: str) -> Path | None:
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = source_root / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _content_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _case_id(payload: CaseCreateRequest) -> str:
    if payload.id and payload.id.strip():
        return _slug(payload.id)
    base = _slug(f"{payload.scenario}-{payload.title}")[:42]
    suffix = uuid4().hex[:8]
    return f"{base}-{suffix}"


def _slug(value: str) -> str:
    lowered = value.strip().lower()
    slug = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "-", lowered).strip("-")
    return slug or f"case-{uuid4().hex[:8]}"


def _remove_case_directory(root: Path, case_id: str) -> None:
    target = (root / case_id).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        return
    if target.exists() and target.is_dir():
        shutil.rmtree(target)


def _agent_run_from_row(row: tuple[object, ...]) -> AgentRunRecord:
    return AgentRunRecord(
        id=str(row[0]),
        case_id=str(row[1]),
        created_at=str(row[2]),
        risk_level=str(row[3]),
        risk_score=int(row[4]),
        model_routes=[
            AgentModelRoute.model_validate(item)
            for item in json.loads(str(row[5]))
        ],
        skill_names=list(json.loads(str(row[6]))),
        estimated_cost_units=int(row[7]),
        primary_strategy=str(row[8]),
    )


def _route_cost_units(route: AgentModelRoute) -> int:
    if "低" in route.cost_tier:
        return 1
    if "中" in route.cost_tier:
        return 3
    if "按需" in route.cost_tier:
        return 5
    return 8


def _supported_provider_run(run: AgentRunRecord) -> AgentRunRecord:
    return AgentRunRecord(
        id=run.id,
        case_id=run.case_id,
        created_at=run.created_at,
        risk_level=run.risk_level,
        risk_score=run.risk_score,
        model_routes=[
            route
            for route in run.model_routes
            if route.provider in SUPPORTED_PROVIDERS
        ],
        skill_names=run.skill_names,
        estimated_cost_units=run.estimated_cost_units,
        primary_strategy=run.primary_strategy,
    )


def _llm_invocation_from_row(row: tuple[object, ...]) -> ModelInvocationAudit:
    return ModelInvocationAudit(
        id=str(row[0]),
        case_id=str(row[1]) if row[1] is not None else None,
        provider=str(row[2]),
        role=str(row[3]),
        model=str(row[4]),
        status=str(row[5]),
        request_payload=dict(json.loads(str(row[6]))),
        response_text=str(row[7]) if row[7] is not None else None,
        error=str(row[8]) if row[8] is not None else None,
        latency_ms=int(row[9]),
        token_usage=dict(json.loads(str(row[10]))),
        created_at=str(row[11]),
    )


def _search_terms(query: str) -> list[str]:
    normalized = query.strip().lower()
    terms = re.findall(r"[0-9a-zA-Z\u4e00-\u9fff]+", normalized)
    compact_terms = [
        item
        for item in terms
        if len(item) > 1 or re.search(r"[\u4e00-\u9fff]", item)
    ]
    return compact_terms or [normalized] if normalized else []


def _knowledge_score(text: str, terms: list[str]) -> float:
    if not terms:
        return 0.0
    score = 0.0
    for term in terms:
        if not term:
            continue
        count = text.count(term)
        if count:
            score += 1.0 + min(count, 6) * 0.4
    coverage = sum(1 for term in terms if term and term in text) / max(len(terms), 1)
    return score + coverage * 2.0


def _case_asset_from_row(row: tuple[object, ...]) -> CaseAsset:
    return CaseAsset(
        id=str(row[0]),
        case_id=str(row[1]),
        filename=str(row[2]),
        content_type=str(row[3]),
        size_bytes=int(row[4]),
        width=int(row[5]) if row[5] is not None else None,
        height=int(row[6]) if row[6] is not None else None,
        sha256=str(row[7]),
        storage_path=str(row[8]),
        preview_url=str(row[9]),
        created_at=str(row[10]),
    )


def _web_snapshot_from_row(row: tuple[object, ...]) -> WebEvidenceSnapshot:
    return WebEvidenceSnapshot(
        id=str(row[0]),
        case_id=str(row[1]),
        requested_url=str(row[2]),
        final_url=str(row[3]),
        title=str(row[4]),
        text=str(row[5]),
        sha256=str(row[6]),
        status=str(row[7]),
        error=str(row[8]) if row[8] is not None else None,
        html_path=str(row[9]),
        text_path=str(row[10]),
        screenshot_path=str(row[11]) if row[11] is not None else None,
        screenshot_url=str(row[12]) if row[12] is not None else None,
        created_at=str(row[13]),
    )


def _external_training_sample_from_row(row: tuple[object, ...]) -> ExternalTrainingSample:
    if len(row) >= 17:
        raw = json.loads(str(row[15]))
        return ExternalTrainingSample(
            id=str(row[0]),
            dataset_name=str(row[1]),
            source=str(row[2]),
            source_url=str(row[3]) if row[3] is not None else None,
            task_type=str(row[4] or "text_risk"),
            split=str(row[5]),
            title=str(row[6]),
            content=str(row[7]),
            image_path=str(row[8]) if row[8] is not None else None,
            image_url=str(row[9]) if row[9] is not None else None,
            image_sha256=str(row[10]) if row[10] is not None else None,
            image_available=bool(row[11]),
            label=str(row[12]),
            risk_score=int(row[13]),
            scenario=str(row[14]),
            raw_payload=dict(raw) if isinstance(raw, dict) else {},
            created_at=str(row[16]),
        )
    raw = json.loads(str(row[10]))
    return ExternalTrainingSample(
        id=str(row[0]),
        dataset_name=str(row[1]),
        source=str(row[2]),
        source_url=str(row[3]) if row[3] is not None else None,
        task_type="text_risk",
        split=str(row[4]),
        title=str(row[5]),
        content=str(row[6]),
        label=str(row[7]),
        risk_score=int(row[8]),
        scenario=str(row[9]),
        raw_payload=dict(raw) if isinstance(raw, dict) else {},
        created_at=str(row[11]),
    )


def _ensure_columns(
    connection: sqlite3.Connection,
    table_name: str,
    columns: dict[str, str],
) -> None:
    existing = {
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    for column, definition in columns.items():
        if column not in existing:
            connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column} {definition}")


def _known_task_types() -> list[str]:
    return [
        "text_risk",
        "vision_aigc",
        "vision_tamper",
        "vision_context_mismatch",
        "vision_generator_attribution",
        "multimodal_fusion",
    ]


def _training_task_status(
    *,
    task_type: str,
    sample_count: int,
    image_available_count: int,
    label_distribution: dict[str, int],
    sources: list[ExternalDatasetSourceSummary],
) -> TrainingTaskStatus:
    requires_images = task_type in {
        "vision_aigc",
        "vision_tamper",
        "vision_context_mismatch",
        "vision_generator_attribution",
        "multimodal_fusion",
    }
    ready_count = image_available_count if requires_images else sample_count
    training_ready = ready_count >= 20
    if sample_count == 0:
        note = "尚未导入该任务的外部数据；内置展示样例不会进入训练集。"
    elif requires_images and image_available_count == 0:
        note = "该任务需要本地图片文件，请导入 image_root 与 image_path_column 可解析的样本。"
    elif not training_ready:
        note = f"已有 {ready_count} 条可训练样本；正式 UI 建议至少 20 条，测试可用 min_samples=4。"
    else:
        note = "该任务已具备正式训练入口；训练仍只使用外部/人工标注样本。"
    return TrainingTaskStatus(
        task_type=task_type,
        sample_count=sample_count,
        image_available_count=image_available_count,
        label_distribution=label_distribution,
        training_ready=training_ready,
        sources=sources,
        note=note,
    )


def _valid_training_artifact(artifact: dict[str, object]) -> bool:
    summary = artifact.get("training_source_summary")
    if not isinstance(summary, dict):
        return False
    base_count = summary.get("base_sample_count")
    excluded_count = summary.get("excluded_demo_cases")
    try:
        return int(base_count) >= 4 and int(excluded_count) >= len(DEMO_CASES)
    except (TypeError, ValueError):
        return False


def _rebuild_knowledge_fts(connection: sqlite3.Connection) -> None:
    connection.execute("DELETE FROM knowledge_fts")
    connection.execute(
        """
        INSERT INTO knowledge_fts (id, title, source, category, content)
        SELECT id, title, source, category, content
        FROM knowledge_documents
        """
    )


def _fts_query(query: str) -> str:
    terms = _search_terms(query)
    if not terms:
        return ""
    return " OR ".join(f'"{term}"' for term in terms[:8])


def _snippet(content: str, limit: int = 360) -> str:
    compact = re.sub(r"\s+", " ", content).strip()
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def _strip_prefix(value: str, prefix: str) -> str | None:
    return value[len(prefix):] if value.startswith(prefix) else None


def recommended_huggingface_datasets_for_task(task_type: str | None = None) -> list[dict[str, str]]:
    return _recommended_huggingface_datasets(task_type)


def _recommended_huggingface_datasets(task_type: str | None = None) -> list[dict[str, str]]:
    if task_type == "vision_tamper":
        return [
            {
                "name": "lorenzo-morelli/image-splicing-deepfake-mix",
                "url": "https://huggingface.co/datasets/lorenzo-morelli/image-splicing-deepfake-mix",
                "reason": "当前篡改线已使用的 HF 拼接/插入篡改图像池，包含 authentic_unmodified 与 splicing_tampered 标签；用于通用篡改证据头训练，不宣称单据类定位 benchmark。",
            },
            {
                "name": "AdoCleanCode/Fakeddit manipulated content subset",
                "url": "https://huggingface.co/datasets/AdoCleanCode/Fakeddit",
                "reason": "当前篡改线已使用其中 manipulated content / negative true 图像样本，作为通用篡改/非篡改监督补充；不能作为单据/凭证专属数据集宣称。",
            },
            {
                "name": "CASIA / IMD2020 / AutoSplice document-tamper follow-up",
                "url": "https://huggingface.co/datasets?search=image%20tampering",
                "reason": "下一步检索并导入篡改任务自己的 mask/bbox 或 document-tamper 数据；只允许 tamper/splice/inpaint/copy-move/document-tamper，不混用生成检测数据。",
            },
        ]
    return [
        {
            "name": "FinanceMTEB/MDFEND-Weibo21",
            "url": "https://huggingface.co/datasets/FinanceMTEB/MDFEND-Weibo21",
            "reason": "中文微博谣言/假新闻检测方向，可用于训练文本风险基线。",
        },
        {
            "name": "MCFEND",
            "url": "https://trustworthycomp.github.io/mcfend/",
            "reason": "中文多源假新闻检测基准，公开资料说明包含约 2.4 万条中文新闻。",
        },
        {
            "name": "CHECKED / Chinese COVID-19 fake news",
            "url": "https://github.com/cyang03/CHECKED",
            "reason": "中文公共事件事实核查数据，可补充公共安全类谣言文本训练。",
        },
    ]
