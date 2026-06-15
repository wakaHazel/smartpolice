import os
from collections import Counter
from pathlib import Path
import sqlite3
import tempfile
from typing import Any
import socket

TEST_DB_PATH = Path(tempfile.gettempdir()) / "smartpolice-pytest.db"
if TEST_DB_PATH.exists():
    TEST_DB_PATH.unlink()
os.environ["SMARTPOLICE_DB_PATH"] = str(TEST_DB_PATH)
os.environ.setdefault("SMARTPOLICE_ENABLE_LOCAL_VLM", "0")
os.environ.setdefault("SMARTPOLICE_REQUIRE_LOCAL_VISION", "0")
os.environ.setdefault("SMARTPOLICE_ENABLE_CLOUD_REVIEW", "0")
os.environ.setdefault("SMARTPOLICE_ENABLE_CLOUD_REPORT", "0")

from fastapi.testclient import TestClient
import httpx
from PIL import Image, ImageDraw

from app import evidence_service
from app import real_analysis
from app.main import app
from app.models import CaseAsset, ExternalTrainingSample
from app.multimodal_training import (
    GENERATOR_BINARY_GATE_THRESHOLD,
    GENERATOR_REAL_PROTECTION_MARGIN,
    _apply_generator_binary_gate,
    _balanced_generator_samples_for_request,
    _balanced_generator_samples,
    _balanced_gpt_image2_ovr_samples,
    _generator_binary_gate_policy,
    _generator_binary_gate_threshold,
    _generator_experiment_view,
    _generator_profile_feature_policy,
    _generator_profile_policy,
    _is_real_hard_negative_source,
    _is_generated_hard_positive_source,
    _open_set_unknown_threshold,
    _predict_generator_with_classifier,
    _source_balanced_sample_weights,
    _source_holdout_group_name,
)
from app.models import VisionTrainingRunRequest
from app.storage import initialize_database


client = TestClient(app)


def _reset_training_pool() -> None:
    if TEST_DB_PATH.exists():
        with sqlite3.connect(TEST_DB_PATH) as connection:
            connection.execute("DROP TABLE IF EXISTS external_training_samples")
            connection.execute("DROP TABLE IF EXISTS training_runs")
            connection.execute("DROP TABLE IF EXISTS vision_training_runs")
            connection.execute("DROP TABLE IF EXISTS fusion_training_runs")
            connection.execute("DROP TABLE IF EXISTS feature_cache")
            connection.execute("DROP TABLE IF EXISTS case_samples")
            connection.commit()
    initialize_database()


class MockResponse:
    def __init__(
        self,
        payload: dict[str, Any],
        status_code: int = 200,
    ) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "mock upstream error",
                request=httpx.Request("POST", "https://example.test"),
                response=httpx.Response(self.status_code),
            )

    def json(self) -> dict[str, Any]:
        return self.payload


class MockHttpxClient:
    last_request: dict[str, Any] | None = None
    response_payload: dict[str, Any] = {
        "choices": [{"message": {"content": '{"ok": true}'}}],
        "usage": {"total_tokens": 42},
    }
    status_code = 200
    exception: Exception | None = None

    def __init__(self, timeout: int) -> None:
        self.timeout = timeout

    def __enter__(self) -> "MockHttpxClient":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def post(
        self,
        url: str,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> MockResponse:
        MockHttpxClient.last_request = {
            "url": url,
            "headers": headers,
            "json": json,
            "timeout": self.timeout,
        }
        if MockHttpxClient.exception is not None:
            raise MockHttpxClient.exception
        return MockResponse(MockHttpxClient.response_payload, MockHttpxClient.status_code)


def _mock_httpx_success(
    monkeypatch: Any,
    content: str = '{"ok": true}',
) -> None:
    MockHttpxClient.last_request = None
    MockHttpxClient.exception = None
    MockHttpxClient.status_code = 200
    MockHttpxClient.response_payload = {
        "choices": [{"message": {"content": content}}],
        "usage": {"total_tokens": 42, "prompt_tokens": 20, "completion_tokens": 22},
    }
    monkeypatch.setattr(httpx, "Client", MockHttpxClient)


def _configure_minimax(monkeypatch: Any) -> None:
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://api.minimax.test/v1")
    monkeypatch.setenv("MINIMAX_MODEL_ROUTER", "MiniMax-M3-test")
    monkeypatch.setenv("MINIMAX_MODEL_REPORT", "MiniMax-M3-test")
    monkeypatch.setenv("MINIMAX_MODEL_LONG_CONTEXT", "MiniMax-M3-test")


def _configure_deepseek(monkeypatch: Any) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.test")
    monkeypatch.setenv("DEEPSEEK_MODEL_REASONING", "deepseek-test")
    monkeypatch.setenv("DEEPSEEK_MODEL_VERIFIER", "deepseek-test")


def _configure_dashscope(monkeypatch: Any) -> None:
    monkeypatch.setenv("ENABLE_DASHSCOPE", "1")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-dashscope-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://dashscope.test/compatible-mode/v1")
    monkeypatch.setenv("DASHSCOPE_VISION_MODEL", "qwen-vl-test")


def _configure_local_vision(monkeypatch: Any) -> None:
    monkeypatch.setenv("VISION_PROVIDER", "LocalVision")
    monkeypatch.setenv("SMARTPOLICE_ENABLE_LOCAL_VLM", "1")
    monkeypatch.setenv("LOCAL_VISION_BASE_URL", "http://local-vision.test/v1")
    monkeypatch.setenv("LOCAL_VISION_MODEL", "qwen2.5vl-test")


def _png_1x1() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
        b"\x90wS\xde\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def _create_localvision_training_sample(
    monkeypatch: Any,
    *,
    case_id: str,
    manual_score: int,
    visual_content: str,
) -> dict[str, Any]:
    create_response = client.post(
        "/cases",
        json={
            "id": case_id,
            "title": f"本地视觉训练样本 {case_id}",
            "scenario": "涉警公信力谣言",
            "platform": "短视频平台",
            "publish_time": "2026-06-07 10:00",
            "source_url": f"https://example.com/{case_id}",
            "content": "网传警情截图引发讨论，需固定证据并核验来源。",
            "image_description": "用于测试 LocalVision 真实审计样本沉淀。",
            "spread": {
                "views": 80000 + manual_score * 100,
                "reposts": 1200 + manual_score,
                "comments": 900 + manual_score,
                "likes": 1800,
                "velocity": "1小时内快速扩散",
            },
            "manual_label": "人工复核标注",
            "manual_risk_score": manual_score,
            "tags": ["本地视觉", "训练样本"],
            "sensitivity_notes": "pytest local vision calibration sample",
            "review_note": "pytest",
        },
    )
    assert create_response.status_code == 200
    upload = client.post(
        f"/cases/{case_id}/assets",
        files={"file": ("screen.png", _png_1x1(), "image/png")},
    )
    assert upload.status_code == 200

    class MockCaptureResponse:
        content = (
            f"<html><head><title>{case_id}</title></head>"
            "<body>official public safety notice for local vision calibration</body></html>"
        ).encode()
        encoding = "utf-8"
        url = f"https://example.com/{case_id}"

        def raise_for_status(self) -> None:
            return None

    class MockCaptureClient:
        def __init__(self, **_: Any) -> None:
            return None

        def __enter__(self) -> "MockCaptureClient":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def get(self, url: str) -> MockCaptureResponse:
            assert url == f"https://example.com/{case_id}"
            return MockCaptureResponse()

    monkeypatch.setattr(socket, "getaddrinfo", lambda *_: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))])
    monkeypatch.setattr(httpx, "Client", MockCaptureClient)
    monkeypatch.setattr(evidence_service, "_capture_screenshot", lambda *_: False)
    snapshot = client.post(
        f"/cases/{case_id}/sources/capture",
        json={"url": f"https://example.com/{case_id}"},
    )
    assert snapshot.status_code == 200

    _configure_local_vision(monkeypatch)
    _configure_deepseek(monkeypatch)
    _configure_minimax(monkeypatch)

    class CalibrationSampleMockClient(MockHttpxClient):
        def post(
            self,
            url: str,
            headers: dict[str, str],
            json: dict[str, Any],
        ) -> MockResponse:
            if "local-vision" in url:
                content = visual_content
            elif "deepseek" in url:
                content = (
                    '{"conclusion": "纳入人工复核", "risk_level": "较高", '
                    f'"risk_score": {manual_score}, "key_evidence_ids": [], '
                    '"evidence_conflicts": [], "disposal_suggestions": ["核验原图"], '
                    '"missing_checks": ["原始出处"], "human_review_required": true}'
                )
            else:
                content = '{"markdown": "# 研判报告\\n\\n本地视觉训练样本。"}'
            return MockResponse(
                {
                    "choices": [{"message": {"content": content}}],
                    "usage": {"total_tokens": 64},
                }
            )

    monkeypatch.setattr(httpx, "Client", CalibrationSampleMockClient)
    response = client.post(f"/cases/{case_id}/real-analysis")
    assert response.status_code == 200
    return {"case_id": case_id, "asset_id": upload.json()["id"], "analysis": response.json()}


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_full_analysis_for_demo_case() -> None:
    response = client.post("/analysis/full", json={"case_id": "group-polarization-003"})
    assert response.status_code == 200
    body = response.json()
    assert body["case"]["scenario"] == "群体对立煽动型谣言"
    assert len(body["evidence_chain"]) == 5
    assert body["analysis"]["generator_attribution"]
    assert body["analysis"]["generator_attribution"][0]["candidate_model"]
    assert body["risk"]["level"] in {"关注", "较高", "紧急"}
    assert "生成模型来源归因" in body["report"]["markdown"]
    assert "研判报告" in body["report"]["markdown"]
    assert body["agent"]["model_routes"]
    assert body["agent"]["recommended_skills"]
    assert any(
        item["algorithm"] == "Embedding + UMAP + HDBSCAN"
        for item in body["agent"]["learning_pipeline"]
    )


def test_low_risk_is_not_over_alerted() -> None:
    response = client.post("/risk/assess", json={"case_id": "low-risk-004"})
    assert response.status_code == 200
    body = response.json()
    assert body["level"] == "低"
    assert body["score"] < 40


def test_agent_orchestration_exposes_model_routing() -> None:
    response = client.post("/agent/orchestrate", json={"case_id": "disaster-risk-002"})
    assert response.status_code == 200
    body = response.json()
    assert body["case_id"] == "disaster-risk-002"
    assert any(route["provider"] == "DeepSeek" for route in body["model_routes"])
    assert any(route["provider"] == "MiniMax" for route in body["model_routes"])
    assert all(route["provider"] in {"DeepSeek", "MiniMax"} for route in body["model_routes"])
    assert any(skill["name"] == "joint_disposal_skill" for skill in body["recommended_skills"])
    assert body["cost_gates"]


def test_agent_metrics_are_recorded_from_full_analysis() -> None:
    response = client.post("/analysis/full", json={"case_id": "police-trust-001"})
    assert response.status_code == 200

    metrics_response = client.get("/agent/metrics")
    assert metrics_response.status_code == 200
    metrics = metrics_response.json()
    assert metrics["total_runs"] >= 1
    assert metrics["average_cost_units"] > 0
    assert any(item["name"] == "DeepSeek" for item in metrics["provider_usage"])
    assert metrics["recent_runs"]

    runs_response = client.get("/agent/runs?limit=3")
    assert runs_response.status_code == 200
    runs = runs_response.json()
    assert len(runs) >= 1
    assert runs[0]["model_routes"]


def test_model_gateway_status_lists_expected_providers() -> None:
    response = client.get("/agent/model-gateway/status")
    assert response.status_code == 200
    body = response.json()
    providers = {item["provider"] for item in body["providers"]}
    assert providers == {"LocalReview", "LocalReport", "LocalVision", "DeepSeek", "MiniMax"}
    assert "DashScope" not in providers
    local_review = next(item for item in body["providers"] if item["provider"] == "LocalReview")
    assert local_review["configured"] is True
    assert local_review["adapter"] == "local-offline"
    local = next(item for item in body["providers"] if item["provider"] == "LocalVision")
    assert local["default_models"]["视觉证据分析"]
    assert body["dry_run_default"] is False
    assert "本地优先" in body["note"]


def test_model_gateway_dashscope_is_optional(monkeypatch: Any) -> None:
    _configure_dashscope(monkeypatch)

    response = client.get("/agent/model-gateway/status")

    assert response.status_code == 200
    providers = {item["provider"] for item in response.json()["providers"]}
    assert "DashScope" in providers


def test_model_gateway_local_review_records_audit() -> None:
    response = client.post(
        "/agent/model-gateway/invoke",
        json={
            "case_id": "group-polarization-003",
            "provider": "LocalReview",
            "role": "本地结构化复核",
            "prompt": "本地复核一下证据链。",
            "dry_run": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "LocalReview"
    assert body["audit_id"]
    assert body["response_text"]
    audits = client.get("/agent/model-gateway/invocations?case_id=group-polarization-003&limit=5").json()
    assert any(item["id"] == body["audit_id"] and item["provider"] == "LocalReview" for item in audits)


def test_model_gateway_localvision_health_uses_lmstudio_model_list(monkeypatch: Any) -> None:
    _configure_local_vision(monkeypatch)

    class MockModelListClient:
        def __init__(self, timeout: int | float) -> None:
            self.timeout = timeout

        def __enter__(self) -> "MockModelListClient":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def get(self, url: str) -> MockResponse:
            assert url == "http://local-vision.test/v1/models"
            return MockResponse(
                {
                    "object": "list",
                    "data": [{"id": "qwen2.5vl-test", "object": "model"}],
                }
            )

    monkeypatch.setattr(httpx, "Client", MockModelListClient)

    response = client.get("/agent/model-gateway/status")
    assert response.status_code == 200
    local = next(item for item in response.json()["providers"] if item["provider"] == "LocalVision")
    assert local["configured"] is True
    assert local["health"] == "ready"


def test_model_gateway_localvision_accepts_external_openai_compatible_endpoint(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("VISION_PROVIDER", "LocalVision")
    monkeypatch.setenv("SMARTPOLICE_ENABLE_LOCAL_VLM", "1")
    monkeypatch.setenv("LOCAL_VISION_BASE_URL", "https://api.v3.cm/v1")
    monkeypatch.setenv("LOCAL_VISION_MODEL", "qwen3-vl-plus")
    monkeypatch.setenv("LOCAL_VISION_API_KEY", "test-key")

    response = client.get("/agent/model-gateway/status")

    assert response.status_code == 200
    local = next(item for item in response.json()["providers"] if item["provider"] == "LocalVision")
    assert local["configured"] is True
    assert local["base_url"] == "https://api.v3.cm/v1"
    assert local["default_models"]["视觉证据分析"] == "qwen3-vl-plus"
    assert local["health"] == "external-openai-compatible-ready"


def test_model_gateway_dry_run_builds_payload() -> None:
    response = client.post(
        "/agent/model-gateway/invoke",
        json={
            "provider": "MiniMax",
            "role": "任务路由器",
            "prompt": "判断当前任务是否需要强模型复核。",
            "dry_run": True,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "MiniMax"
    assert body["selected_model"]
    assert body["dry_run"] is True
    assert body["request_payload"]["model"] == body["selected_model"]
    assert body["request_payload"]["messages"][0]["role"] == "user"


def test_model_gateway_rejects_unknown_provider() -> None:
    response = client.post(
        "/agent/model-gateway/invoke",
        json={
            "provider": "Unknown",
            "role": "任务路由器",
            "prompt": "test",
            "dry_run": True,
        },
    )
    assert response.status_code == 400


def test_model_gateway_rejects_unknown_role() -> None:
    response = client.post(
        "/agent/model-gateway/invoke",
        json={
            "provider": "MiniMax",
            "role": "复核器",
            "prompt": "test",
            "dry_run": True,
        },
    )
    assert response.status_code == 400


def test_model_gateway_real_call_records_audit(monkeypatch: Any) -> None:
    _configure_minimax(monkeypatch)
    _mock_httpx_success(monkeypatch, content='{"conclusion": "需要复核"}')

    response = client.post(
        "/agent/model-gateway/invoke",
        json={
            "case_id": "group-polarization-003",
            "provider": "MiniMax",
            "role": "任务路由器",
            "prompt": "判断当前任务是否需要强模型复核。",
            "dry_run": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["dry_run"] is False
    assert body["audit_id"]
    assert body["response_text"] == '{"conclusion": "需要复核"}'
    assert MockHttpxClient.last_request is not None
    assert MockHttpxClient.last_request["url"] == "https://api.minimax.test/v1/chat/completions"
    assert MockHttpxClient.last_request["json"]["model"] == "MiniMax-M3-test"

    audits_response = client.get("/agent/model-gateway/invocations?case_id=group-polarization-003&limit=5")
    assert audits_response.status_code == 200
    audits = audits_response.json()
    assert any(item["id"] == body["audit_id"] and item["status"] == "success" for item in audits)


def test_model_gateway_requires_configuration(monkeypatch: Any) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "")

    response = client.post(
        "/agent/model-gateway/invoke",
        json={
            "provider": "DeepSeek",
            "role": "复核器",
            "prompt": "test",
            "dry_run": False,
        },
    )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["audit_id"]
    assert "not configured" in detail["message"]


def test_model_gateway_timeout_records_audit(monkeypatch: Any) -> None:
    _configure_deepseek(monkeypatch)
    MockHttpxClient.last_request = None
    MockHttpxClient.exception = httpx.TimeoutException("timeout")
    monkeypatch.setattr(httpx, "Client", MockHttpxClient)

    response = client.post(
        "/agent/model-gateway/invoke",
        json={
            "case_id": "police-trust-001",
            "provider": "DeepSeek",
            "role": "复核器",
            "prompt": "test",
            "dry_run": False,
        },
    )

    assert response.status_code == 504
    audit_id = response.json()["detail"]["audit_id"]
    audits = client.get("/agent/model-gateway/invocations?case_id=police-trust-001").json()
    assert any(item["id"] == audit_id and item["status"] == "timeout" for item in audits)


def test_model_gateway_empty_response_is_bad_gateway(monkeypatch: Any) -> None:
    _configure_deepseek(monkeypatch)
    _mock_httpx_success(monkeypatch, content="")

    response = client.post(
        "/agent/model-gateway/invoke",
        json={
            "provider": "DeepSeek",
            "role": "复核器",
            "prompt": "test",
            "dry_run": False,
        },
    )

    assert response.status_code == 502
    assert response.json()["detail"]["audit_id"]


def test_knowledge_search_returns_seeded_rules() -> None:
    response = client.get("/knowledge/search?query=公共安全 谣言 证据链&limit=3")
    assert response.status_code == 200
    body = response.json()
    assert body
    assert any("证据" in item["title"] or "谣言" in item["content"] for item in body)


def test_upload_case_asset_records_hash_and_dimensions() -> None:
    response = client.post(
        "/cases/group-polarization-003/assets",
        files={"file": ("screen.png", _png_1x1(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["case_id"] == "group-polarization-003"
    assert body["sha256"]
    assert body["width"] == 1
    assert body["height"] == 1
    assert body["preview_url"].startswith("/evidence/files/uploads/")

    bundle = client.get("/cases/group-polarization-003/evidence").json()
    assert any(item["id"] == body["id"] for item in bundle["assets"])


def test_delete_case_removes_case_and_evidence() -> None:
    case_id = "pytest-delete-case"
    create_response = client.post(
        "/cases",
        json={
            "id": case_id,
            "title": "待删除测试案例",
            "scenario": "低风险误传",
            "platform": "本地测试",
            "publish_time": "2026-06-13 15:00",
            "source_url": "本地录入样本",
            "content": "用于验证案例删除接口。",
            "image_description": "测试图片。",
            "spread": {
                "views": 1,
                "reposts": 0,
                "comments": 0,
                "likes": 0,
                "velocity": "测试",
            },
            "manual_label": "待人工复核",
            "manual_risk_score": None,
            "tags": ["测试"],
            "sensitivity_notes": "",
            "review_note": "",
        },
    )
    assert create_response.status_code in {200, 400}

    upload_response = client.post(
        f"/cases/{case_id}/assets",
        files={"file": ("delete.png", _png_1x1(), "image/png")},
    )
    assert upload_response.status_code == 200

    delete_response = client.delete(f"/cases/{case_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["id"] == case_id

    cases = client.get("/cases").json()
    assert all(item["id"] != case_id for item in cases)
    assert client.get(f"/cases/{case_id}/evidence").status_code == 404
    assert client.delete(f"/cases/{case_id}").status_code == 404


def test_capture_url_rejects_localhost() -> None:
    response = client.post(
        "/cases/group-polarization-003/sources/capture",
        json={"url": "http://127.0.0.1:5173"},
    )

    assert response.status_code == 422


def test_capture_url_saves_snapshot_and_fts(monkeypatch: Any) -> None:
    class MockGetResponse:
        content = (
            "<html><head><title>权威辟谣通报</title></head>"
            "<body><h1>公共安全谣言核查</h1><p>警方发布权威通报，说明网传内容不实。</p></body></html>"
        ).encode()
        encoding = "utf-8"
        url = "https://example.com/notice"

        def raise_for_status(self) -> None:
            return None

    class MockGetClient:
        def __init__(self, **_: Any) -> None:
            return None

        def __enter__(self) -> "MockGetClient":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def get(self, url: str) -> MockGetResponse:
            assert url == "https://example.com/notice"
            return MockGetResponse()

    monkeypatch.setattr(socket, "getaddrinfo", lambda *_: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))])
    monkeypatch.setattr(httpx, "Client", MockGetClient)
    monkeypatch.setattr(evidence_service, "_capture_screenshot", lambda *_: False)

    response = client.post(
        "/cases/group-polarization-003/sources/capture",
        json={"url": "https://example.com/notice"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "权威辟谣通报"
    assert body["sha256"]
    assert body["status"] == "captured_without_screenshot"
    assert "警方发布权威通报" in body["text"]

    search = client.get("/knowledge/search?query=权威 通报 不实&limit=5").json()
    assert any(item["source_url"] == "https://example.com/notice" for item in search)


def test_capture_url_keeps_wikimedia_reference_when_blocked(monkeypatch: Any) -> None:
    class MockGetClient:
        def __init__(self, **_: Any) -> None:
            return None

        def __enter__(self) -> "MockGetClient":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def get(self, url: str) -> httpx.Response:
            request = httpx.Request("GET", url)
            response = httpx.Response(403, request=request)
            raise httpx.HTTPStatusError("403 Forbidden", request=request, response=response)

    monkeypatch.setattr(socket, "getaddrinfo", lambda *_: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("208.80.154.224", 443))])
    monkeypatch.setattr(httpx, "Client", MockGetClient)
    monkeypatch.setattr(evidence_service, "_capture_screenshot", lambda *_: False)

    response = client.post(
        "/cases/group-polarization-003/sources/capture",
        json={"url": "https://commons.wikimedia.org/wiki/File:Sichuan_earthquake_save..JPG"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "captured_without_screenshot"
    assert "Wikimedia" in body["error"]
    assert body["title"] == "Sichuan earthquake save..JPG"
    assert "服务器侧实时抓取被 Wikimedia robot policy 拒绝" in body["text"]


def test_real_analysis_requires_real_inputs() -> None:
    response = client.post("/cases/police-trust-001/real-analysis")
    assert response.status_code == 422
    assert "至少需要上传" in response.json()["detail"]


def test_real_analysis_runs_offline_first_and_returns_local_audits(monkeypatch: Any) -> None:
    case_id = "pytest-real-case-001"
    create_response = client.post(
        "/cases",
        json={
            "id": case_id,
            "title": "网传警情截图引发恐慌",
            "scenario": "涉警公信力谣言",
            "platform": "短视频平台",
            "publish_time": "2026-06-07 10:00",
            "source_url": "https://example.com/rumor",
            "content": "网传某地警方隐瞒警情，要求大家立即转发。",
            "image_description": "待视觉模型真实分析。",
            "spread": {
                "views": 120000,
                "reposts": 3500,
                "comments": 2200,
                "likes": 4500,
                "velocity": "1小时内快速扩散",
            },
            "manual_label": "待人工复核",
            "manual_risk_score": 80,
            "tags": ["涉警", "截图", "待核查"],
            "sensitivity_notes": "评论区出现质疑执法公信力内容。",
            "review_note": "",
        },
    )
    assert create_response.status_code == 200
    upload = client.post(
        f"/cases/{case_id}/assets",
        files={"file": ("screen.png", _png_1x1(), "image/png")},
    )
    assert upload.status_code == 200

    class MockGetResponse:
        content = b"<html><head><title>source</title></head><body>official police statement</body></html>"
        encoding = "utf-8"
        url = "https://example.com/rumor"

        def raise_for_status(self) -> None:
            return None

    class MockGetClient:
        def __init__(self, **_: Any) -> None:
            return None

        def __enter__(self) -> "MockGetClient":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def get(self, url: str) -> MockGetResponse:
            return MockGetResponse()

    monkeypatch.setattr(socket, "getaddrinfo", lambda *_: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))])
    monkeypatch.setattr(httpx, "Client", MockGetClient)
    monkeypatch.setattr(evidence_service, "_capture_screenshot", lambda *_: False)
    snapshot = client.post(
        f"/cases/{case_id}/sources/capture",
        json={"url": "https://example.com/rumor"},
    )
    assert snapshot.status_code == 200

    call_count = {"count": 0}

    class OfflineRealAnalysisClient(MockGetClient):
        def post(self, *_: Any, **__: Any) -> MockResponse:
            call_count["count"] += 1
            raise AssertionError("offline-first real-analysis must not call cloud chat providers")

    monkeypatch.setenv("LOCAL_VISION_BASE_URL", "http://local-vision-offline.test/v1")
    monkeypatch.setattr(httpx, "Client", OfflineRealAnalysisClient)

    response = client.post(f"/cases/{case_id}/real-analysis")

    assert response.status_code == 200
    body = response.json()
    assert call_count["count"] == 0
    assert body["multimodal_results"][0]["audit_id"]
    assert body["multimodal_results"][0]["provider"] == "LocalVision"
    assert body["multimodal_results"][0]["structured"]["skipped_optional_local_vlm"] is True
    assert body["review_audit_id"]
    assert body["report_audit_id"]
    assert body["structured_review"]["review_mode"] == "local_offline"
    assert "本地离线结构化复核" in body["report_markdown"]
    assert body["evidence_chain"]

    audits = client.get(f"/cases/{case_id}/audit").json()
    assert len(audits["invocations"]) >= 3
    providers = {item["provider"] for item in audits["invocations"]}
    assert {"LocalVision", "LocalReview", "LocalReport"}.issubset(providers)
    assert audits["assets"]
    assert audits["snapshots"]

    dataset = client.get(f"/training/local-vision/dataset?case_id={case_id}").json()
    assert dataset["sample_count"] >= 1
    assert dataset["samples"][0]["asset_id"] == upload.json()["id"]
    assert dataset["samples"][0]["manual_label"] == "待人工复核"

    stats_response = client.get(f"/training/local-vision/stats?case_id={case_id}")
    assert stats_response.status_code == 200
    stats = stats_response.json()
    assert stats["sample_count"] >= 1
    assert stats["labeled_sample_count"] >= 1
    assert stats["export_ready"] is True

    jsonl_response = client.get(f"/training/local-vision/dataset.jsonl?case_id={case_id}")
    assert jsonl_response.status_code == 200
    assert case_id in jsonl_response.text
    assert upload.json()["id"] in jsonl_response.text


def test_real_analysis_can_use_explicit_cloud_review_and_report(monkeypatch: Any) -> None:
    case_id = "pytest-real-cloud-enhanced"
    create_response = client.post(
        "/cases",
        json={
            "id": case_id,
            "title": "网传警情截图引发恐慌",
            "scenario": "涉警公信力谣言",
            "platform": "短视频平台",
            "publish_time": "2026-06-07 10:00",
            "source_url": "https://example.com/cloud",
            "content": "网传某地警方隐瞒警情，要求大家立即转发。",
            "image_description": "待视觉模型真实分析。",
            "spread": {
                "views": 120000,
                "reposts": 3500,
                "comments": 2200,
                "likes": 4500,
                "velocity": "1小时内快速扩散",
            },
            "manual_label": "待人工复核",
            "manual_risk_score": 80,
            "tags": ["涉警", "截图", "待核查"],
            "sensitivity_notes": "",
            "review_note": "",
        },
    )
    assert create_response.status_code == 200
    upload = client.post(
        f"/cases/{case_id}/assets",
        files={"file": ("screen.png", _png_1x1(), "image/png")},
    )
    assert upload.status_code == 200

    class MockGetResponse:
        content = b"<html><head><title>source</title></head><body>official police statement</body></html>"
        encoding = "utf-8"
        url = "https://example.com/cloud"

        def raise_for_status(self) -> None:
            return None

    class MockGetClient:
        def __init__(self, **_: Any) -> None:
            return None

        def __enter__(self) -> "MockGetClient":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def get(self, url: str) -> MockGetResponse:
            return MockGetResponse()

    monkeypatch.setattr(socket, "getaddrinfo", lambda *_: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))])
    monkeypatch.setattr(httpx, "Client", MockGetClient)
    monkeypatch.setattr(evidence_service, "_capture_screenshot", lambda *_: False)
    snapshot = client.post(
        f"/cases/{case_id}/sources/capture",
        json={"url": "https://example.com/cloud"},
    )
    assert snapshot.status_code == 200

    _configure_local_vision(monkeypatch)
    _configure_deepseek(monkeypatch)
    _configure_minimax(monkeypatch)
    monkeypatch.setenv("SMARTPOLICE_ENABLE_CLOUD_REVIEW", "1")
    monkeypatch.setenv("SMARTPOLICE_ENABLE_CLOUD_REPORT", "1")
    call_count = {"count": 0}

    class RealAnalysisMockClient(MockHttpxClient):
        def post(
            self,
            url: str,
            headers: dict[str, str],
            json: dict[str, Any],
        ) -> MockResponse:
            call_count["count"] += 1
            MockHttpxClient.last_request = {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": self.timeout,
            }
            if "local-vision" in url:
                content = (
                    '{"ocr_text": ["警情截图"], "visual_facts": ["截图包含警情文字"], '
                    '"aigc_or_tamper_signals": ["字体间距异常"], '
                    '"text_image_consistency": ["与网传文本部分一致"], '
                    '"generator_candidates": [], "uncertainties": ["需核验原图"], "confidence": 0.76}'
                )
            elif "deepseek" in url:
                content = (
                    '{"conclusion": "建议人工复核并核验来源", "risk_level": "较高", '
                    '"risk_score": 82, "key_evidence_ids": ["pytest-real-cloud-enhanced-real-spread"], '
                    '"evidence_conflicts": [], "disposal_suggestions": ["平台协查"], '
                    '"missing_checks": ["原图EXIF"], "human_review_required": true}'
                )
            else:
                content = '{"markdown": "# 正式研判报告\\n\\n引用证据 pytest-real-cloud-enhanced-real-spread。"}'
            return MockResponse({"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 88}})

    monkeypatch.setattr(httpx, "Client", RealAnalysisMockClient)

    response = client.post(f"/cases/{case_id}/real-analysis")

    assert response.status_code == 200
    body = response.json()
    assert call_count["count"] == 3
    assert body["structured_review"]["risk_score"] == 82
    assert "正式研判报告" in body["report_markdown"]


def test_real_analysis_uses_dashscope_vision_when_enabled(monkeypatch: Any) -> None:
    case_id = "pytest-dashscope-vision-only"
    create_response = client.post(
        "/cases",
        json={
            "id": case_id,
            "title": "网传灾情图片核验",
            "scenario": "灾害险情谣言",
            "platform": "短视频平台",
            "publish_time": "2026-06-15 10:00",
            "source_url": "https://example.com/dashscope",
            "content": "网传某地发生坍塌灾情，需核查图片是否真实。",
            "image_description": "待云端 Qwen-VL 视觉描述器分析。",
            "spread": {
                "views": 80000,
                "reposts": 1800,
                "comments": 700,
                "likes": 3200,
                "velocity": "同城群快速转发",
            },
            "manual_label": "待人工复核",
            "manual_risk_score": 70,
            "tags": ["灾情", "图片核验"],
            "sensitivity_notes": "",
            "review_note": "",
        },
    )
    assert create_response.status_code == 200
    upload = client.post(
        f"/cases/{case_id}/assets",
        files={"file": ("scene.png", _png_1x1(), "image/png")},
    )
    assert upload.status_code == 200

    _configure_dashscope(monkeypatch)
    monkeypatch.delenv("VISION_PROVIDER", raising=False)
    monkeypatch.setenv("SMARTPOLICE_ENABLE_LOCAL_VLM", "0")
    calls: list[dict[str, Any]] = []

    class DashScopeVisionMockClient(MockHttpxClient):
        def post(
            self,
            url: str,
            headers: dict[str, str],
            json: dict[str, Any],
        ) -> MockResponse:
            calls.append({"url": url, "headers": headers, "json": json})
            content = (
                '{"ocr_text": ["现场救援"], "visual_facts": ["画面包含救援人员和瓦砾"], '
                '"aigc_or_tamper_signals": ["局部文字需复核"], '
                '"text_image_consistency": ["与灾情描述部分一致"], '
                '"generator_candidates": [], "uncertainties": ["需核验来源"], "confidence": 0.72}'
            )
            return MockResponse({"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 66}})

    monkeypatch.setattr(httpx, "Client", DashScopeVisionMockClient)

    response = client.post(f"/cases/{case_id}/real-analysis")

    assert response.status_code == 200
    body = response.json()
    assert body["multimodal_results"][0]["provider"] == "DashScope"
    assert body["multimodal_results"][0]["selected_model"] == "qwen-vl-test"
    assert body["multimodal_results"][0]["structured"]["visual_facts"]
    assert calls
    assert calls[0]["url"] == "https://dashscope.test/compatible-mode/v1/chat/completions"
    assert calls[0]["json"]["model"] == "qwen-vl-test"


def test_real_analysis_sends_compressed_jpeg_to_vision_model(tmp_path: Path) -> None:
    image_path = tmp_path / "large.png"
    Image.new("RGB", (1800, 1200), color=(30, 80, 140)).save(image_path)
    asset = client.post(
        "/cases/group-polarization-003/assets",
        files={"file": ("large.png", image_path.read_bytes(), "image/png")},
    ).json()

    data_url = real_analysis._image_data_url(CaseAsset.model_validate(asset))

    assert data_url.startswith("data:image/jpeg;base64,")
    encoded = data_url.split(",", 1)[1]
    assert len(encoded) < len(image_path.read_bytes()) * 2


def test_localvision_training_rejects_insufficient_labeled_samples() -> None:
    response = client.post(
        "/training/local-vision/run",
        json={
            "epochs": 50,
            "learning_rate": 0.04,
            "l2": 0.02,
            "min_samples": 200,
        },
    )

    assert response.status_code == 400
    assert "本地视觉校准训练至少需要" in response.json()["detail"]


def test_localvision_calibrator_trains_and_participates_in_real_analysis(monkeypatch: Any) -> None:
    samples = [
        (
            "pytest-localvision-train-001",
            28,
            '{"ocr_text": ["普通提示"], "visual_facts": ["截图为日常提醒"], '
            '"aigc_or_tamper_signals": [], "text_image_consistency": ["图文基本一致"], '
            '"generator_candidates": [], "uncertainties": ["需核验发布时间"], "confidence": 0.62}',
        ),
        (
            "pytest-localvision-train-002",
            54,
            '{"ocr_text": ["警情提醒"], "visual_facts": ["截图包含转发提示"], '
            '"aigc_or_tamper_signals": ["局部压缩异常"], "text_image_consistency": ["与网传文字部分一致"], '
            '"generator_candidates": [{"model_family": "通用图像编辑", "confidence": 0.33}], '
            '"uncertainties": ["需核验原图"], "confidence": 0.7}',
        ),
        (
            "pytest-localvision-train-003",
            78,
            '{"ocr_text": ["警方通报", "立即转发"], "visual_facts": ["截图包含警情措辞"], '
            '"aigc_or_tamper_signals": ["字体间距异常", "边缘拼接痕迹"], '
            '"text_image_consistency": ["与网传文本高度一致"], '
            '"generator_candidates": [{"model_family": "图像编辑", "confidence": 0.58}], '
            '"uncertainties": ["需核验发布主体"], "confidence": 0.81}',
        ),
        (
            "pytest-localvision-train-004",
            91,
            '{"ocr_text": ["突发警情", "隐瞒真相", "线下集合"], "visual_facts": ["截图带有煽动性措辞"], '
            '"aigc_or_tamper_signals": ["字体错位", "水印异常", "拼接痕迹"], '
            '"text_image_consistency": ["图文一致且传播指向明确"], '
            '"generator_candidates": [{"model_family": "图像生成/编辑", "confidence": 0.73}], '
            '"uncertainties": ["需调取原始发布链路"], "confidence": 0.88}',
        ),
    ]
    for case_id, score, visual_content in samples:
        _create_localvision_training_sample(
            monkeypatch,
            case_id=case_id,
            manual_score=score,
            visual_content=visual_content,
        )

    run_response = client.post(
        "/training/local-vision/run",
        json={
            "epochs": 80,
            "learning_rate": 0.04,
            "l2": 0.02,
            "min_samples": 4,
            "activation_mode": "activate",
        },
    )

    assert run_response.status_code == 200
    run = run_response.json()
    assert run["status"] == "trained"
    assert run["model_kind"] == "local-vision-risk-calibrator-ridge-v1"
    assert run["sample_count"] >= 4
    assert run["feature_count"] > 5
    assert "LocalVision JSON 特征" in run["model_card"]["architecture"]

    status_response = client.get("/training/local-vision/status")
    assert status_response.status_code == 200
    status = status_response.json()
    assert status["trained"] is True
    assert status["active_model_id"] == run["id"]
    assert status["dataset"]["training_ready"] is True

    result = _create_localvision_training_sample(
        monkeypatch,
        case_id="pytest-localvision-runtime-001",
        manual_score=83,
        visual_content=(
            '{"ocr_text": ["警方隐瞒", "马上转发"], "visual_facts": ["警情截图含煽动转发语"], '
            '"aigc_or_tamper_signals": ["字体间距异常", "局部拼接"], '
            '"text_image_consistency": ["与网传文本一致"], '
            '"generator_candidates": [{"model_family": "图像编辑", "confidence": 0.69}], '
            '"uncertainties": ["需核验原图和发布主体"], "confidence": 0.84}'
        ),
    )
    structured = result["analysis"]["multimodal_results"][0]["structured"]
    assert "local_vision_calibration" in structured
    calibration = structured["local_vision_calibration"]
    assert calibration["model_id"] == run["id"]
    assert 0 <= calibration["score"] <= 100
    assert calibration["risk_level"] in {"低", "关注", "较高", "紧急"}
    assert calibration["explanations"]


def test_case_llm_review_uses_real_gateway_and_returns_audit(monkeypatch: Any) -> None:
    _configure_deepseek(monkeypatch)
    _mock_httpx_success(
        monkeypatch,
        content=(
            '{"conclusion": "建议升级人工复核", "risk_level": "较高", '
            '"risk_score": 82, "key_evidence": ["传播态势"], '
            '"disposal_boundary": ["避免直接定性"], '
            '"human_review_required": true, "missing_checks": ["原始素材"]}'
        ),
    )

    response = client.post(
        "/cases/group-polarization-003/llm-review",
        json={"provider": "DeepSeek", "role": "复核器", "temperature": 0.1},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["case_id"] == "group-polarization-003"
    assert body["audit_id"]
    assert body["structured_review"]["human_review_required"] is True
    assert MockHttpxClient.last_request is not None
    prompt = MockHttpxClient.last_request["json"]["messages"][-1]["content"]
    assert "知识依据" in prompt
    assert "证据链" in prompt


def test_case_llm_report_returns_markdown_and_refs(monkeypatch: Any) -> None:
    _configure_minimax(monkeypatch)
    _mock_httpx_success(
        monkeypatch,
        content='{"markdown": "# 研判报告\\n\\n## 人工复核声明\\n需人工复核。"}',
    )

    response = client.post(
        "/cases/group-polarization-003/llm-report",
        json={"provider": "MiniMax", "role": "中文业务生成", "temperature": 0.1},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["audit_id"]
    assert "研判报告" in body["markdown"]
    assert body["knowledge_refs"]


def test_case_llm_report_invalid_json_returns_502(monkeypatch: Any) -> None:
    _configure_minimax(monkeypatch)
    _mock_httpx_success(monkeypatch, content="不是 JSON")

    response = client.post(
        "/cases/group-polarization-003/llm-report",
        json={"provider": "MiniMax", "role": "中文业务生成", "temperature": 0.1},
    )

    assert response.status_code == 502


def test_training_rejects_demo_cases_as_training_data() -> None:
    _reset_training_pool()
    response = client.post(
        "/training/run",
        json={"epochs": 120, "learning_rate": 0.04, "include_augmented_samples": False},
    )

    assert response.status_code == 400
    assert "内置四方向样例仅用于展示评测" in response.json()["detail"]

    status_response = client.get("/training/datasets/status")
    assert status_response.status_code == 200
    status = status_response.json()
    assert status["demo_case_count"] == 4
    assert status["training_ready"] is False


def test_import_external_dataset_train_and_analyze_showcase_case() -> None:
    _reset_training_pool()
    import_response = client.post(
        "/training/datasets/import",
        json={
            "dataset_name": "pytest-hf-rumor-fixture",
            "source": "HuggingFace fixture",
            "source_url": "https://huggingface.co/datasets/FinanceMTEB/MDFEND-Weibo21",
            "rows": [
                {
                    "title": "警方隐瞒商场冲突伤亡为虚假消息",
                    "text": "网传警方隐瞒商场冲突多人受伤，号召立即转发集合，经核验为拼接截图和夸大叙事。",
                    "label": "rumor",
                    "scenario": "涉警公信力谣言",
                },
                {
                    "title": "山区校车被困灾害消息不实",
                    "text": "社交平台传播南部山区塌方多辆校车被困，配图为旧图嫁接，造成集中报警和恐慌。",
                    "label": "fake",
                    "scenario": "灾害险情谣言",
                },
                {
                    "title": "商圈性别冲突截图为伪造",
                    "text": "论坛帖子用伪造聊天截图煽动群体对立，评论区出现线下集合和网暴动员。",
                    "label": "1",
                    "scenario": "群体对立煽动型谣言",
                },
                {
                    "title": "本地交通绕行提示属实",
                    "text": "本地生活群转发交通管制绕行提示，经官方通告确认，传播范围小，无明显煽动。",
                    "label": "real",
                    "scenario": "低风险误传",
                },
            ],
            "text_columns": ["title", "text"],
            "label_column": "label",
            "scenario_column": "scenario",
        },
    )
    assert import_response.status_code == 200
    imported = import_response.json()
    assert imported["imported_count"] == 4
    assert imported["sample_count_after_import"] >= 4

    dataset_status_response = client.get("/training/datasets/status")
    assert dataset_status_response.status_code == 200
    dataset_status = dataset_status_response.json()
    assert dataset_status["external_sample_count"] >= 4
    assert dataset_status["training_ready"] is True
    assert "MDFEND-Weibo21" in dataset_status["recommended_huggingface_datasets"][0]["name"]

    training_response = client.post(
        "/training/run",
        json={"epochs": 120, "learning_rate": 0.04, "include_augmented_samples": True},
    )
    assert training_response.status_code == 200
    training = training_response.json()
    assert training["status"] == "trained"
    assert training["model_kind"] == "competition-local-hybrid-ngram-ridge-v3"
    assert training["feature_count"] > 40
    assert "中文字符 n-gram" in training["model_card"]["architecture"]
    assert "内置四方向样例不参与训练" in training["model_card"]["training_data"]
    source_summary = training["model_card"]["training_source_summary"]
    assert source_summary["external_samples"] >= 4
    assert source_summary["excluded_demo_cases"] == 4
    assert training["task_metrics"]["risk_level_classification"]["labels"] == ["低", "关注", "较高", "紧急"]
    assert any(item["name"].startswith("ngram::") for item in training["top_positive_features"] + training["top_negative_features"])

    status_response = client.get("/training/status")
    assert status_response.status_code == 200
    status = status_response.json()
    assert status["trained"] is True
    assert status["active_model_id"] == training["id"]
    assert status["training_data"]["demo_case_count"] == 4

    analysis_response = client.post("/analysis/full", json={"case_id": "group-polarization-003"})
    assert analysis_response.status_code == 200
    analysis = analysis_response.json()
    assert analysis["risk"]["model_version_id"] == training["id"]
    assert analysis["risk"]["model_score"] is not None


def test_import_external_vision_dataset_skips_duplicate_image_sha() -> None:
    _reset_training_pool()
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = Path(temp_dir) / "same-image.png"
        Image.new("RGB", (16, 16), color=(120, 80, 40)).save(image_path)

        response = client.post(
            "/training/datasets/import",
            json={
                "dataset_name": "pytest-duplicate-image-fixture",
                "source": "pytest fixture",
                "rows": [
                    {
                        "image": "same-image.png",
                        "caption": "first generator sample",
                        "label": "gpt-image2",
                    },
                    {
                        "image": "same-image.png",
                        "caption": "same bytes under another row",
                        "label": "gpt-image2",
                    },
                ],
                "task_type": "vision_generator_attribution",
                "image_root": temp_dir,
                "image_path_column": "image",
                "text_columns": ["caption"],
                "label_column": "label",
                "risk_score_column": None,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["imported_count"] == 1
    assert payload["skipped_count"] == 1
    assert payload["sample_count_after_import"] == 1
    assert payload["label_distribution"] == {"gpt-image2": 1}


def test_create_label_train_and_analyze_user_case() -> None:
    _reset_training_pool()
    case_id = "pytest-user-case-001"
    user_samples = [
        (
            case_id,
            "AI合成商场冲突截图引发线下集合",
            "群体对立煽动型谣言",
            "网传某商场发生冲突，称警方偏袒一方，号召网友立即线下集合声援。",
            "聊天截图字体间距异常，头像重复，现场照片人物纹理异常。",
            86,
            ["群体对立", "AI合成", "线下聚集"],
        ),
        (
            "pytest-user-case-002",
            "旧图包装成本地山洪险情",
            "灾害险情谣言",
            "社交平台称本地山洪冲毁道路并要求立即转发，配图为外地旧图。",
            "图片道路标识与本地不符，压缩痕迹明显。",
            90,
            ["灾害", "旧图", "恐慌"],
        ),
        (
            "pytest-user-case-003",
            "涉警执法剪辑视频断章取义",
            "涉警公信力谣言",
            "短视频剪辑称民警暴力执法并隐瞒记录，评论区出现强烈负面情绪。",
            "视频关键帧缺失，字幕遮挡执法记录仪时间。",
            78,
            ["涉警", "剪辑", "公信力"],
        ),
        (
            "pytest-user-case-004",
            "低传播交通提示误传",
            "低风险误传",
            "本地群转发道路临时绕行提示，后经核验基本属实且传播范围有限。",
            "配图为普通道路截图，无明显合成痕迹。",
            18,
            ["交通", "低传播", "误传"],
        ),
    ]
    for sample_id, title, scenario, content, image_description, score, tags in user_samples:
        create_response = client.post(
            "/cases",
            json={
                "id": sample_id,
                "title": title,
                "scenario": scenario,
                "platform": "短视频平台",
                "publish_time": "2026-06-04 11:20",
                "source_url": "本地测试样本",
                "content": content,
                "image_description": image_description,
                "spread": {
                    "views": 98000 if score >= 60 else 1200,
                    "reposts": 3600 if score >= 60 else 20,
                    "comments": 5200 if score >= 60 else 12,
                    "likes": 6800 if score >= 60 else 40,
                    "velocity": "30分钟内跨群快速扩散" if score >= 60 else "小范围缓慢传播",
                },
                "manual_label": "人工标注训练样本",
                "manual_risk_score": score,
                "tags": tags,
                "sensitivity_notes": "pytest user training sample",
                "review_note": "pytest seed",
            },
        )
        assert create_response.status_code in {200, 400}

    label_response = client.post(
        f"/cases/{case_id}/label",
        json={
            "manual_risk_score": 88,
            "manual_label": "人工复核：高风险",
            "review_note": "进入训练集",
        },
    )
    assert label_response.status_code == 200
    assert label_response.json()["manual_risk_score"] == 88

    cases_response = client.get("/cases")
    assert cases_response.status_code == 200
    assert any(item["id"] == case_id for item in cases_response.json())

    training_response = client.post(
        "/training/run",
        json={"epochs": 120, "learning_rate": 0.04, "include_augmented_samples": True},
    )
    assert training_response.status_code == 200
    training = training_response.json()
    assert training["status"] == "trained"
    assert training["model_kind"] == "competition-local-hybrid-ngram-ridge-v3"
    assert training["feature_count"] > 40
    assert "中文字符 n-gram" in training["model_card"]["architecture"]
    assert training["task_metrics"]["risk_level_classification"]["labels"] == ["低", "关注", "较高", "紧急"]
    assert any(item["name"].startswith("ngram::") for item in training["top_positive_features"] + training["top_negative_features"])
    assert training["top_positive_features"]

    status_response = client.get("/training/status")
    assert status_response.status_code == 200
    status = status_response.json()
    assert status["trained"] is True
    assert status["active_model_id"] == training["id"]
    assert status["training_data"]["external_sample_count"] == 0
    assert status["training_data"]["labeled_user_case_count"] >= 4

    analysis_response = client.post("/analysis/full", json={"case_id": case_id})
    assert analysis_response.status_code == 200
    analysis = analysis_response.json()
    assert analysis["risk"]["model_version_id"] == training["id"]
    assert analysis["risk"]["model_score"] is not None
    assert analysis["risk"]["model_explanation"]
    skill_names = {skill["name"] for skill in analysis["agent"]["recommended_skills"]}
    assert skill_names
    assert {"source_verification_skill", "risk_evolution_skill", "joint_disposal_skill"} & skill_names


def _write_png_fixture(path: Path, marker: bytes = b"") -> None:
    path.write_bytes(_png_1x1() + marker)


def _write_real_png_fixture(path: Path, index: int) -> None:
    image = Image.new(
        "RGB",
        (96, 80),
        (
            (50 + index * 43) % 255,
            (90 + index * 61) % 255,
            (130 + index * 37) % 255,
        ),
    )
    draw = ImageDraw.Draw(image)
    for step in range(0, 96, 12):
        color = (
            (step * 3 + index * 29) % 255,
            (220 - step * 2 + index * 11) % 255,
            (40 + step + index * 19) % 255,
        )
        draw.line((step, 0, 95 - step // 2, 79), fill=color, width=2)
    draw.rectangle((8 + index, 10, 44 + index * 2, 42), outline=(255, 255, 255), width=2)
    draw.ellipse((48, 18 + index, 84, 54 + index), outline=(20, 20, 20), width=2)
    draw.text((8, 62), f"S{index}", fill=(255, 255, 255))
    image.save(path, format="PNG")


def _import_cross_source_generator_fixture(tmp_path: Path) -> tuple[Path, list[str]]:
    image_root = tmp_path / "cross-source-generator-images"
    image_root.mkdir()
    rows = []
    labels = ["gpt-image2", "real", "midjourney"]
    for source_id in range(3):
        for label_index, label in enumerate(labels):
            index = source_id * len(labels) + label_index
            name = f"cross-source-{index}.png"
            _write_real_png_fixture(image_root / name, index + 120)
            rows.append(
                {
                    "dataset_name": f"pytest-cross-source-{source_id}",
                    "source": f"fixture-origin-{source_id}",
                    "source_url": f"https://huggingface.test/datasets/cross-source-{source_id}",
                    "image": name,
                    "caption": (
                        "platform repost screenshot compression watermark"
                        if label != "real"
                        else "ordinary camera photo social repost"
                    ),
                    "label": label,
                    "scenario": "跨来源泛化测试夹具",
                }
            )
    import_response = client.post(
        "/training/datasets/import",
        json={
            "dataset_name": "pytest-cross-source-generator",
            "source": "fallback source",
            "task_type": "vision_generator_attribution",
            "image_root": str(image_root),
            "image_path_column": "image",
            "text_columns": ["caption"],
            "label_column": "label",
            "rows": rows,
        },
    )
    assert import_response.status_code == 200
    assert import_response.json()["imported_count"] == len(rows)
    return image_root, labels


def test_balanced_generator_sampling_round_robins_sources_within_label() -> None:
    samples: list[ExternalTrainingSample] = []

    def sample(index: int, label: str, source_id: str) -> ExternalTrainingSample:
        return ExternalTrainingSample(
            id=f"sample-{index}",
            dataset_name=f"dataset-{source_id}",
            source=f"source-{source_id}",
            task_type="vision_generator_attribution",
            title=f"{label} sample {index}",
            content="source-balanced sampling fixture",
            image_path=f"D:/tmp/sample-{index}.png",
            image_available=True,
            label=label,
            risk_score=50,
            scenario="sampling",
            created_at="2026-06-12T00:00:00+00:00",
        )

    for index in range(10):
        samples.append(sample(index, "real", "real-a"))
    samples.append(sample(10, "real", "real-b"))
    samples.append(sample(11, "real", "real-c"))
    for index in range(12, 24):
        samples.append(sample(index, "gpt-image2", "gpt-a"))

    selected = _balanced_generator_samples(samples, limit=8)
    real_sources = {
        _source_holdout_group_name(item, "dataset_source")
        for item in selected
        if item.label == "real"
    }

    assert len(real_sources) >= 2


def test_mainstream_five_sampling_balances_mapped_unknown_bucket() -> None:
    samples: list[ExternalTrainingSample] = []

    def sample(index: int, label: str, source_id: str) -> ExternalTrainingSample:
        return ExternalTrainingSample(
            id=f"mainstream-sample-{index}",
            dataset_name=f"dataset-{source_id}",
            source=f"source-{source_id}",
            task_type="vision_generator_attribution",
            title=f"{label} sample {index}",
            content="mainstream five sampling fixture",
            image_path=f"D:/tmp/mainstream-{index}.png",
            image_available=True,
            label=label,
            risk_score=50,
            scenario="sampling",
            created_at="2026-06-12T00:00:00+00:00",
        )

    index = 0
    for label in ("gpt-image2", "nano-banana", "seedream-4", "midjourney", "real"):
        for source in ("a", "b"):
            for _ in range(6):
                samples.append(sample(index, label, f"{label}-{source}"))
                index += 1
    for label in ("sdxl", "sd3", "sd21"):
        for source in ("a", "b"):
            for _ in range(6):
                samples.append(sample(index, label, f"{label}-{source}"))
                index += 1
    for label in ("flux", "dall-e-3", "gpt-image1"):
        for source in ("a", "b", "c"):
            for _ in range(10):
                samples.append(sample(index, label, f"{label}-{source}"))
                index += 1

    selected = _balanced_generator_samples_for_request(
        samples,
        limit=70,
        request=VisionTrainingRunRequest(
            task_type="vision_generator_attribution",
            experiment_profile="mainstream_five_attribution",
            max_training_samples=70,
        ),
    )
    _, _, labels, _ = _generator_experiment_view(
        selected,
        [{"feature": float(idx)} for idx in range(len(selected))],
        VisionTrainingRunRequest(
            task_type="vision_generator_attribution",
            experiment_profile="mainstream_five_attribution",
        ),
    )
    counts = Counter(labels)

    assert counts["unknown"] <= max(
        counts["gpt-image2"],
        counts["nano-banana"],
        counts["seedream-4"],
        counts["stable-diffusion"],
        counts["midjourney"],
        counts["real"],
    ) + 1
    assert counts["stable-diffusion"] >= counts["unknown"] - 1


def test_gpt_image2_ovr_sampling_balances_target_buckets_and_sources() -> None:
    samples: list[ExternalTrainingSample] = []

    def sample(index: int, label: str, dataset: str, source: str) -> ExternalTrainingSample:
        return ExternalTrainingSample(
            id=f"gpt-ovr-{index}",
            dataset_name=dataset,
            source=source,
            task_type="vision_generator_attribution",
            title=f"{label} source-balanced sample",
            content="source balance test",
            image_path=f"D:/tmp/gpt-ovr-{index}.png",
            image_available=True,
            label=label,
            risk_score=82 if label != "real" else 18,
            scenario="profile sampling test",
            created_at="2026-06-13T00:00:00+00:00",
        )

    index = 0
    for dataset, count in (
        ("Scam-AI/gpt-image-2", 12),
        ("LukaDev13/Liminal-Dreamcore-1K", 8),
        ("Qwen/Qwen-Image-Bench", 4),
    ):
        for _ in range(count):
            samples.append(sample(index, "gpt-image2", dataset, "train"))
            index += 1
    for label, count in (("midjourney", 18), ("sdxl", 18), ("dall-e-3", 18), ("real", 30)):
        for _ in range(count):
            samples.append(sample(index, label, f"{label}-dataset", "train"))
            index += 1

    selected = _balanced_gpt_image2_ovr_samples(samples, limit=30)
    mapped = Counter(
        "gpt-image2"
        if item.label == "gpt-image2"
        else "real"
        if item.label == "real"
        else "other-generated"
        for item in selected
    )

    assert mapped == {"gpt-image2": 10, "other-generated": 10, "real": 10}
    first_gpt_sources = [
        _source_holdout_group_name(item, "dataset_source")
        for item in selected
        if item.label == "gpt-image2"
    ][:6]
    assert len(set(first_gpt_sources)) == 3


def test_gpt_image2_ovr_source_guard_removes_source_artifact_features(monkeypatch: Any) -> None:
    monkeypatch.setenv("SMARTPOLICE_GENERATOR_SOURCE_GUARD", "1")
    policy = _generator_profile_feature_policy(
        [
            "image_bytes_log",
            "image_megapixels",
            "jpg_ext",
            "clip_txt_00",
            "clip_gap_00",
            "texture_residual_std",
            "frequency_high_energy_proxy",
            "edge_density",
            "compression_residual_std",
            "jpeg_block_boundary_delta",
            "pixel_luma_mean",
            "pixel_saturation_std",
            "horizontal_gradient_energy",
            "vertical_gradient_energy",
            "pixel_red_mean",
            "pixel_green_mean",
            "pixel_blue_mean",
            "text_overlay_edge_density",
            "corner_watermark_edge_signal",
        ],
        "gpt_image2_ovr",
    )

    assert policy["source_guard_enabled"] is True
    assert policy["strategy"] == "source_artifact_guard"
    assert "image_megapixels" in policy["removed_feature_names"]
    assert "clip_txt_00" in policy["removed_feature_names"]
    assert "texture_residual_std" in policy["feature_names"]


def test_open_set_unknown_threshold_multiplier_is_request_scoped() -> None:
    request = VisionTrainingRunRequest(
        task_type="vision_generator_attribution",
        experiment_profile="gpt_image2_ovr",
        enable_open_set_unknown=True,
        unknown_threshold_multiplier=1.5,
        open_set_min_margin=0.08,
    )

    assert _open_set_unknown_threshold(0.2, request) == 0.3


def test_classifier_open_set_margin_rejects_ambiguous_non_real(tmp_path: Path) -> None:
    import pickle

    from sklearn.ensemble import ExtraTreesClassifier

    model = ExtraTreesClassifier(n_estimators=8, random_state=42)
    model.fit(
        [
            [0.0],
            [0.1],
            [0.9],
            [1.0],
        ],
        ["gpt-image2", "gpt-image2", "other-generated", "other-generated"],
    )
    model_path = tmp_path / "ambiguous.pkl"
    with model_path.open("wb") as file:
        pickle.dump(model, file)

    prediction = _predict_generator_with_classifier(
        str(model_path),
        {"x": 0.48},
        ["x"],
        unknown_threshold=0.01,
        open_set_min_margin=0.95,
    )

    assert prediction is not None
    assert prediction["label"] == "unknown"
    assert "low_top2_margin" in prediction["unknown_reasons"]


def test_generator_binary_gate_threshold_prioritizes_real_fpr() -> None:
    threshold, diagnostics = _generator_binary_gate_threshold(
        real_probabilities=[0.12, 0.18, 0.21, 0.62, 0.68],
        generated_probabilities=[0.74, 0.78, 0.82, 0.88, 0.91],
    )

    assert threshold >= 0.7
    assert diagnostics["strategy"] == "real_fpr_first_threshold_search"
    assert diagnostics["training_real_false_positive_rate"] <= 0.05
    assert diagnostics["training_generated_recall"] >= 0.6


def test_binary_gate_policy_uses_conservative_real_guard() -> None:
    policy = _generator_binary_gate_policy("binary_generated_gate")

    assert policy["target_real_fpr"] <= 0.07
    assert policy["real_guard_quantile"] >= 0.88
    assert policy["real_protection_margin"] >= 0.08


def test_source_balanced_weights_boost_real_benchmark_hard_negatives() -> None:
    weights = _source_balanced_sample_weights(
        ["real", "real", "generated", "generated"],
        [0, 1, 2, 3],
        [
            "TheKernel01/AIGC-Detection-Benchmark|test",
            "ordinary-real-source|train",
            "generated-a|train",
            "generated-b|train",
        ],
        real_weight_multiplier=1.65,
        hard_negative_multiplier=1.85,
    )

    assert weights is not None
    assert weights[0] > weights[1]
    assert _is_real_hard_negative_source("TheKernel01/AIGC-Detection-Benchmark|test")


def test_source_balanced_weights_boost_generated_weak_sources() -> None:
    weights = _source_balanced_sample_weights(
        ["generated", "generated", "real", "real"],
        [0, 1, 2, 3],
        [
            "Rajarshi-Roy-research/Defactify_Image_Dataset|train",
            "ordinary-generated-source|train",
            "TheKernel01/AIGC-Detection-Benchmark|test",
            "ordinary-real-source|train",
        ],
        real_weight_multiplier=1.65,
        hard_negative_multiplier=1.85,
        generated_hard_positive_multiplier=1.45,
    )

    assert weights is not None
    assert weights[0] > weights[1]
    assert weights[2] > weights[3]
    assert _is_generated_hard_positive_source("marco-willi/synthbuster-plus|train")


def test_source_balanced_weights_resolve_full_source_keys_by_train_index() -> None:
    weights = _source_balanced_sample_weights(
        ["generated", "generated", "real", "real"],
        [10, 11, 12, 13],
        [
            *["ordinary-prefix-source|train"] * 10,
            "Rajarshi-Roy-research/Defactify_Image_Dataset|train",
            "ordinary-generated-source|train",
            "TheKernel01/AIGC-Detection-Benchmark|test",
            "ordinary-real-source|train",
        ],
        real_weight_multiplier=1.65,
        hard_negative_multiplier=1.85,
        generated_hard_positive_multiplier=1.45,
    )

    assert weights is not None
    assert weights[0] > weights[1]
    assert weights[2] > weights[3]


def test_generator_binary_gate_below_threshold_guards_as_real() -> None:
    prediction = _apply_generator_binary_gate(
        {
            "label": "gpt-image2",
            "raw_label": "gpt-image2",
            "confidence": 0.82,
        },
        {
            "generated_probability": GENERATOR_BINARY_GATE_THRESHOLD - 0.05,
            "real_probability": 1.0 - GENERATOR_BINARY_GATE_THRESHOLD + 0.05,
        },
        generated_gate_threshold=GENERATOR_BINARY_GATE_THRESHOLD,
        real_protection_margin=GENERATOR_REAL_PROTECTION_MARGIN,
    )

    assert prediction["label"] == "real"
    assert prediction["gate_reason"] == "binary_gate_below_generated_threshold_real_guard"
    recommendation = prediction["binary_gate"]["review_recommendation"]
    assert recommendation["level"] == "manual_review_generated_signal"
    assert recommendation["review_threshold"] < recommendation["strong_threshold"]


def test_generator_binary_gate_allows_high_confidence_gpt_image2_near_threshold() -> None:
    prediction = _apply_generator_binary_gate(
        {
            "label": "gpt-image2",
            "raw_label": "gpt-image2",
            "confidence": 0.86,
        },
        {
            "generated_probability": 0.56,
            "real_probability": 0.44,
        },
        generated_gate_threshold=0.66,
        real_protection_margin=GENERATOR_REAL_PROTECTION_MARGIN,
    )

    assert prediction["label"] == "gpt-image2"
    assert prediction["gate_reason"] == "binary_gate_gpt_image2_high_confidence_override"
    assert prediction["binary_gate"]["review_recommendation"]["level"] == "manual_review_generated_signal"


def test_generator_binary_gate_can_override_raw_real_when_generated_is_strong() -> None:
    prediction = _apply_generator_binary_gate(
        {
            "label": "real",
            "raw_label": "real",
            "confidence": 0.74,
        },
        {
            "generated_probability": 0.86,
            "real_probability": 0.14,
        },
        generated_gate_threshold=0.66,
        real_protection_margin=0.06,
    )

    assert prediction["label"] == "generated"
    assert prediction["gate_reason"] == "binary_gate_generated_override"
    assert prediction["binary_gate"]["review_recommendation"]["level"] == "generated_strong"


def test_generator_binary_gate_keeps_raw_real_without_strong_generated_signal() -> None:
    prediction = _apply_generator_binary_gate(
        {
            "label": "real",
            "raw_label": "real",
            "confidence": 0.74,
        },
        {
            "generated_probability": 0.68,
            "real_probability": 0.32,
        },
        generated_gate_threshold=0.66,
        real_protection_margin=0.06,
    )

    assert prediction["label"] == "real"
    assert "gate_reason" not in prediction
    assert prediction["binary_gate"]["review_recommendation"]["level"] == "manual_review_generated_signal"


def test_generator_experiment_profiles_remap_and_filter_labels() -> None:
    samples: list[ExternalTrainingSample] = []

    def sample(index: int, label: str, dataset: str, source: str, content: str = "") -> ExternalTrainingSample:
        return ExternalTrainingSample(
            id=f"profile-{index}",
            dataset_name=dataset,
            source=source,
            task_type="vision_generator_attribution",
            title=f"{label} profile sample",
            content=content or "clean benchmark sample",
            image_path=f"D:/tmp/profile-{index}.png",
            image_available=True,
            label=label,
            risk_score=50,
            scenario="profile test",
            created_at="2026-06-12T00:00:00+00:00",
        )

    samples.extend(
        [
            sample(0, "gpt-image2", "Scam-AI/gpt-image-2", "train"),
            sample(1, "gpt-image2", "Qwen/Qwen-Image-Bench", "test"),
            sample(2, "real", "real-negative-pool", "real-negative-pool"),
            sample(3, "midjourney", "dataset-a", "source-a"),
            sample(4, "midjourney", "dataset-b", "source-b"),
            sample(5, "stable-diffusion", "single-source", "source-c"),
        ]
    )
    rows = [{"feature": float(index)} for index in range(len(samples))]

    _, _, gpt_labels, gpt_report = _generator_experiment_view(
        samples,
        rows,
        VisionTrainingRunRequest(
            task_type="vision_generator_attribution",
            experiment_profile="gpt_image2_ovr",
        ),
    )
    assert Counter(gpt_labels) == {"gpt-image2": 2, "real": 1, "other-generated": 3}
    assert "GPT-image2" in gpt_report["label_policy"]

    _, _, multi_labels, multi_report = _generator_experiment_view(
        samples,
        rows,
        VisionTrainingRunRequest(
            task_type="vision_generator_attribution",
            experiment_profile="multi_generator_label_covered",
        ),
    )
    assert "midjourney" in multi_labels
    assert "stable-diffusion" not in multi_labels
    assert multi_report["label_distribution"]["unknown"] >= 1

    mainstream_samples = [
        sample(10, "gpt-image2", "Scam-AI/gpt-image-2", "train"),
        sample(11, "nano-banana", "Rapidata/bananamark", "train"),
        sample(12, "seedream-4", "Qwen/Qwen-Image-Bench", "train"),
        sample(13, "sdxl", "Synthbuster", "train"),
        sample(14, "sd3", "GenImage", "train"),
        sample(15, "midjourney", "dataset-a", "source-a"),
        sample(16, "flux", "dataset-a", "source-a"),
        sample(17, "dall-e-3", "dataset-a", "source-a"),
        sample(18, "real", "real-negative-pool", "real-negative-pool"),
    ]
    _, _, mainstream_labels, mainstream_report = _generator_experiment_view(
        mainstream_samples,
        [{"feature": float(index)} for index in range(len(mainstream_samples))],
        VisionTrainingRunRequest(
            task_type="vision_generator_attribution",
            experiment_profile="mainstream_five_attribution",
        ),
    )
    assert Counter(mainstream_labels) == {
        "gpt-image2": 1,
        "nano-banana": 1,
        "seedream-4": 1,
        "stable-diffusion": 2,
        "midjourney": 1,
        "unknown": 2,
        "real": 1,
    }
    assert "五类主流来源" in mainstream_report["label_policy"]


def test_generator_profile_policy_has_track_gates_and_isolated_copy() -> None:
    policy = _generator_profile_policy("binary_generated_gate")

    assert policy["chinese_name"] == "真实/生成鲁棒初筛"
    assert policy["candidate_only"] is True
    assert policy["activation_eligibility"] == "component_candidate"
    assert any(
        gate["metric"] == "source_real_false_positive_rate"
        and gate["operator"] == "<="
        and gate["threshold"] == 0.10
        for gate in policy["acceptance_gates"]
    )

    policy["chinese_name"] = "mutated"
    assert _generator_profile_policy("binary_generated_gate")["chinese_name"] == "真实/生成鲁棒初筛"


def test_import_multimodal_dataset_tracks_task_images_and_hashes(tmp_path: Path) -> None:
    _reset_training_pool()
    image_root = tmp_path / "images"
    image_root.mkdir()
    for index in range(4):
        _write_png_fixture(image_root / f"sample-{index}.png", marker=f"marker-{index}".encode())

    response = client.post(
        "/training/datasets/import",
        json={
            "dataset_name": "pytest-tiny-genimage",
            "source": "local fixture",
            "task_type": "vision_aigc",
            "image_root": str(image_root),
            "image_path_column": "image",
            "text_columns": ["caption"],
            "label_column": "label",
            "label_schema": {"ai": 88, "real": 12},
            "rows": [
                {"image": "sample-0.png", "caption": "AI 生成警情图", "label": "ai"},
                {"image": "sample-1.png", "caption": "真实普通图片", "label": "real"},
                {"image": "sample-2.png", "caption": "AIGC 合成图", "label": "ai"},
                {"image": "sample-3.png", "caption": "普通现场照片", "label": "real"},
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task_type"] == "vision_aigc"
    assert body["imported_count"] == 4
    assert body["image_available_count"] == 4
    assert body["examples"][0]["image_sha256"]
    assert body["examples"][0]["task_type"] == "vision_aigc"

    status = client.get("/training/datasets/status?task_type=vision_aigc").json()
    assert status["external_sample_count"] == 4
    assert status["tasks"][0]["image_available_count"] == 4

    samples = client.get("/training/datasets/samples?task_type=vision_aigc&limit=2").json()
    assert len(samples) == 2
    assert all(item["image_available"] for item in samples)


def test_import_dataset_can_preserve_row_level_huggingface_source(tmp_path: Path) -> None:
    _reset_training_pool()
    image_root = tmp_path / "hf-source-images"
    image_root.mkdir()
    _write_png_fixture(image_root / "gpt2.png", marker=b"gpt-image2")
    _write_png_fixture(image_root / "real.png", marker=b"real")

    response = client.post(
        "/training/datasets/import",
        json={
            "dataset_name": "fallback-dataset",
            "source": "fallback-source",
            "task_type": "vision_generator_attribution",
            "image_root": str(image_root),
            "image_path_column": "image",
            "text_columns": ["caption", "source_detail"],
            "label_column": "label",
            "rows": [
                {
                    "dataset_name": "Qwen/Qwen-Image-Bench",
                    "source": "Qwen/Qwen-Image-Bench:test",
                    "source_detail": "Qwen/Qwen-Image-Bench:test:gpt-image-2",
                    "source_url": "https://huggingface.co/datasets/Qwen/Qwen-Image-Bench",
                    "image": "gpt2.png",
                    "caption": "Qwen benchmark GPT image sample",
                    "label": "gpt-image2",
                },
                {
                    "dataset_name": "Robo531/ai-detector-benchmark-test-data",
                    "source": "Robo531/ai-detector-benchmark-test-data:train",
                    "source_url": "https://huggingface.co/datasets/Robo531/ai-detector-benchmark-test-data",
                    "image": "real.png",
                    "caption": "real image sample",
                    "label": "real",
                },
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["imported_count"] == 2
    status = client.get("/training/datasets/status?task_type=vision_generator_attribution").json()
    source_names = {item["dataset_name"] for item in status["sources"]}
    assert "Qwen/Qwen-Image-Bench" in source_names
    assert "Robo531/ai-detector-benchmark-test-data" in source_names
    samples = client.get("/training/datasets/samples?task_type=vision_generator_attribution&limit=5").json()
    qwen_sample = next(item for item in samples if item["label"] == "gpt-image2")
    assert qwen_sample["dataset_name"] == "Qwen/Qwen-Image-Bench"
    assert qwen_sample["source_url"] == "https://huggingface.co/datasets/Qwen/Qwen-Image-Bench"


def test_import_vision_dataset_requires_local_image_columns() -> None:
    _reset_training_pool()

    response = client.post(
        "/training/datasets/import",
        json={
            "dataset_name": "bad-vision-fixture",
            "source": "local fixture",
            "task_type": "vision_tamper",
            "rows": [{"caption": "拼接图", "label": "fake"}],
        },
    )

    assert response.status_code == 400
    assert "image_root" in response.json()["detail"]


def test_vision_and_fusion_training_use_external_samples_only(tmp_path: Path) -> None:
    _reset_training_pool()
    image_root = tmp_path / "mm-images"
    image_root.mkdir()
    rows = []
    for index, score in enumerate([12, 34, 76, 92]):
        name = f"mm-{index}.png"
        _write_png_fixture(image_root / name, marker=bytes([index + 1]) * (index + 3))
        rows.append(
            {
                "image": name,
                "caption": "AI 合成 警情 截图" if score > 60 else "普通现场图片",
                "label": "risk" if score > 60 else "normal",
                "risk_score": score,
                "scenario": "涉警公信力谣言" if score > 60 else "低风险误传",
            }
        )

    for task_type in ["vision_aigc", "vision_tamper", "vision_context_mismatch", "multimodal_fusion"]:
        response = client.post(
            "/training/datasets/import",
            json={
                "dataset_name": f"pytest-{task_type}",
                "source": "local fixture",
                "task_type": task_type,
                "image_root": str(image_root),
                "image_path_column": "image",
                "text_columns": ["caption"],
                "label_column": "label",
                "risk_score_column": "risk_score",
                "scenario_column": "scenario",
                "rows": rows,
            },
        )
        assert response.status_code == 200
        assert response.json()["imported_count"] == 4

    vision_run = client.post(
        "/training/vision/run",
        json={
            "task_type": "vision_aigc",
            "epochs": 60,
            "learning_rate": 0.04,
            "l2": 0.02,
            "min_samples": 4,
            "activation_mode": "activate",
        },
    )
    assert vision_run.status_code == 200
    vision_body = vision_run.json()
    assert vision_body["status"] == "active_trained"
    assert vision_body["task_type"] == "vision_aigc"
    assert "police-trust-001" in vision_body["model_card"]["excluded_demo_cases"]
    assert vision_body["validation_rmse"] >= 0
    assert "confusion_matrix" in vision_body
    assert vision_body["model_card"]["validation_protocol"]["method"] == "deterministic_stratified_holdout"
    assert vision_body["model_card"]["task_filter"]["selected_count"] == 4

    vision_status = client.get("/training/vision/status?task_type=vision_aigc").json()
    assert vision_status["trained"] is True
    assert vision_status["data"]["sample_count"] == 4

    fusion_run = client.post(
        "/training/fusion/run",
        json={
            "epochs": 60,
            "learning_rate": 0.04,
            "l2": 0.02,
            "min_samples": 4,
            "activation_mode": "activate",
        },
    )
    assert fusion_run.status_code == 200
    fusion_body = fusion_run.json()
    assert fusion_body["status"] == "trained"
    assert fusion_body["model_kind"] == "local-multimodal-fusion-ensemble-v2"
    assert "low-risk-004" in fusion_body["model_card"]["excluded_demo_cases"]
    assert fusion_body["risk_level_accuracy"] >= 0
    assert fusion_body["model_card"]["metrics"]["validation"]["confusion_matrix"]
    assert fusion_body["model_card"]["validation_protocol"]["validation_count"] == fusion_body["validation_count"]
    ensemble = fusion_body["model_card"]["ensemble_selection"]
    assert ensemble["selected_model"] in {"ridge", "knn", "ensemble"}
    assert ensemble["prototype_count"] >= 1

    fusion_status = client.get("/training/fusion/status").json()
    assert fusion_status["trained"] is True
    assert fusion_status["data"]["sample_count"] == 4

    demo = client.post("/training/evaluation/demo-cases")
    assert demo.status_code == 200
    demo_body = demo.json()
    assert demo_body["demo_case_count"] == 4
    assert "不写入训练集" in demo_body["note"]


def test_generator_attribution_training_and_real_analysis_output(tmp_path: Path, monkeypatch: Any) -> None:
    _reset_training_pool()
    empty_summary = client.get("/training/vision/competition-summary")
    assert empty_summary.status_code == 200
    empty_body = empty_summary.json()
    assert empty_body["active_model_id"] is None
    assert empty_body["training_pool"]["demo_cases_excluded"] is True
    assert "不触发重训" in empty_body["note"]

    image_root = tmp_path / "generator-images"
    image_root.mkdir()
    rows = []
    labels = ["gpt-image2", "midjourney", "sdxl", "real"]
    for index, label in enumerate(labels):
        name = f"generator-{index}.png"
        _write_png_fixture(image_root / name, marker=f"{label}-{index}".encode())
        rows.append(
            {
                "image": name,
                "caption": f"{label} source attribution sample",
                "label": label,
                "scenario": "生成模型归因",
            }
        )

    import_response = client.post(
        "/training/datasets/import",
        json={
            "dataset_name": "pytest-generator-attribution",
            "source": "local fixture",
            "task_type": "vision_generator_attribution",
            "image_root": str(image_root),
            "image_path_column": "image",
            "text_columns": ["caption"],
            "label_column": "label",
            "rows": rows,
        },
    )
    assert import_response.status_code == 200
    import_body = import_response.json()
    assert import_body["imported_count"] == 4
    assert import_body["task_type"] == "vision_generator_attribution"
    assert import_body["label_distribution"]["gpt-image2"] == 1

    train_response = client.post(
        "/training/vision/run",
        json={
            "task_type": "vision_generator_attribution",
            "epochs": 60,
            "learning_rate": 0.04,
            "l2": 0.02,
            "min_samples": 4,
            "activation_mode": "activate",
        },
    )
    assert train_response.status_code == 200
    train_body = train_response.json()
    assert train_body["model_kind"] == "local-generator-attribution-extratrees-v2"
    assert train_body["label_distribution"]["gpt-image2"] == 1
    assert "classification_metrics" in train_body["model_card"]
    assert train_body["model_card"]["primary_classifier"]["model"] == "ExtraTreesClassifier"
    assert train_body["model_card"]["binary_generated_gate"]["target"] == "generated_vs_real"
    assert "gpt-image2" in train_body["model_card"]["source_classes"]
    assert "疑似 GPT-image2" in train_body["model_card"]["boundary"]
    assert train_body["model_card"]["validation_protocol"]["method"] == "deterministic_class_stratified_holdout"

    summary_response = client.get("/training/vision/competition-summary")
    assert summary_response.status_code == 200
    summary = summary_response.json()
    assert summary["active_model_id"] == train_body["id"]
    assert summary["training_pool"]["sample_count"] == 4
    assert summary["validation_metrics"]["available"] is True
    assert summary["model_lifecycle"]["active_locked"] is True
    assert "不替代 C2PA" in " ".join(summary["limitations"])

    case_id = "pytest-generator-attribution-case"
    case_response = client.post(
        "/cases",
        json={
            "id": case_id,
            "title": "网传生成图来源待核验",
            "scenario": "涉警公信力谣言",
            "platform": "短视频平台",
            "publish_time": "2026-06-08 11:00",
            "source_url": "https://example.com/generator",
            "content": "网传图片疑似由 GPT-image2 生成并配涉警文字。",
            "image_description": "待生成模型来源归因。",
            "spread": {
                "views": 18000,
                "reposts": 600,
                "comments": 300,
                "likes": 500,
                "velocity": "快速扩散",
            },
            "manual_label": "待人工复核",
            "manual_risk_score": None,
            "tags": ["生成图", "归因"],
            "sensitivity_notes": "",
            "review_note": "",
        },
    )
    assert case_response.status_code == 200
    upload = client.post(
        f"/cases/{case_id}/assets",
        files={"file": ("generator.png", (image_root / "generator-0.png").read_bytes(), "image/png")},
    )
    assert upload.status_code == 200

    class MockGetResponse:
        content = b"<html><head><title>generator source</title></head><body>public page</body></html>"
        encoding = "utf-8"
        url = "https://example.com/generator"

        def raise_for_status(self) -> None:
            return None

    class MockGetClient:
        def __init__(self, **_: Any) -> None:
            return None

        def __enter__(self) -> "MockGetClient":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def get(self, url: str) -> MockGetResponse:
            return MockGetResponse()

    monkeypatch.setattr(socket, "getaddrinfo", lambda *_: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))])
    monkeypatch.setattr(httpx, "Client", MockGetClient)
    monkeypatch.setattr(evidence_service, "_capture_screenshot", lambda *_: False)
    snapshot = client.post(
        f"/cases/{case_id}/sources/capture",
        json={"url": "https://example.com/generator"},
    )
    assert snapshot.status_code == 200

    analysis_response = client.post(f"/cases/{case_id}/real-analysis")
    assert analysis_response.status_code == 200
    analysis_body = analysis_response.json()
    attribution = analysis_body["vision_evidence_models"]["vision_generator_attribution"]
    assert attribution["trained"] is True
    assert attribution["enabled"] is True
    assert attribution["model_kind"] == "local-generator-attribution-extratrees-v2"
    assert attribution["top_candidate"] in {"gpt-image2", "midjourney", "sdxl", "real", "unknown"}
    assert "asset_predictions" in attribution
    assert attribution["candidate_ranking"]
    assert {"rank", "label", "probability", "confidence_percent"} <= set(attribution["candidate_ranking"][0])
    assert "candidates" in attribution["asset_predictions"][0]
    assert attribution["asset_predictions"][0]["candidate_ranking"]
    assert {"rank", "label", "probability", "confidence_percent"} <= set(
        attribution["asset_predictions"][0]["candidate_ranking"][0]
    )
    assert "binary_gate" in attribution["asset_predictions"][0]
    assert "review_recommendation" in attribution["asset_predictions"][0]
    assert "C2PA" in attribution["boundary"]

    forensics_response = client.post(f"/cases/{case_id}/image-forensics")
    assert forensics_response.status_code == 200
    forensics_body = forensics_response.json()
    assert forensics_body["aggregate"]["candidate_ranking"]
    assert forensics_body["aggregate"]["ranked_candidates"]
    assert forensics_body["asset_results"][0]["candidate_ranking"]
    assert "review_recommendation" in forensics_body["asset_results"][0]
    assert {"rank", "label", "probability", "confidence_percent"} <= set(
        forensics_body["asset_results"][0]["candidate_ranking"][0]
    )

    cached_forensics = client.get(f"/cases/{case_id}/image-forensics")
    assert cached_forensics.status_code == 200
    assert cached_forensics.json()["asset_results"][0]["sha256"] == forensics_body["asset_results"][0]["sha256"]


def test_image_forensics_cache_is_cleared_after_new_upload() -> None:
    case_id = "pytest-image-cache-clear-001"
    create_response = client.post(
        "/cases",
        json={
            "id": case_id,
            "title": "图像研判缓存测试",
            "scenario": "灾害险情谣言",
            "platform": "本地测试",
            "publish_time": "2026-06-14 22:30",
            "source_url": "本地测试",
            "content": "测试上传图片后图像来源研判缓存会失效。",
            "image_description": "测试图片",
            "spread": {
                "views": 1000,
                "reposts": 20,
                "comments": 30,
                "likes": 40,
                "velocity": "低速传播",
            },
            "manual_label": "待人工复核",
            "manual_risk_score": 50,
            "tags": ["缓存测试"],
            "sensitivity_notes": "",
            "review_note": "",
        },
    )
    assert create_response.status_code == 200
    upload = client.post(
        f"/cases/{case_id}/assets",
        files={"file": ("first.png", _png_1x1(), "image/png")},
    )
    assert upload.status_code == 200
    assert client.get(f"/cases/{case_id}/image-forensics").status_code == 404
    assert client.post(f"/cases/{case_id}/image-forensics").status_code == 200
    assert client.get(f"/cases/{case_id}/image-forensics").status_code == 200
    second_upload = client.post(
        f"/cases/{case_id}/assets",
        files={"file": ("second.png", _png_1x1(), "image/png")},
    )
    assert second_upload.status_code == 200
    assert client.get(f"/cases/{case_id}/image-forensics").status_code == 404


def test_image_forensics_prefers_trained_model_over_demo_prior(monkeypatch: Any) -> None:
    case_id = "pytest-gpt-image-demo-prior-001"
    create_response = client.post(
        "/cases",
        json={
            "id": case_id,
            "title": "GPT-image 车站图片真实模型覆盖测试",
            "scenario": "涉警公信力谣言",
            "platform": "本地测试",
            "publish_time": "2026-06-15 09:30",
            "source_url": "本地测试",
            "content": "文件名和文字故意命中 GPT-image 演示样本关键词，但本地必须采用真实模型输出。",
            "image_description": "测试图片",
            "spread": {
                "views": 1000,
                "reposts": 20,
                "comments": 30,
                "likes": 40,
                "velocity": "低速传播",
            },
            "manual_label": "待人工复核",
            "manual_risk_score": 50,
            "tags": ["gpt-image", "真实模型"],
            "sensitivity_notes": "",
            "review_note": "",
        },
    )
    assert create_response.status_code == 200
    upload = client.post(
        f"/cases/{case_id}/assets",
        files={"file": ("gptimage-station-police-conflict.png", _png_1x1(), "image/png")},
    )
    assert upload.status_code == 200
    asset_id = upload.json()["id"]

    def trained_prediction(_: list[Any], case_text: str = "") -> dict[str, object]:
        assert "GPT-image" in case_text
        return {
            "vision_generator_attribution": {
                "trained": True,
                "enabled": True,
                "model_id": "pytest-real-model",
                "model_kind": "local-generator-attribution-extratrees-v2",
                "asset_predictions": [
                    {
                        "asset_id": asset_id,
                        "top_candidate": "real",
                        "confidence": 0.91,
                        "candidate_ranking": [
                            {"rank": 1, "label": "real", "probability": 0.91, "confidence": 0.91},
                            {"rank": 2, "label": "other-generated", "probability": 0.06, "confidence": 0.06},
                            {"rank": 3, "label": "gpt-image2", "probability": 0.03, "confidence": 0.03},
                        ],
                    }
                ],
            }
        }

    monkeypatch.delenv("SMARTPOLICE_ENABLE_DEMO_FORENSICS_FALLBACK", raising=False)
    monkeypatch.setattr("app.image_forensics.predict_vision_for_assets", trained_prediction)

    response = client.post(f"/cases/{case_id}/image-forensics")

    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == "pytest-real-model"
    assert body["asset_results"][0]["top_candidate"] == "real"
    assert body["asset_results"][0]["confidence"] == 0.91
    assert body["asset_results"][0]["candidate_ranking"][0]["label"] == "real"
    assert body["asset_results"][0]["review_recommendation"] == {}


def test_image_forensics_keeps_known_public_real_photo_as_real() -> None:
    case_id = "pytest-known-public-real-photo-001"
    create_response = client.post(
        "/cases",
        json={
            "id": case_id,
            "title": "公开来源真实灾情救援照片核验",
            "scenario": "灾害险情核查",
            "platform": "Wikimedia Commons / 演示导入",
            "publish_time": "2008-05-14",
            "source_url": "https://commons.wikimedia.org/wiki/File:Sichuan_earthquake_save..JPG",
            "content": "公开来源真实照片，用于和 AI 生成灾情图片形成对照。",
            "image_description": "真实灾害救援现场。",
            "spread": {
                "views": 64000,
                "reposts": 1400,
                "comments": 620,
                "likes": 3100,
                "velocity": "公开来源核验",
            },
            "manual_label": "公开来源真实灾情救援照片",
            "manual_risk_score": 32,
            "tags": ["真实照片", "Wikimedia Commons", "汶川地震", "救援现场"],
            "sensitivity_notes": "",
            "review_note": "",
        },
    )
    assert create_response.status_code == 200
    upload = client.post(
        f"/cases/{case_id}/assets",
        files={"file": ("real-sichuan-earthquake-rescue.jpg", _png_1x1(), "image/png")},
    )
    assert upload.status_code == 200

    response = client.post(f"/cases/{case_id}/image-forensics")

    assert response.status_code == 200
    body = response.json()
    asset_result = body["asset_results"][0]
    assert asset_result["top_candidate"] == "real"
    assert asset_result["candidate_ranking"][0]["label"] == "real"
    assert asset_result["candidate_ranking"][0]["probability"] >= 0.45
    assert asset_result["candidate_ranking"][1]["label"] == "gpt-image2"
    assert body["aggregate"]["top_candidate"] == "real"


def test_generator_attribution_robustness_run_uses_real_image_perturbations(tmp_path: Path) -> None:
    _reset_training_pool()
    image_root = tmp_path / "robust-generator-images"
    image_root.mkdir()
    rows = []
    labels = ["gpt-image2", "gpt-image2", "midjourney", "sdxl", "flux", "real", "real", "dall-e-3"]
    for index, label in enumerate(labels):
        name = f"robust-{index}.png"
        _write_real_png_fixture(image_root / name, index)
        rows.append(
            {
                "image": name,
                "caption": "文字密集截图 水印 转发 压缩" if index % 2 == 0 else "ordinary photo without generator token",
                "label": label,
                "scenario": "传播扰动鲁棒性评估",
            }
        )

    import_response = client.post(
        "/training/datasets/import",
        json={
            "dataset_name": "pytest-generator-robustness",
            "source": "local real-image fixture",
            "task_type": "vision_generator_attribution",
            "image_root": str(image_root),
            "image_path_column": "image",
            "text_columns": ["caption"],
            "label_column": "label",
            "rows": rows,
        },
    )
    assert import_response.status_code == 200
    assert import_response.json()["imported_count"] == len(labels)

    train_response = client.post(
        "/training/vision/run",
        json={
            "task_type": "vision_generator_attribution",
            "epochs": 80,
            "learning_rate": 0.04,
            "l2": 0.02,
            "min_samples": 4,
            "activation_mode": "activate",
        },
    )
    assert train_response.status_code == 200
    train_body = train_response.json()
    assert train_body["feature_count"] > 0
    assert "频域" in train_body["model_card"]["architecture"]
    assert any("文本富集" in item for item in train_body["model_card"]["leakage_controls"])

    response = client.post(
        "/training/vision/robustness-run",
        json={
            "task_type": "vision_generator_attribution",
            "limit": 8,
            "conditions": ["clean", "jpeg_q85", "jpeg_q60", "screenshot_resave", "center_crop", "watermark"],
            "include_sample_predictions": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task_type"] == "vision_generator_attribution"
    assert body["sample_count"] == 8
    assert body["label_distribution"]["gpt-image2"] == 2
    condition_names = {item["condition"] for item in body["conditions"]}
    assert condition_names == {"clean", "jpeg_q85", "jpeg_q60", "screenshot_resave", "center_crop", "watermark"}
    assert all(item["sample_count"] == 8 for item in body["conditions"])
    assert all(0 <= item["accuracy"] <= 1 for item in body["conditions"])
    assert body["conditions"][0]["confidence_delta_from_clean"] == 0
    assert body["conditions"][1]["confidence_delta_from_clean"] is not None
    assert body["conditions"][0]["sample_predictions"]
    assert "compression_traces" in body["feature_groups"]
    assert "frequency_and_texture" in body["feature_groups"]
    assert "propagation_disturbance" in body["feature_groups"]
    assert body["model_card_update"]["does_not_retrain"] is True
    assert "GPT-image-2" in body["model_card_update"]["target"]
    assert body["conclusions"]


def test_generator_attribution_training_augmentation_keeps_clean_validation_and_dataset_count(
    tmp_path: Path,
) -> None:
    _reset_training_pool()
    image_root = tmp_path / "augmented-generator-images"
    image_root.mkdir()
    rows = []
    labels = ["gpt-image2", "gpt-image2", "midjourney", "sdxl", "flux", "real", "real", "dall-e-3"]
    for index, label in enumerate(labels):
        name = f"augmented-{index}.png"
        _write_real_png_fixture(image_root / name, index + 20)
        rows.append(
            {
                "image": name,
                "caption": "截图 转发 水印 压缩 文字覆盖" if index % 2 == 0 else "ordinary scene after reposting",
                "label": label,
                "scenario": "GPT-image-2 传播扰动训练增强",
            }
        )

    import_response = client.post(
        "/training/datasets/import",
        json={
            "dataset_name": "pytest-generator-augmentation",
            "source": "local real-image fixture",
            "task_type": "vision_generator_attribution",
            "image_root": str(image_root),
            "image_path_column": "image",
            "text_columns": ["caption"],
            "label_column": "label",
            "rows": rows,
        },
    )
    assert import_response.status_code == 200
    assert import_response.json()["imported_count"] == len(labels)
    status_before = client.get("/training/datasets/status?task_type=vision_generator_attribution").json()
    assert status_before["external_sample_count"] == len(labels)

    train_response = client.post(
        "/training/vision/run",
        json={
            "task_type": "vision_generator_attribution",
            "epochs": 80,
            "learning_rate": 0.04,
            "l2": 0.02,
            "min_samples": 4,
            "activation_mode": "activate",
            "enable_perturbation_augmentation": True,
            "augmentation_conditions": ["jpeg_q85", "jpeg_q60", "watermark"],
            "max_augmented_samples": 12,
        },
    )

    assert train_response.status_code == 200
    body = train_response.json()
    assert body["sample_count"] == len(labels)
    protocol = body["model_card"]["augmentation_protocol"]
    assert protocol["enabled"] is True
    assert protocol["original_train_count"] == body["model_card"]["validation_protocol"]["train_count"]
    assert protocol["generated_augmentation_count"] > 0
    assert protocol["augmented_train_count"] > protocol["original_train_count"]
    assert protocol["augmented_train_count"] <= protocol["original_train_count"] + 12
    assert protocol["not_written_to_dataset"] is True
    assert protocol["validation_policy"] == "clean holdout only"
    assert set(protocol["condition_counts"]) <= {"jpeg_q85", "jpeg_q60", "watermark"}
    assert protocol["cache_misses"] == protocol["generated_augmentation_count"]
    assert protocol["cache_hits"] == 0
    assert "feature_cache_policy" in protocol
    assert any("扰动增强" in item for item in body["training_trace"])
    assert body["model_card"]["classification_metrics"]["validation"]["accuracy"] >= 0

    cached_train_response = client.post(
        "/training/vision/run",
        json={
            "task_type": "vision_generator_attribution",
            "epochs": 80,
            "learning_rate": 0.04,
            "l2": 0.02,
            "min_samples": 4,
            "activation_mode": "activate",
            "enable_perturbation_augmentation": True,
            "augmentation_conditions": ["jpeg_q85", "jpeg_q60", "watermark"],
            "max_augmented_samples": 12,
        },
    )
    assert cached_train_response.status_code == 200
    cached_protocol = cached_train_response.json()["model_card"]["augmentation_protocol"]
    assert cached_protocol["generated_augmentation_count"] == protocol["generated_augmentation_count"]
    assert cached_protocol["cache_hits"] == cached_protocol["generated_augmentation_count"]
    assert cached_protocol["cache_misses"] == 0
    assert cached_protocol["condition_counts"] == protocol["condition_counts"]

    status_after = client.get("/training/datasets/status?task_type=vision_generator_attribution").json()
    assert status_after["external_sample_count"] == len(labels)
    assert status_after["tasks"][0]["sample_count"] == len(labels)

    robustness_response = client.post(
        "/training/vision/robustness-run",
        json={
            "task_type": "vision_generator_attribution",
            "limit": 8,
            "conditions": ["clean", "jpeg_q85", "watermark"],
            "include_sample_predictions": False,
        },
    )
    assert robustness_response.status_code == 200
    robustness_body = robustness_response.json()
    assert {item["condition"] for item in robustness_body["conditions"]} == {"clean", "jpeg_q85", "watermark"}


def test_generator_attribution_training_uses_source_holdout_strategy(tmp_path: Path) -> None:
    _reset_training_pool()
    _import_cross_source_generator_fixture(tmp_path)

    train_response = client.post(
        "/training/vision/run",
        json={
            "task_type": "vision_generator_attribution",
            "epochs": 80,
            "learning_rate": 0.04,
            "l2": 0.02,
            "min_samples": 4,
            "activation_mode": "candidate",
            "validation_strategy": "source_holdout",
            "source_holdout_fraction": 0.25,
            "min_source_holdout_samples": 2,
        },
    )

    assert train_response.status_code == 200
    body = train_response.json()
    protocol = body["model_card"]["validation_protocol"]
    assert protocol["method"] in {"source_holdout", "source_stratified_holdout"}
    assert protocol["validation_count"] >= 2
    assert protocol["source_overlap_count"] >= 0
    assert any("验证策略" in item for item in body["training_trace"])
    assert body["status"] == "candidate_trained"


def test_generator_experiment_profile_cannot_activate_directly(tmp_path: Path) -> None:
    _reset_training_pool()
    _import_cross_source_generator_fixture(tmp_path)

    response = client.post(
        "/training/vision/run",
        json={
            "task_type": "vision_generator_attribution",
            "epochs": 80,
            "learning_rate": 0.04,
            "l2": 0.02,
            "min_samples": 4,
            "activation_mode": "activate",
            "experiment_profile": "gpt_image2_ovr",
        },
    )

    assert response.status_code == 400
    assert "只能保存为 candidate" in response.json()["detail"]


def test_generator_attribution_training_sample_limit_bounds_profile_run(tmp_path: Path) -> None:
    _reset_training_pool()
    _import_cross_source_generator_fixture(tmp_path)

    response = client.post(
        "/training/vision/run",
        json={
            "task_type": "vision_generator_attribution",
            "epochs": 80,
            "learning_rate": 0.04,
            "l2": 0.02,
            "min_samples": 4,
            "max_training_samples": 6,
            "activation_mode": "candidate",
            "experiment_profile": "binary_generated_gate",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["sample_count"] == 6
    assert body["status"] == "candidate_trained"
    assert body["model_card"]["experiment_profile"]["profile"] == "binary_generated_gate"
    assert body["model_card"]["profile_policy"]["chinese_name"] == "真实/生成鲁棒初筛"
    assert body["model_card"]["profile_policy"]["candidate_only"] is True
    assert any(
        gate["metric"] == "source_generated_recall"
        for gate in body["model_card"]["profile_policy"]["acceptance_gates"]
    )


def test_generator_attribution_anti_cheat_audit_reports_leakage_risks(tmp_path: Path) -> None:
    _reset_training_pool()
    image_root = tmp_path / "anti-cheat-images"
    image_root.mkdir()
    rows = []
    labels = ["gpt-image2", "gpt-image2", "midjourney", "sdxl", "flux", "real", "real", "dall-e-3"]
    for index, label in enumerate(labels):
        name = f"audit-{index}.png"
        _write_real_png_fixture(image_root / name, index + 40)
        rows.append(
            {
                "dataset_name": "pytest-source-a" if index < 4 else "pytest-source-b",
                "source": "pytest-source-a:train" if index < 4 else "pytest-source-b:train",
                "image": name,
                "caption": f"{label} caption with repost watermark context",
                "label": label,
                "scenario": "anti cheat audit fixture",
            }
        )

    import_response = client.post(
        "/training/datasets/import",
        json={
            "dataset_name": "pytest-generator-audit",
            "source": "local audit fixture",
            "task_type": "vision_generator_attribution",
            "image_root": str(image_root),
            "image_path_column": "image",
            "text_columns": ["caption"],
            "label_column": "label",
            "rows": rows,
        },
    )
    assert import_response.status_code == 200
    assert import_response.json()["imported_count"] == len(labels)

    train_response = client.post(
        "/training/vision/run",
        json={
            "task_type": "vision_generator_attribution",
            "epochs": 80,
            "learning_rate": 0.04,
            "l2": 0.02,
            "min_samples": 4,
            "activation_mode": "activate",
            "validation_strategy": "source_holdout",
            "min_source_holdout_samples": 2,
        },
    )
    assert train_response.status_code == 200

    audit_response = client.post(
        "/training/vision/anti-cheat-audit",
        json={
            "task_type": "vision_generator_attribution",
            "holdout_key": "dataset",
            "max_holdout_groups": 2,
            "min_holdout_samples": 2,
            "source_holdout_sample_limit": 6,
            "feature_ablation_limit": 8,
            "include_source_holdout": True,
            "include_feature_ablation": True,
        },
    )

    assert audit_response.status_code == 200
    body = audit_response.json()
    assert body["active_model_id"] == train_response.json()["id"]
    assert body["training_validation"]
    assert body["source_holdout"]["protocol"]["method"] == "leave_one_source_group_out"
    assert body["source_holdout"]["protocol"]["sample_limit"] == 6
    assert body["source_holdout"]["protocol"]["sampled_from_count"] == len(labels)
    assert body["feature_ablation"]["results"]
    ablation_sets = {item["feature_set"] for item in body["feature_ablation"]["results"]}
    assert "visual_forensics_only" in ablation_sets
    assert "no_text_context_proxy" in ablation_sets
    assert body["leakage_checks"]["validation_count"] >= 1
    assert body["leakage_checks"]["note"]
    assert body["suspicious_feature_names"]
    assert "候选门控" in " ".join(body["recommended_claims"])
    assert "不可宣传" in body["verdict"] or "未发现" in body["verdict"]


def test_generator_attribution_candidate_lifecycle_requires_explicit_activation(tmp_path: Path) -> None:
    _reset_training_pool()
    image_root = tmp_path / "candidate-generator-images"
    image_root.mkdir()
    rows = []
    labels = ["gpt-image2", "gpt-image2", "midjourney", "sdxl", "flux", "real", "real", "dall-e-3"]
    for index, label in enumerate(labels):
        name = f"candidate-{index}.png"
        _write_real_png_fixture(image_root / name, index + 100)
        rows.append(
            {
                "image": name,
                "caption": "social platform repost compression watermark" if index % 2 == 0 else "camera photo",
                "label": label,
                "scenario": "候选模型生命周期",
            }
        )

    import_response = client.post(
        "/training/datasets/import",
        json={
            "dataset_name": "pytest-generator-candidate-lifecycle",
            "source": "local real-image fixture",
            "task_type": "vision_generator_attribution",
            "image_root": str(image_root),
            "image_path_column": "image",
            "text_columns": ["caption"],
            "label_column": "label",
            "rows": rows,
        },
    )
    assert import_response.status_code == 200

    active_response = client.post(
        "/training/vision/run",
        json={
            "task_type": "vision_generator_attribution",
            "epochs": 80,
            "learning_rate": 0.04,
            "l2": 0.02,
            "min_samples": 4,
            "activation_mode": "activate",
        },
    )
    assert active_response.status_code == 200
    active_run = active_response.json()
    assert active_run["status"] == "active_trained"

    candidate_response = client.post(
        "/training/vision/run",
        json={
            "task_type": "vision_generator_attribution",
            "epochs": 80,
            "learning_rate": 0.04,
            "l2": 0.02,
            "min_samples": 4,
        },
    )
    assert candidate_response.status_code == 200
    candidate_run = candidate_response.json()
    assert candidate_run["status"] == "candidate_trained"
    assert candidate_run["model_card"]["lifecycle"]["activation_mode"] == "candidate"

    status_after_candidate = client.get(
        "/training/vision/status?task_type=vision_generator_attribution",
    ).json()
    assert status_after_candidate["active_model_id"] == active_run["id"]
    assert status_after_candidate["latest_candidate"]["id"] == candidate_run["id"]
    assert status_after_candidate["candidate_vs_active"]["candidate_model_id"] == candidate_run["id"]

    runs_response = client.get("/training/vision/runs?task_type=vision_generator_attribution&limit=5")
    assert runs_response.status_code == 200
    run_records = runs_response.json()
    assert run_records[0]["run"]["id"] == candidate_run["id"]
    assert run_records[0]["is_active"] is False
    assert any(item["run"]["id"] == active_run["id"] and item["is_active"] for item in run_records)

    evaluation_response = client.post(
        "/training/vision/evaluate-candidate",
        json={
            "task_type": "vision_generator_attribution",
            "candidate_model_id": candidate_run["id"],
            "limit": 8,
            "conditions": ["clean", "jpeg_q85", "watermark"],
            "include_source_holdout": False,
            "include_feature_ablation": False,
        },
    )
    assert evaluation_response.status_code == 200
    evaluation = evaluation_response.json()
    assert evaluation["candidate_model_id"] == candidate_run["id"]
    assert evaluation["active_model_id_before"] == active_run["id"]
    assert evaluation["activated"] is False
    assert "passed" in evaluation["gate"]
    assert {item["condition"] for item in evaluation["conditions"]} == {"clean", "jpeg_q85", "watermark"}

    activation_response = client.post(
        "/training/vision/activate",
        json={
            "task_type": "vision_generator_attribution",
            "run_id": candidate_run["id"],
        },
    )
    assert activation_response.status_code == 200
    activation = activation_response.json()
    assert activation["active_model_id"] == candidate_run["id"]
    assert activation["previous_active_model_id"] == active_run["id"]

    status_after_activation = client.get(
        "/training/vision/status?task_type=vision_generator_attribution",
    ).json()
    assert status_after_activation["active_model_id"] == candidate_run["id"]
    assert status_after_activation["latest_run"]["status"] == "active_trained"


def test_generator_attribution_augmentation_cache_warmup_does_not_train_or_pollute_dataset(
    tmp_path: Path,
) -> None:
    _reset_training_pool()
    image_root = tmp_path / "warmup-generator-images"
    image_root.mkdir()
    rows = []
    labels = ["gpt-image2", "gpt-image2", "midjourney", "sdxl", "flux", "real", "real", "dall-e-3"]
    for index, label in enumerate(labels):
        name = f"warmup-{index}.png"
        _write_real_png_fixture(image_root / name, index + 80)
        rows.append(
            {
                "image": name,
                "caption": "platform repost screenshot watermark" if index % 2 == 0 else "ordinary photo",
                "label": label,
                "scenario": "增强缓存预热",
            }
        )

    import_response = client.post(
        "/training/datasets/import",
        json={
            "dataset_name": "pytest-generator-warmup",
            "source": "local real-image fixture",
            "task_type": "vision_generator_attribution",
            "image_root": str(image_root),
            "image_path_column": "image",
            "text_columns": ["caption"],
            "label_column": "label",
            "rows": rows,
        },
    )
    assert import_response.status_code == 200
    train_response = client.post(
        "/training/vision/run",
        json={
            "task_type": "vision_generator_attribution",
            "epochs": 80,
            "learning_rate": 0.04,
            "l2": 0.02,
            "min_samples": 4,
            "activation_mode": "activate",
        },
    )
    assert train_response.status_code == 200
    active_before = client.get("/training/vision/status?task_type=vision_generator_attribution").json()[
        "active_model_id"
    ]

    payload = {
        "task_type": "vision_generator_attribution",
        "limit": 6,
        "conditions": ["jpeg_q85", "watermark"],
    }
    first_response = client.post("/training/vision/augmentation-cache-warmup", json=payload)
    assert first_response.status_code == 200
    first = first_response.json()
    assert first["does_not_train"] is True
    assert first["does_not_change_active_model"] is True
    assert first["sample_count"] == 6
    assert first["condition_counts"] == {"jpeg_q85": 6, "watermark": 6}
    assert first["cache_hits"] == 0
    assert first["cache_misses"] == 12
    assert "feature_cache" in first["feature_cache_policy"]

    second_response = client.post("/training/vision/augmentation-cache-warmup", json=payload)
    assert second_response.status_code == 200
    second = second_response.json()
    assert second["cache_hits"] == 12
    assert second["cache_misses"] == 0
    assert second["condition_counts"] == first["condition_counts"]

    active_after = client.get("/training/vision/status?task_type=vision_generator_attribution").json()[
        "active_model_id"
    ]
    assert active_after == active_before
    status_after = client.get("/training/datasets/status?task_type=vision_generator_attribution").json()
    assert status_after["external_sample_count"] == len(labels)


def test_generator_attribution_source_holdout_reports_cross_source_metrics(tmp_path: Path) -> None:
    _reset_training_pool()
    image_root = tmp_path / "source-holdout-images"
    image_root.mkdir()
    rows = []
    labels = [
        "gpt-image2",
        "real",
        "midjourney",
        "gpt-image2",
        "real",
        "sdxl",
        "gpt-image2",
        "real",
        "flux",
    ]
    for index, label in enumerate(labels):
        name = f"holdout-{index}.png"
        _write_real_png_fixture(image_root / name, index + 40)
        source_id = index // 3
        rows.append(
            {
                "dataset_name": f"pytest-source-{source_id}",
                "source": f"fixture-split-{source_id}",
                "source_url": f"https://huggingface.test/datasets/source-{source_id}",
                "image": name,
                "caption": "platform compressed screenshot with watermark" if index % 2 else "clean social image sample",
                "label": label,
                "scenario": "跨来源留出评估",
            }
        )

    import_response = client.post(
        "/training/datasets/import",
        json={
            "dataset_name": "pytest-source-holdout",
            "source": "fallback source",
            "task_type": "vision_generator_attribution",
            "image_root": str(image_root),
            "image_path_column": "image",
            "text_columns": ["caption"],
            "label_column": "label",
            "rows": rows,
        },
    )
    assert import_response.status_code == 200
    assert import_response.json()["imported_count"] == len(labels)

    response = client.post(
        "/training/vision/source-holdout-run",
        json={
            "task_type": "vision_generator_attribution",
            "holdout_key": "dataset_source",
            "min_train_samples": 4,
            "min_holdout_samples": 2,
            "max_holdout_groups": 5,
            "sample_limit": 6,
            "enable_perturbation_augmentation": True,
            "augmentation_conditions": ["jpeg_q85", "jpeg_q60"],
            "max_augmented_samples": 8,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task_type"] == "vision_generator_attribution"
    assert body["sample_count"] == 6
    assert body["source_count"] == 3
    assert body["protocol"]["method"] == "leave_one_source_group_out"
    assert body["protocol"]["sample_limit"] == 6
    assert body["protocol"]["sampled_from_count"] == len(labels)
    assert body["protocol"]["saves_model"] is False
    assert body["protocol"]["uses_demo_cases"] is False
    assert body["protocol"]["augmentation"]["enabled"] is True
    assert body["protocol"]["label_covered_diagnostic"]["method"] == "source_stratified_label_covered_holdout"
    assert len(body["groups"]) == 3
    assert any(not item["skipped"] for item in body["groups"])
    completed = [item for item in body["groups"] if not item["skipped"]]
    assert all(0 <= item["accuracy"] <= 1 for item in completed)
    assert body["aggregate"]["completed_group_count"] >= 1
    assert "seen_class_holdout_count" in body["aggregate"]
    assert "mean_seen_class_macro_f1" in body["aggregate"]
    assert "mean_binary_macro_f1" in body["aggregate"]
    assert "mean_real_false_positive_rate" in body["aggregate"]
    assert "overall_real_false_positive_rate" in body["aggregate"]
    assert "real_false_positive_count" in body["aggregate"]
    assert "real_support" in body["aggregate"]
    assert "label_covered_macro_f1" in body["aggregate"]
    assert "label_covered_binary_macro_f1" in body["aggregate"]
    assert all("seen_class_macro_f1" in item for item in body["groups"])
    assert all("unseen_holdout_labels" in item for item in body["groups"])
    assert all("binary_macro_f1" in item for item in body["groups"])
    assert all("real_false_positive_rate" in item for item in body["groups"])
    assert all("real_false_positive_count" in item for item in body["groups"])
    assert all("real_support" in item for item in body["groups"])
    assert body["conclusions"]
    assert "C2PA" in body["limitations"][2]


def test_generator_attribution_feature_ablation_reports_feature_group_deltas(tmp_path: Path) -> None:
    _reset_training_pool()
    image_root = tmp_path / "feature-ablation-images"
    image_root.mkdir()
    rows = []
    labels = ["gpt-image2", "gpt-image2", "midjourney", "sdxl", "flux", "real", "real", "dall-e-3"]
    for index, label in enumerate(labels):
        name = f"ablation-{index}.png"
        _write_real_png_fixture(image_root / name, index + 60)
        rows.append(
            {
                "image": name,
                "caption": "text rich screenshot watermark compressed" if index % 2 == 0 else "camera style outdoor photo",
                "label": label,
                "scenario": "特征组消融实验",
            }
        )

    import_response = client.post(
        "/training/datasets/import",
        json={
            "dataset_name": "pytest-feature-ablation",
            "source": "local real-image fixture",
            "task_type": "vision_generator_attribution",
            "image_root": str(image_root),
            "image_path_column": "image",
            "text_columns": ["caption"],
            "label_column": "label",
            "rows": rows,
        },
    )
    assert import_response.status_code == 200

    response = client.post(
        "/training/vision/feature-ablation-run",
        json={
            "task_type": "vision_generator_attribution",
            "limit": 20,
            "min_samples": 4,
            "feature_sets": [
                "all",
                "visual_forensics_only",
                "text_context_proxy_only",
                "no_text_context_proxy",
                "frequency_texture_only",
                "compression_traces_only",
                "no_frequency_texture",
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task_type"] == "vision_generator_attribution"
    assert body["sample_count"] == len(labels)
    assert body["validation_count"] >= 1
    assert "frequency_and_texture" in body["feature_groups"]
    assert "text_context_proxy" in body["feature_groups"]
    result_names = {item["feature_set"] for item in body["results"]}
    assert result_names == {
        "all",
        "visual_forensics_only",
        "text_context_proxy_only",
        "no_text_context_proxy",
        "frequency_texture_only",
        "compression_traces_only",
        "no_frequency_texture",
    }
    assert body["deltas_from_all"]["no_text_context_proxy"]["macro_f1_delta"] <= 1
    assert body["deltas_from_all"]["no_frequency_texture"]["accuracy_delta"] <= 1
    assert body["conclusions"]
    assert "C2PA" in body["limitations"][3]


def test_real_analysis_exposes_untrained_multimodal_model_fields(monkeypatch: Any) -> None:
    _reset_training_pool()
    case_id = "pytest-real-untrained-fields"
    create_response = client.post(
        "/cases",
        json={
            "id": case_id,
            "title": "网传截图待核验",
            "scenario": "涉警公信力谣言",
            "platform": "短视频平台",
            "publish_time": "2026-06-08 10:00",
            "source_url": "https://example.com/untrained",
            "content": "网传警方隐瞒警情并要求立即转发。",
            "image_description": "待视觉核验。",
            "spread": {
                "views": 20000,
                "reposts": 800,
                "comments": 500,
                "likes": 900,
                "velocity": "快速扩散",
            },
            "manual_label": "待人工复核",
            "manual_risk_score": None,
            "tags": ["涉警", "截图"],
            "sensitivity_notes": "",
            "review_note": "",
        },
    )
    assert create_response.status_code == 200
    upload = client.post(
        f"/cases/{case_id}/assets",
        files={"file": ("screen.png", _png_1x1(), "image/png")},
    )
    assert upload.status_code == 200

    class MockGetResponse:
        content = b"<html><head><title>source</title></head><body>official notice</body></html>"
        encoding = "utf-8"
        url = "https://example.com/untrained"

        def raise_for_status(self) -> None:
            return None

    class MockGetClient:
        def __init__(self, **_: Any) -> None:
            return None

        def __enter__(self) -> "MockGetClient":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def get(self, url: str) -> MockGetResponse:
            return MockGetResponse()

    monkeypatch.setattr(socket, "getaddrinfo", lambda *_: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))])
    monkeypatch.setattr(httpx, "Client", MockGetClient)
    monkeypatch.setattr(evidence_service, "_capture_screenshot", lambda *_: False)
    snapshot = client.post(
        f"/cases/{case_id}/sources/capture",
        json={"url": "https://example.com/untrained"},
    )
    assert snapshot.status_code == 200

    response = client.post(f"/cases/{case_id}/real-analysis")

    assert response.status_code == 200
    body = response.json()
    assert body["text_risk_model"]["trained"] is False
    assert body["fusion_model"]["trained"] is False
    assert body["vision_evidence_models"]["vision_aigc"]["trained"] is False
    assert body["baseline_risk"]["model_score"] is None
