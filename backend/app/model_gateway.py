from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import httpx
from dotenv import load_dotenv

from app.local_reasoning import (
    LOCAL_REPORT_MODEL,
    LOCAL_REPORT_PROVIDER,
    LOCAL_REPORT_ROLE,
    LOCAL_REVIEW_MODEL,
    LOCAL_REVIEW_PROVIDER,
    LOCAL_REVIEW_ROLE,
    local_gateway_response,
    record_local_invocation,
)
from app.models import (
    ModelInvocationAudit,
    ModelGatewayStatus,
    ModelInvocationRequest,
    ModelInvocationResult,
    ModelProviderStatus,
)
from app.storage import record_llm_invocation


ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    api_key_env: str
    base_url_env: str
    default_base_url: str
    model_env_by_role: dict[str, str]
    default_models: dict[str, str]
    roles: list[str]
    path: str
    requires_api_key: bool = True
    local_offline: bool = False


LOCAL_PROVIDERS = [
    ProviderConfig(
        provider=LOCAL_REVIEW_PROVIDER,
        api_key_env="",
        base_url_env="",
        default_base_url="local://offline-review",
        model_env_by_role={},
        default_models={LOCAL_REVIEW_ROLE: LOCAL_REVIEW_MODEL},
        roles=[LOCAL_REVIEW_ROLE],
        path="",
        requires_api_key=False,
        local_offline=True,
    ),
    ProviderConfig(
        provider=LOCAL_REPORT_PROVIDER,
        api_key_env="",
        base_url_env="",
        default_base_url="local://offline-report",
        model_env_by_role={},
        default_models={LOCAL_REPORT_ROLE: LOCAL_REPORT_MODEL},
        roles=[LOCAL_REPORT_ROLE],
        path="",
        requires_api_key=False,
        local_offline=True,
    ),
]


BASE_PROVIDERS = [
    ProviderConfig(
        provider="LocalVision",
        api_key_env="LOCAL_VISION_API_KEY",
        base_url_env="LOCAL_VISION_BASE_URL",
        default_base_url="http://127.0.0.1:1235/v1",
        model_env_by_role={
            "视觉证据分析": "LOCAL_VISION_MODEL",
        },
        default_models={
            "视觉证据分析": "Qwen3-VL-8B-Instruct",
        },
        roles=["视觉证据分析"],
        path="/chat/completions",
        requires_api_key=False,
    ),
    ProviderConfig(
        provider="DeepSeek",
        api_key_env="DEEPSEEK_API_KEY",
        base_url_env="DEEPSEEK_BASE_URL",
        default_base_url="https://api.deepseek.com",
        model_env_by_role={
            "复杂推理主控": "DEEPSEEK_MODEL_REASONING",
            "复核器": "DEEPSEEK_MODEL_VERIFIER",
        },
        default_models={
            "复杂推理主控": "deepseek-v4-pro",
            "复核器": "deepseek-v4-pro",
        },
        roles=["复杂推理主控", "复核器"],
        path="/chat/completions",
    ),
    ProviderConfig(
        provider="MiniMax",
        api_key_env="MINIMAX_API_KEY",
        base_url_env="MINIMAX_BASE_URL",
        default_base_url="https://api.minimax.io/v1",
        model_env_by_role={
            "任务路由器": "MINIMAX_MODEL_ROUTER",
            "长上下文证据读取": "MINIMAX_MODEL_LONG_CONTEXT",
            "中文业务生成": "MINIMAX_MODEL_REPORT",
        },
        default_models={
            "任务路由器": "MiniMax-M3",
            "长上下文证据读取": "MiniMax-M3",
            "中文业务生成": "MiniMax-M3",
        },
        roles=["任务路由器", "长上下文证据读取", "中文业务生成"],
        path="/chat/completions",
    ),
]


OPTIONAL_DASHSCOPE_PROVIDERS = [
    ProviderConfig(
        provider="DashScope",
        api_key_env="DASHSCOPE_API_KEY",
        base_url_env="DASHSCOPE_BASE_URL",
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_env_by_role={
            "视觉证据分析": "DASHSCOPE_VISION_MODEL",
        },
        default_models={
            "视觉证据分析": "qwen-vl-plus",
        },
        roles=["视觉证据分析"],
        path="/chat/completions",
    ),
]


class ModelGatewayError(Exception):
    status_code = 502

    def __init__(self, message: str, audit_id: str | None = None) -> None:
        super().__init__(message)
        self.audit_id = audit_id


class ProviderNotConfiguredError(ModelGatewayError):
    status_code = 503


class ProviderTimeoutError(ModelGatewayError):
    status_code = 504


class ProviderResponseError(ModelGatewayError):
    status_code = 502


def get_model_gateway_status() -> ModelGatewayStatus:
    _load_env_file()
    return ModelGatewayStatus(
        providers=[_provider_status(config) for config in _providers()],
        dry_run_default=False,
        note="主链路本地优先：本地训练模型、CLIP/图片特征、LocalReview 与 LocalReport 可离线完成复核和报告。DeepSeek、MiniMax、DashScope 仅作为显式启用的云端增强。",
    )


def invoke_model(request: ModelInvocationRequest) -> ModelInvocationResult:
    _load_env_file()
    config = _provider_config(request.provider)
    _validate_role(config, request.role)
    selected_model = _model_for_role(config, request.role)
    base_url = _base_url_for(config)
    api_key = _env(config.api_key_env)
    configured = _configured(config, api_key, base_url)
    payload = _chat_payload(request, selected_model)

    if request.dry_run:
        return ModelInvocationResult(
            provider=config.provider,
            role=request.role,
            selected_model=selected_model,
            configured=configured,
            dry_run=True,
            request_payload=payload,
            error=None if configured else "Provider API key or base URL is not configured.",
        )
    if config.local_offline:
        return _invoke_local_provider(
            request=request,
            config=config,
            selected_model=selected_model,
            payload=payload,
        )

    return _send_payload(
        config=config,
        case_id=request.case_id,
        role=request.role,
        selected_model=selected_model,
        payload=payload,
        api_key=api_key,
        base_url=base_url,
        configured=configured,
    )


def invoke_chat_payload(
    *,
    case_id: str | None,
    provider: str,
    role: str,
    payload: dict[str, object],
) -> ModelInvocationResult:
    _load_env_file()
    config = _provider_config(provider)
    _validate_role(config, role)
    selected_model = _model_for_role(config, role)
    request_payload = {**payload, "model": selected_model}
    base_url = _base_url_for(config)
    api_key = _env(config.api_key_env)
    configured = _configured(config, api_key, base_url)
    if config.local_offline:
        request = ModelInvocationRequest(
            case_id=case_id,
            provider=config.provider,
            role=role,
            prompt=_prompt_from_payload(payload),
            dry_run=False,
        )
        return _invoke_local_provider(
            request=request,
            config=config,
            selected_model=selected_model,
            payload=request_payload,
        )
    return _send_payload(
        config=config,
        case_id=case_id,
        role=role,
        selected_model=selected_model,
        payload=request_payload,
        api_key=api_key,
        base_url=base_url,
        configured=configured,
    )


def _invoke_local_provider(
    *,
    request: ModelInvocationRequest,
    config: ProviderConfig,
    selected_model: str,
    payload: dict[str, object],
) -> ModelInvocationResult:
    response_payload = local_gateway_response(
        provider=config.provider,
        role=request.role,
        prompt=request.prompt,
    )
    audit_id, response_text = record_local_invocation(
        case_id=request.case_id,
        provider=config.provider,
        role=request.role,
        model=selected_model,
        request_payload=payload,
        response_payload=response_payload,
    )
    return ModelInvocationResult(
        provider=config.provider,
        role=request.role,
        selected_model=selected_model,
        configured=True,
        dry_run=False,
        request_payload=payload,
        audit_id=audit_id,
        response_text=response_text,
    )


def _send_payload(
    *,
    config: ProviderConfig,
    case_id: str | None,
    role: str,
    selected_model: str,
    payload: dict[str, object],
    api_key: str | None,
    base_url: str | None,
    configured: bool,
) -> ModelInvocationResult:
    request_for_audit = ModelInvocationRequest(
        case_id=case_id,
        provider=config.provider,
        role=role,
        prompt="",
        dry_run=False,
    )
    audit_payload = _redact_payload(payload)
    if not configured:
        audit_id = _record_audit(
            request=request_for_audit,
            config=config,
            selected_model=selected_model,
            payload=audit_payload,
            status="not_configured",
            response_text=None,
            error="Provider API key or base URL is not configured.",
            latency_ms=0,
            token_usage={},
        )
        raise ProviderNotConfiguredError(
            "Provider API key or base URL is not configured.",
            audit_id=audit_id,
        )

    started = perf_counter()
    try:
        with httpx.Client(timeout=_request_timeout_for(config)) as client:
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            response = client.post(
                f"{base_url.rstrip('/')}{config.path}",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
    except httpx.TimeoutException as exc:
        latency_ms = _elapsed_ms(started)
        audit_id = _record_audit(
            request=request_for_audit,
            config=config,
            selected_model=selected_model,
            payload=audit_payload,
            status="timeout",
            response_text=None,
            error=str(exc),
            latency_ms=latency_ms,
            token_usage={},
        )
        raise ProviderTimeoutError("Provider request timed out.", audit_id=audit_id) from exc
    except httpx.HTTPError as exc:
        latency_ms = _elapsed_ms(started)
        audit_id = _record_audit(
            request=request_for_audit,
            config=config,
            selected_model=selected_model,
            payload=audit_payload,
            status="upstream_error",
            response_text=None,
            error=str(exc),
            latency_ms=latency_ms,
            token_usage={},
        )
        raise ProviderResponseError("Provider request failed.", audit_id=audit_id) from exc
    except ValueError as exc:
        latency_ms = _elapsed_ms(started)
        audit_id = _record_audit(
            request=request_for_audit,
            config=config,
            selected_model=selected_model,
            payload=audit_payload,
            status="invalid_json",
            response_text=None,
            error=str(exc),
            latency_ms=latency_ms,
            token_usage={},
        )
        raise ProviderResponseError("Provider returned invalid JSON.", audit_id=audit_id) from exc

    response_text = _extract_response_text(body)
    token_usage = _extract_usage(body)
    latency_ms = _elapsed_ms(started)
    if not response_text.strip():
        audit_id = _record_audit(
            request=request_for_audit,
            config=config,
            selected_model=selected_model,
            payload=audit_payload,
            status="empty_response",
            response_text=response_text,
            error="Provider response did not contain message content.",
            latency_ms=latency_ms,
            token_usage=token_usage,
        )
        raise ProviderResponseError(
            "Provider response did not contain message content.",
            audit_id=audit_id,
        )
    audit_id = _record_audit(
        request=request_for_audit,
        config=config,
        selected_model=selected_model,
        payload=audit_payload,
        status="success",
        response_text=response_text,
        error=None,
        latency_ms=latency_ms,
        token_usage=token_usage,
    )

    return ModelInvocationResult(
        provider=config.provider,
        role=role,
        selected_model=selected_model,
        configured=True,
        dry_run=False,
        request_payload=audit_payload,
        audit_id=audit_id,
        response_text=response_text,
    )


def _provider_status(config: ProviderConfig) -> ModelProviderStatus:
    base_url = _base_url_for(config)
    api_key = _env(config.api_key_env)
    missing_env = []
    if config.requires_api_key and config.api_key_env and not api_key:
        missing_env.append(config.api_key_env)
    if not base_url and config.base_url_env:
        missing_env.append(config.base_url_env)
    default_models = {
        role: _model_for_role(config, role)
        for role in config.roles
    }
    runtime_ready = True
    runtime_health = "ready"
    if config.local_offline:
        runtime_health = "local-offline-ready"
    elif config.provider == "LocalVision" and not missing_env:
        runtime_ready, runtime_health = _local_vision_health(base_url, next(iter(default_models.values())))
    return ModelProviderStatus(
        provider=config.provider,
        configured=not missing_env and runtime_ready,
        base_url=base_url or config.default_base_url,
        default_models=default_models,
        roles=config.roles,
        missing_env=missing_env,
        adapter="local-offline" if config.local_offline else "openai-compatible-chat-completions",
        health=runtime_health if not missing_env else "waiting-for-configuration",
    )


def _provider_config(provider: str) -> ProviderConfig:
    normalized = provider.strip().lower()
    for config in _providers():
        if config.provider.lower() == normalized:
            return config
    supported = ", ".join(config.provider for config in _providers())
    raise ValueError(f"Unsupported provider: {provider}. Supported providers: {supported}.")


def _validate_role(config: ProviderConfig, role: str) -> None:
    if role not in config.roles:
        supported = ", ".join(config.roles)
        raise ValueError(
            f"Unsupported role for {config.provider}: {role}. Supported roles: {supported}."
        )


def _model_for_role(config: ProviderConfig, role: str) -> str:
    model_env = config.model_env_by_role.get(role)
    if model_env:
        configured_model = _env(model_env)
        if configured_model:
            return configured_model
    return config.default_models.get(role, next(iter(config.default_models.values())))


def _chat_payload(request: ModelInvocationRequest, selected_model: str) -> dict[str, object]:
    messages: list[dict[str, str]] = []
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})
    messages.append({"role": "user", "content": request.prompt})
    return {
        "model": selected_model,
        "messages": messages,
        "temperature": request.temperature,
    }


def _extract_response_text(body: dict[str, object]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _extract_usage(body: dict[str, object]) -> dict[str, object]:
    usage = body.get("usage")
    return usage if isinstance(usage, dict) else {}


def _record_audit(
    *,
    request: ModelInvocationRequest,
    config: ProviderConfig,
    selected_model: str,
    payload: dict[str, object],
    status: str,
    response_text: str | None,
    error: str | None,
    latency_ms: int,
    token_usage: dict[str, object],
) -> str:
    audit_id = str(uuid4())
    record_llm_invocation(
        ModelInvocationAudit(
            id=audit_id,
            case_id=request.case_id,
            provider=config.provider,
            role=request.role,
            model=selected_model,
            status=status,
            request_payload=payload,
            response_text=response_text,
            error=error,
            latency_ms=latency_ms,
            token_usage=token_usage,
            created_at=datetime.now(UTC).isoformat(),
        )
    )
    return audit_id


def _redact_payload(value: object) -> object:
    if isinstance(value, dict):
        redacted: dict[str, object] = {}
        for key, item in value.items():
            if key == "url" and isinstance(item, str) and item.startswith("data:image/"):
                redacted[key] = _redact_data_url(item)
            else:
                redacted[key] = _redact_payload(item)
        return redacted
    if isinstance(value, list):
        return [_redact_payload(item) for item in value]
    return value


def _redact_data_url(value: str) -> str:
    prefix, _, payload = value.partition(",")
    return f"{prefix},<base64-redacted:{len(payload)} chars>"


def _elapsed_ms(started: float) -> int:
    return max(0, round((perf_counter() - started) * 1000))


def _env(name: str) -> str | None:
    if not name:
        return None
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _base_url_for(config: ProviderConfig) -> str | None:
    if config.local_offline:
        return config.default_base_url
    return _env(config.base_url_env) or config.default_base_url


def _configured(
    config: ProviderConfig,
    api_key: str | None,
    base_url: str | None,
) -> bool:
    if not base_url:
        return False
    if config.local_offline:
        return True
    if config.requires_api_key:
        return bool(api_key)
    return True


def _providers() -> list[ProviderConfig]:
    providers = [*LOCAL_PROVIDERS, *BASE_PROVIDERS]
    if _env_flag("ENABLE_DASHSCOPE") or _env_flag("SMARTPOLICE_ENABLE_DASHSCOPE"):
        providers.extend(OPTIONAL_DASHSCOPE_PROVIDERS)
    return providers


def _env_flag(name: str) -> bool:
    value = _env(name)
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def _prompt_from_payload(payload: dict[str, object]) -> str:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(str(item["text"]))
    return "\n".join(parts)


def _local_vision_health(base_url: str | None, model: str) -> tuple[bool, str]:
    if not base_url:
        return False, "waiting-for-local-base-url"
    try:
        with httpx.Client(timeout=2) as client:
            response = client.get(f"{base_url.rstrip('/')}/models")
            response.raise_for_status()
            body = response.json()
    except httpx.HTTPError:
        return False, "waiting-for-lmstudio"
    except ValueError:
        return False, "invalid-local-model-list"
    models = body.get("data")
    if not isinstance(models, list):
        return False, "invalid-local-model-list"
    available = {
        item.get("id")
        for item in models
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    normalized_model = _normalize_model_name(model)
    normalized_available = {_normalize_model_name(item) for item in available}
    if normalized_model not in normalized_available:
        return False, f"missing-lmstudio-model:{model}"
    return True, "ready"


def _normalize_model_name(value: str) -> str:
    return value.lower().replace("_", "-").replace(".", "-").strip()


def _request_timeout_for(config: ProviderConfig) -> float:
    if config.provider == "LocalVision":
        configured = _env("LOCAL_VISION_TIMEOUT_SECONDS")
        if configured:
            try:
                return max(30.0, float(configured))
            except ValueError:
                return 300.0
        return 300.0
    return 60.0


def _load_env_file() -> None:
    if os.getenv("PYTEST_CURRENT_TEST"):
        return
    load_dotenv(ENV_PATH, override=False)
