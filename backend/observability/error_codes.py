"""Machine-readable failure codes for dev / agent diagnosis."""

from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    BUNDLE_NOT_READY = "BUNDLE_NOT_READY"
    BUNDLE_MANIFEST_INVALID = "BUNDLE_MANIFEST_INVALID"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    BACKEND_DECLARED_BUT_MISSING = "BACKEND_DECLARED_BUT_MISSING"
    OOM = "OOM"
    WEIGHT_KEY_MISMATCH = "WEIGHT_KEY_MISMATCH"
    SHAPE_MISMATCH = "SHAPE_MISMATCH"
    CONDITIONING_ERROR = "CONDITIONING_ERROR"
    VAE_DECODE_ERROR = "VAE_DECODE_ERROR"
    CANCELLED = "CANCELLED"
    ORPHAN_RECOVERY = "ORPHAN_RECOVERY"
    FEATURE_BACKEND_GAP = "FEATURE_BACKEND_GAP"
    INTERNAL_ERROR = "INTERNAL_ERROR"


_HINTS: dict[ErrorCode, dict[str, Any]] = {
    ErrorCode.BUNDLE_NOT_READY: {
        "zh": ["检查模型是否已完整安装", "在模型库中安装缺失组件"],
        "en": ["Verify the model is fully installed", "Install missing components from the model library"],
        "checks": ["GET /api/models/{model_id}", "inspect models/ bundle directories"],
    },
    ErrorCode.BACKEND_UNAVAILABLE: {
        "zh": ["检查本机 MLX/CUDA 是否可用"],
        "en": ["Check MLX/CUDA availability on this host"],
        "checks": ["GET /api/system/health"],
    },
    ErrorCode.OOM: {
        "zh": ["降低分辨率或步数", "调整设置中的 mlx_memory_limit"],
        "en": ["Reduce resolution or steps", "Adjust mlx_memory_limit in settings"],
        "checks": ["GET /api/system/health", "GET /api/settings/system"],
    },
    ErrorCode.CANCELLED: {
        "zh": ["用户取消，非引擎缺陷"],
        "en": ["User cancelled — not an engine defect"],
        "checks": [],
    },
    ErrorCode.ORPHAN_RECOVERY: {
        "zh": ["服务重启导致任务中断，请重新提交"],
        "en": ["Server restarted mid-run — resubmit the task"],
        "checks": [],
    },
    ErrorCode.FEATURE_BACKEND_GAP: {
        "zh": ["该能力在当前后端不可用（如 ControlNet 仅 MLX）"],
        "en": ["Capability unavailable on current backend (e.g. ControlNet MLX-only)"],
        "checks": ["GET /api/registry", "GET /api/system/health"],
    },
    ErrorCode.INTERNAL_ERROR: {
        "zh": ["查看 trace.json 与 error 日志行"],
        "en": ["Inspect trace.json and error-level log lines"],
        "checks": ["GET /api/tasks/{id}/logs", "outputs/work/{task_id}/trace.json"],
    },
}


def failure_hints(code: ErrorCode, *, locale: str = "zh") -> dict[str, Any]:
    row = _HINTS.get(code, _HINTS[ErrorCode.INTERNAL_ERROR])
    loc = "en" if locale.startswith("en") else "zh"
    return {
        "hints": row.get(loc, row.get("zh", [])),
        "recommended_checks": list(row.get("checks", [])),
    }


def classify_exception_message(message: str) -> ErrorCode:
    """Best-effort mapping from legacy error strings."""
    m = (message or "").lower()
    if "cancel" in m:
        return ErrorCode.CANCELLED
    if "orphan" in m or "restarted while task" in m:
        return ErrorCode.ORPHAN_RECOVERY
    if (
        "out of memory" in m
        or "oom" in m
        or "memory" in m
        and "alloc" in m
        or "sigkill" in m
        or "code=-9" in m
        or "exit code -9" in m
        or "unified memory pressure" in m
    ):
        return ErrorCode.OOM
    if "bundle" in m and ("missing" in m or "not ready" in m or "incomplete" in m):
        return ErrorCode.BUNDLE_NOT_READY
    if "manifest" in m:
        return ErrorCode.BUNDLE_MANIFEST_INVALID
    if "cuda" in m and ("unavailable" in m or "not implemented" in m):
        return ErrorCode.FEATURE_BACKEND_GAP
    if "mlx" in m and "unavailable" in m:
        return ErrorCode.BACKEND_UNAVAILABLE
    if "shape" in m or "dimension" in m:
        return ErrorCode.SHAPE_MISMATCH
    if "key" in m and ("weight" in m or "param" in m or "remap" in m):
        return ErrorCode.WEIGHT_KEY_MISMATCH
    if "vae" in m and "decode" in m:
        return ErrorCode.VAE_DECODE_ERROR
    if "encode" in m or "tokenizer" in m or "conditioning" in m:
        return ErrorCode.CONDITIONING_ERROR
    return ErrorCode.INTERNAL_ERROR


def classify_failed_span(span_name: str | None, error_message: str) -> ErrorCode:
    code = classify_exception_message(error_message)
    if code != ErrorCode.INTERNAL_ERROR:
        return code
    if span_name == "validate_bundle":
        return ErrorCode.BUNDLE_NOT_READY
    if span_name in ("encode", "encode_prompt"):
        return ErrorCode.CONDITIONING_ERROR
    if span_name in ("load_backbone", "load_transformer"):
        return ErrorCode.WEIGHT_KEY_MISMATCH
    if span_name == "decode" or span_name == "decode_vae":
        return ErrorCode.VAE_DECODE_ERROR
    if span_name == "denoise":
        return ErrorCode.SHAPE_MISMATCH
    return ErrorCode.INTERNAL_ERROR
