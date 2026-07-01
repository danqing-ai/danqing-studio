"""Canonical bilingual display labels for models_registry version keys."""

from __future__ import annotations

import re
from typing import Any

_MLX_Q_RE = re.compile(r"^mlx-q(\d+)(?:-(.+))?$")
_MLX_BF16_RE = re.compile(r"^mlx-bf16(?:-(.+))?$")


def expected_version_display_name(
    version_key: str,
    version_entry: dict[str, Any] | None = None,
    *,
    skip: bool = False,
) -> dict[str, str] | None:
    """Return canonical ``versions.*.name`` or None when entry uses custom labels."""
    if skip:
        return None

    vk = str(version_key or "").strip().lower()
    if not vk:
        return None

    ver = version_entry or {}
    source_type = str(ver.get("source_type") or "")

    static: dict[str, tuple[str, str]] = {
        "fp16": ("FP16 完整版", "FP16 Full"),
        "bf16": ("BF16 完整版", "BF16 Full"),
        "fp8": ("FP8 版", "FP8"),
        "encoders": ("共享编码器", "Shared Encoders"),
        "xl-sft": ("XL SFT (4B)", "XL SFT (4B)"),
    }
    if vk in static:
        zh, en = static[vk]
        return {"zh": zh, "en": en}

    if source_type == "derived" or vk in ("int4", "int8"):
        if vk == "int8" or (isinstance(ver.get("quantization"), dict) and ver["quantization"].get("bits") == 8):
            return {"zh": "INT8 量化版", "en": "INT8 Quantized"}
        if vk == "int4" or (isinstance(ver.get("quantization"), dict) and ver["quantization"].get("bits") == 4):
            return {"zh": "INT4 量化版", "en": "INT4 Quantized"}

    bf16 = _MLX_BF16_RE.match(vk)
    if bf16:
        suffix = (bf16.group(1) or "").upper().replace("-", " ")
        if suffix:
            return {"zh": f"BF16 完整版 ({suffix})", "en": f"BF16 Full ({suffix})"}
        return {"zh": "BF16 完整版", "en": "BF16 Full"}

    q_match = _MLX_Q_RE.match(vk)
    if q_match or (source_type == "prequantized" and vk.startswith("mlx-q")):
        if q_match:
            bit = q_match.group(1)
            suffix = (q_match.group(2) or "").upper().replace("-", " ")
        else:
            bit = "4" if "q4" in vk else "8"
            suffix = vk.replace("mlx-q4", "").replace("mlx-q8", "").lstrip("-").upper().replace("-", " ")
        zh = f"{bit}-bit 量化版"
        en = f"{bit}-bit Quantized"
        if suffix:
            zh = f"{zh} ({suffix})"
            en = f"{en} ({suffix})"
        return {"zh": zh, "en": en}

    return None
