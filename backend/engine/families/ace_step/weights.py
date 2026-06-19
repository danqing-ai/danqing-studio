"""
ACE-Step weight remap — maps safetensor keys to DanQing ``_param_map`` keys.

MLX/CUDA loading lives in ``weights_mlx.py`` (and PyTorch paths use registry remap).
"""
from __future__ import annotations

import re
from typing import Any, List, Tuple

ACE_STEP_XL_MODEL_IDS = frozenset(
    {
        "ace-step-xl-sft",
        "ace-step-xl-turbo",
        "ace-step-xl-base",
    }
)
ACE_STEP_LORA_MODEL_IDS = ACE_STEP_XL_MODEL_IDS

ACE_STEP_DIT_SUBDIR_BY_MODEL: dict[str, str] = {
    "ace-step-xl-sft": "acestep-v15-xl-sft",
    "ace-step-xl-turbo": "acestep-v15-xl-turbo",
    "ace-step-xl-base": "acestep-v15-xl-base",
}


def ace_step_dit_subdir_for_model(model_id: str) -> str | None:
    mid = (model_id or "").split(":", 1)[0].strip()
    return ACE_STEP_DIT_SUBDIR_BY_MODEL.get(mid)


def ace_step_lora_scope_key(value: str) -> str:
    mid = (value or "").split(":", 1)[0].strip()
    if mid in ACE_STEP_XL_MODEL_IDS:
        return "ace_step_xl"
    return mid


def ace_step_lora_base_compatible(model_id: str, lora_base: str) -> bool:
    model_key = (model_id or "").split(":", 1)[0].strip()
    lora_key = (lora_base or "").split(":", 1)[0].strip()
    if not lora_key or lora_key == model_key:
        return True
    if model_key in ACE_STEP_XL_MODEL_IDS and lora_key in ACE_STEP_XL_MODEL_IDS:
        return True
    return False


def remap_ace_step_lora_module_prefix(module: str) -> str:
    """Map LoRA checkpoint module stem → DanQing ``AceStepDiTMLX`` prefix (no ``.weight``)."""
    m = (module or "").strip().strip(".")
    while ".." in m:
        m = m.replace("..", ".")
    for pref in (
        "base_model.model.",
        "base_model.",
        "pipe.dit.",
        "pipe.dit.decoder.",
        "dit.decoder.",
        "dit.",
        "decoder.",
        "model.",
    ):
        if m.startswith(pref):
            m = m[len(pref) :]
    if m.endswith(".weight") or m.endswith(".bias"):
        m = m.rsplit(".", 1)[0]
    return m


def remap_ace_step_lora_keys(lora_weights: dict) -> dict[str, tuple[Any, Any, float]]:
    """Group ACE-Step / PEFT LoRA tensors for ``merge_lora_adapters_common``."""
    from backend.engine.common.bundle.weights import _lora_key_to_module

    default_alpha = 128.0
    alphas_by_tgt: dict[str, float] = {}
    for key, tensor in lora_weights.items():
        lk = key.lower()
        if "alpha" not in lk:
            continue
        if any(x in lk for x in ("lora_down", "lora_up", "lora_a", "lora_b")):
            continue
        if not lk.endswith(".alpha"):
            continue
        base = key[: -len(".alpha")] if key.lower().endswith(".alpha") else key
        tgt = remap_ace_step_lora_module_prefix(_lora_key_to_module(base))
        if not tgt:
            continue
        try:
            val = tensor.item() if hasattr(tensor, "item") else float(tensor)
            alphas_by_tgt[tgt] = float(val)
        except (TypeError, ValueError):
            pass

    groups: dict[str, dict[str, Any]] = {}
    for key, tensor in lora_weights.items():
        lk = key.lower()
        if lk.endswith(".alpha") and "lora_" not in lk:
            continue
        module = _lora_key_to_module(key)
        if module not in groups:
            groups[module] = {}
        if re.search(r"\.lora_a(?:\.default)?\.(?:weight|bias)$", lk) or re.search(
            r"\.lora_down(?:\.default)?\.(?:weight|bias)$", lk
        ):
            groups[module]["down"] = tensor
        elif re.search(r"\.lora_b(?:\.default)?\.(?:weight|bias)$", lk) or re.search(
            r"\.lora_up(?:\.default)?\.(?:weight|bias)$", lk
        ):
            groups[module]["up"] = tensor

    out: dict[str, tuple[Any, Any, float]] = {}
    for hf_stem, parts in groups.items():
        if "down" not in parts or "up" not in parts:
            continue
        tgt = remap_ace_step_lora_module_prefix(hf_stem)
        if not tgt:
            continue
        alpha = float(alphas_by_tgt.get(tgt, default_alpha))
        out[tgt] = (parts["down"], parts["up"], alpha)
    return out


def _strip_decoder_prefix(safetensor_key: str) -> str:
    if safetensor_key.startswith("decoder."):
        return safetensor_key[len("decoder.") :]
    return safetensor_key


def _convert_decoder_tensor_for_mlx(safetensor_key: str, array: Any) -> Tuple[str, Any]:
    """Map PyTorch decoder checkpoint keys/layout to ``AceStepDiTMLX`` parameters."""
    import numpy as np

    key = _strip_decoder_prefix(safetensor_key)
    np_val = np.asarray(array, dtype=np.float32)

    if key.startswith("proj_in.1."):
        key = key.replace("proj_in.1.", "proj_in.", 1)
        if key.endswith(".weight"):
            np_val = np_val.swapaxes(1, 2)
    elif key.startswith("proj_out.1."):
        key = key.replace("proj_out.1.", "proj_out.", 1)
        if key.endswith(".weight"):
            np_val = np_val.transpose(1, 2, 0)

    return key, np_val


def remap_ace_step_weights(
    raw_weights: List[Tuple[str, Any]],
) -> List[Tuple[str, Any]]:
    remapped: List[Tuple[str, Any]] = []
    for key, tensor in raw_weights:
        remapped.append((_strip_decoder_prefix(key), tensor))
    return remapped
