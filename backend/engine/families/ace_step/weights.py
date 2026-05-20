"""
ACE-Step weight remap — maps safetensor keys to DanQing ``_param_map`` keys.

MLX/CUDA loading lives in ``weights_mlx.py`` (and PyTorch paths use registry remap).
"""
from __future__ import annotations

from typing import Any, List, Tuple


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
