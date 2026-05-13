"""
ACE-Step weight remap — maps safetensor keys to DanQing ``_param_map`` keys.

The checkpoint uses keys of the form ``decoder.layers.{N}.{component}.{param}``.
The DanQing model stores parameters in a flat ``_param_map`` built from
``model.named_parameters()`` (mlx) / ``model.named_parameters()`` (torch).
"""
from __future__ import annotations

from typing import Any, List, Tuple


# Expected mapping from safetensor key → DanQing internal parameter path.
# Both MLX (``AceStepDiTMLX``) and CUDA (``AceStepDiTCuda``) use the same
# internal structure, so a single remap works for both backends.
#
# Safetensor key example:
#   decoder.layers.0.self_attn.k_proj.weight
#   decoder.layers.31.scale_shift_table
#   decoder.condition_embedder.weight
#
# DanQing internal path example (MLX nn.Module named_parameters):
#   layers.0.self_attn.k_proj.weight
#   layers.31.scale_shift_table
#   condition_embedder.weight
#
# The only difference is the "decoder." prefix.  The timestep embedding and
# rotary embedding buffers (cos/sin) are computed on-the-fly, not loaded from
# weights.

def _strip_decoder_prefix(safetensor_key: str) -> str:
    """Remove the leading ``decoder.`` from a safetensor key."""
    if safetensor_key.startswith("decoder."):
        return safetensor_key[len("decoder."):]
    return safetensor_key


def remap_ace_step_weights(
    raw_weights: List[Tuple[str, Any]],
) -> List[Tuple[str, Any]]:
    """Convert safetensor keys to DanQing internal parameter paths.

    Args:
        raw_weights: List of (key, tensor) pairs from safetensors.

    Returns:
        List of (internal_key, tensor) pairs suitable for ``TransformerBase.load_weights``.
    """
    remapped: List[Tuple[str, Any]] = []
    for key, tensor in raw_weights:
        internal_key = _strip_decoder_prefix(key)
        remapped.append((internal_key, tensor))
    return remapped
