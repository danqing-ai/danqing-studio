"""LTX video model weight remapping — diffusers -> DanQing key translation.

Matches the diffusers ``LTXVideoTransformer3DModel`` checkpoint structure:
  715 keys total: 15 top-level + 25 per block x 28 blocks.
"""
from __future__ import annotations

import re
from typing import Any


def remap_ltx_weights(weights: dict[str, Any]) -> dict[str, Any]:
    """Translate diffusers LTX weight keys to DanQing LTXTransformer keys.

    Handles:
    - Flat layer naming (no nn.Sequential in our code)
    - proj_in Linear -> PatchEmbed3D Conv3d weight reshape
    - Per-block attention / MLP renaming
    """

    remapped: dict[str, Any] = {}

    # Exact top-level rewrites
    _TOP_MAP: dict[str, str] = {
        "proj_in.bias":                                 "patch_embed.proj.bias",
        "time_embed.emb.timestep_embedder.linear_1.weight": "time_embed.mlp_in.weight",
        "time_embed.emb.timestep_embedder.linear_1.bias":   "time_embed.mlp_in.bias",
        "time_embed.emb.timestep_embedder.linear_2.weight": "time_embed.mlp_out.weight",
        "time_embed.emb.timestep_embedder.linear_2.bias":   "time_embed.mlp_out.bias",
        "time_embed.linear.weight":                     "time_embed_out.weight",
        "time_embed.linear.bias":                       "time_embed_out.bias",
        "scale_shift_table":                            "output_modulation",
        "caption_projection.linear_1.weight":           "caption_proj_in.weight",
        "caption_projection.linear_1.bias":             "caption_proj_in.bias",
        "caption_projection.linear_2.weight":           "caption_proj_out.weight",
        "caption_projection.linear_2.bias":             "caption_proj_out.bias",
        "proj_out.weight":                              "proj_out.weight",
        "proj_out.bias":                                "proj_out.bias",
    }

    _BLOCK_RE = re.compile(r"^transformer_blocks\.(\d+)\.(.*)$")

    for key, tensor in weights.items():
        # proj_in.weight needs reshape: diffusers Linear(128,2048) [2048,128]
        # -> MLX Conv3d weight [2048, 1, 1, 1, 128]
        if key == "proj_in.weight":
            remapped["patch_embed.proj.weight"] = _reshape_proj_in_weight(tensor)
            continue

        if key in _TOP_MAP:
            remapped[_TOP_MAP[key]] = tensor
            continue

        m = _BLOCK_RE.match(key)
        if m:
            block_idx = m.group(1)
            tail = m.group(2)
            new_key = _remap_block_tail(block_idx, tail)
            if new_key is not None:
                remapped[new_key] = tensor
            continue

        remapped[key] = tensor

    return remapped


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BLOCK_TAIL_MAP: dict[str, str] = {
    # self-attention (attn1)
    "attn1.norm_q.weight":      "self_attn.q_norm.weight",
    "attn1.norm_k.weight":      "self_attn.k_norm.weight",
    "attn1.to_q.weight":        "self_attn.to_q.weight",
    "attn1.to_q.bias":          "self_attn.to_q.bias",
    "attn1.to_k.weight":        "self_attn.to_k.weight",
    "attn1.to_k.bias":          "self_attn.to_k.bias",
    "attn1.to_v.weight":        "self_attn.to_v.weight",
    "attn1.to_v.bias":          "self_attn.to_v.bias",
    "attn1.to_out.0.weight":    "self_attn.to_out.weight",
    "attn1.to_out.0.bias":      "self_attn.to_out.bias",
    # cross-attention (attn2)
    "attn2.norm_q.weight":      "cross_attn.q_norm.weight",
    "attn2.norm_k.weight":      "cross_attn.k_norm.weight",
    "attn2.to_q.weight":        "cross_attn.to_q.weight",
    "attn2.to_q.bias":          "cross_attn.to_q.bias",
    "attn2.to_k.weight":        "cross_attn.to_k.weight",
    "attn2.to_k.bias":          "cross_attn.to_k.bias",
    "attn2.to_v.weight":        "cross_attn.to_v.weight",
    "attn2.to_v.bias":          "cross_attn.to_v.bias",
    "attn2.to_out.0.weight":    "cross_attn.to_out.weight",
    "attn2.to_out.0.bias":      "cross_attn.to_out.bias",
    # feed-forward (ff)
    "ff.net.0.proj.weight":     "mlp_in.weight",
    "ff.net.0.proj.bias":       "mlp_in.bias",
    "ff.net.2.weight":          "mlp_out.weight",
    "ff.net.2.bias":            "mlp_out.bias",
    # per-block scale_shift_table
    "scale_shift_table":        "scale_shift_table",
}


def _remap_block_tail(block_idx: str, tail: str) -> str | None:
    mapped = _BLOCK_TAIL_MAP.get(tail)
    if mapped is None:
        return None
    return f"blocks.{block_idx}.{mapped}"


def _reshape_proj_in_weight(tensor: Any) -> Any:
    """Reshape diffusers Linear(128,2048) weight [2048,128] -> MLX Conv3d [2048,1,1,1,128]."""
    out_ch, in_ch = tensor.shape
    return tensor.reshape(out_ch, 1, 1, 1, in_ch)  # type: ignore[attr-defined]


def restore_diffusers_names_from_mlx_forge_ltx(weights: dict[str, Any]) -> dict[str, Any]:
    """Undo ``mlx_forge.recipes.ltx_23.sanitize_transformer_key`` so ``remap_ltx_weights`` can run.

    mlx-forge rewrites diffusers tails (``.ff.net.0.proj.`` → ``.ff.proj_in.`` etc.); DanQing's
    ``remap_ltx_weights`` expects the original diffusers-style keys.
    """
    out: dict[str, Any] = {}
    for k, v in weights.items():
        nk = k
        nk = nk.replace(".ff.proj_out.", ".ff.net.2.")
        nk = nk.replace(".ff.proj_in.", ".ff.net.0.proj.")
        nk = nk.replace(".audio_ff.proj_out.", ".audio_ff.net.2.")
        nk = nk.replace(".audio_ff.proj_in.", ".audio_ff.net.0.proj.")
        nk = nk.replace(".linear2.", ".linear_2.")
        nk = nk.replace(".linear1.", ".linear_1.")
        nk = re.sub(r"(\.attn[12]\.)to_out\.(weight|bias)$", r"\1to_out.0.\2", nk)
        out[nk] = v
    return out
    for k, v in weights.items():
        nk = k
        nk = nk.replace(".ff.proj_out.", ".ff.net.2.")
        nk = nk.replace(".ff.proj_in.", ".ff.net.0.proj.")
        nk = nk.replace(".audio_ff.proj_out.", ".audio_ff.net.2.")
        nk = nk.replace(".audio_ff.proj_in.", ".audio_ff.net.0.proj.")
        nk = nk.replace(".linear2.", ".linear_2.")
        nk = nk.replace(".linear1.", ".linear_1.")
        nk = re.sub(r"(\.attn[12]\.)to_out\.(weight|bias)$", r"\1to_out.0.\2", nk)
        out[nk] = v
    return out
