"""Wan video model weight mapping — original Wan keys + diffusers fallback."""
from __future__ import annotations

import re


def _is_diffusers_wan(weights: dict) -> bool:
    return any("attn1" in k for k in weights)


def _patch_embedding_weight_to_linear(key: str, tensor) -> tuple[str, object]:
    """Conv3d patch kernel → ``Linear`` weight for Wan patchify.

    Checkpoints store ``patch_embedding`` as 3D conv. We run a ``Linear`` whose columns
    match input flatten order ``[C, pt, ph, pw]`` per token (see ``WanModelMLX`` patchify).
    """
    if not (key.endswith(".weight") and "patch_embedding" in key and getattr(tensor, "ndim", 0) == 5):
        return key, tensor
    t = tensor
    # PyTorch / safetensors: ``[O, I, T, H, W]``; older MLX ports used ``[O, T, H, W, I]``.
    if int(t.shape[1]) <= 4 and int(t.shape[-1]) > 4:
        t = t.transpose(0, 4, 1, 2, 3)
    return key, t.reshape(int(t.shape[0]), -1)


def _remap_diffusers_to_wan(key: str) -> str:
    new_key = key
    new_key = new_key.replace("patch_embedding.", "patch_embedding.")
    new_key = re.sub(r"\.attn1\.to_q\.", ".self_attn.q.", new_key)
    new_key = re.sub(r"\.attn1\.to_k\.", ".self_attn.k.", new_key)
    new_key = re.sub(r"\.attn1\.to_v\.", ".self_attn.v.", new_key)
    new_key = re.sub(r"\.attn1\.to_out\.0\.", ".self_attn.o.", new_key)
    new_key = re.sub(r"\.attn2\.to_q\.", ".cross_attn.q.", new_key)
    new_key = re.sub(r"\.attn2\.to_k\.", ".cross_attn.k.", new_key)
    new_key = re.sub(r"\.attn2\.to_v\.", ".cross_attn.v.", new_key)
    new_key = re.sub(r"\.attn2\.to_out\.0\.", ".cross_attn.o.", new_key)
    new_key = new_key.replace(".ffn.net.0.proj.", ".ffn.layer_0.")
    new_key = new_key.replace(".ffn.net.2.", ".ffn.layer_2.")
    new_key = new_key.replace("scale_shift_table", "modulation")
    new_key = new_key.replace("proj_out.", "head.1.")
    return new_key


def remap_wan_weights(weights: dict) -> dict:
    """Map checkpoint keys to ``WanModelMLX._param_map`` flat names."""
    diffusers = _is_diffusers_wan(weights)
    remapped: dict = {}
    for key, tensor in weights.items():
        new_key = key
        if new_key.startswith("transformer."):
            new_key = new_key[len("transformer.") :]
        new_key = _remap_diffusers_to_wan(new_key) if diffusers else new_key
        # Original Wan FFN keys
        new_key = new_key.replace(".ffn.0.weight", ".ffn.layer_0.weight")
        new_key = new_key.replace(".ffn.0.bias", ".ffn.layer_0.bias")
        new_key = new_key.replace(".ffn.2.weight", ".ffn.layer_2.weight")
        new_key = new_key.replace(".ffn.2.bias", ".ffn.layer_2.bias")
        new_key = new_key.replace(".norm_q.weight", ".norm_q.weight")
        # Official Wan ``Head`` module: ``head.head`` linear → flat ``head.1`` (see ``_param_map``).
        new_key = new_key.replace("head.head.", "head.1.")
        new_key, tensor = _patch_embedding_weight_to_linear(new_key, tensor)
        remapped[new_key] = tensor
    return remapped
