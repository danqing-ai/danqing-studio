"""DMD LoRA loader (pre-merge) for LongCat-Video-Avatar-1.5.

PyTorch reference: `pipeline_longcat_video.py` calls
`self.dit.load_lora(...)` + `self.dit.enable_loras([...])` to runtime-patch
each LoRA'd module's `forward` (see `longcat_video_dit.py:197-249`). For MLX
inference, we instead **pre-merge**: load LoRA tensors once, compute
    W' = W_base + multiplier * alpha_scale * (lora_up @ lora_down)
and overwrite the corresponding base weights. This avoids runtime monkey-
patching, costs no extra forward-pass FLOPs, and works for any always-on
inference path (DMD distillation, the current use case).

LoRA key encoding (Meituan-specific, from
`longcat_video_dit.py:319` source):
- Tensor name format: `lora___lorahyphen___<MODULE_PATH>.<TENSOR>`
- `___lorahyphen___` decodes to `.` in MODULE_PATH (since `.` is reserved
  for safetensors hierarchy).
- TENSOR is one of `lora_down.weight`, `lora_up.weight`, `alpha_scale`, or
  for fused projections (qkv, kv_linear): `lora_up.blocks.{idx}.weight`.

For fused-QKV / fused-KV layers (where the base Linear's output is a
concatenation of N projections):
- `lora_down.weight` shape: `(N * rank, in_features)` — shared down projection
- `lora_up.blocks.{i}.weight` shape: `(out_per_split, rank)` — one up per split
The merged delta for split i is `up_i @ down[i*rank:(i+1)*rank, :]`,
concatenated along output axis 0.

This module ONLY handles the merge math; the lifecycle (download + dtype
handling + parameter update) is in `pipeline_mlx.py`.
"""

from __future__ import annotations

from typing import Any, Iterable

import mlx.core as mx


def decode_module_name(lora_key: str) -> tuple[str, str]:
    """Split a Meituan-encoded LoRA tensor name into `(module_path, tensor)`.

    Example:
        `lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_down.weight`
        →  `("blocks.0.attn.qkv", "lora_down.weight")`

        `lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_up.blocks.1.weight`
        →  `("blocks.0.attn.qkv", "lora_up.blocks.1.weight")`
    """
    # Strip the `lora___lorahyphen___` prefix and decode the rest.
    PREFIX = "lora___lorahyphen___"
    SEP = "___lorahyphen___"
    if not lora_key.startswith(PREFIX):
        raise ValueError(f"Unexpected LoRA key prefix: {lora_key}")
    body = lora_key[len(PREFIX) :]
    # The body has form `<encoded_module>.<tail>` where `tail` starts with
    # `lora_down` / `lora_up` / `alpha_scale`. Find the first `.` AFTER all
    # `___lorahyphen___` segments.
    decoded = body.replace(SEP, ".")
    # Now decoded looks like `blocks.0.attn.qkv.lora_down.weight`. The tail
    # always starts with `lora_down`, `lora_up`, or `alpha_scale` — find that.
    for marker in (".lora_down.", ".lora_up.", ".alpha_scale"):
        idx = decoded.rfind(marker)
        if idx >= 0:
            module_path = decoded[:idx]
            tensor = decoded[idx + 1 :]
            return module_path, tensor
    raise ValueError(f"Cannot find LoRA tail marker in: {decoded}")


def group_lora_tensors(state_dict: dict[str, mx.array]) -> dict[str, dict[str, mx.array]]:
    """Group LoRA tensors by target module_path.

    Returns `{module_path: {tensor_subkey: tensor}}`. `tensor_subkey` is e.g.
    `lora_down.weight`, `lora_up.weight`, `lora_up.blocks.0.weight`,
    `alpha_scale`.
    """
    grouped: dict[str, dict[str, mx.array]] = {}
    for k, v in state_dict.items():
        module_path, tail = decode_module_name(k)
        grouped.setdefault(module_path, {})[tail] = v
    return grouped


def compute_merged_delta(group: dict[str, mx.array], multiplier: float = 1.0) -> mx.array:
    """Compute the weight delta `multiplier * alpha_scale * (up @ down)` for
    one LoRA target module. Handles both single and split-fused variants.

    Returns the delta in PT weight layout `(out_features, in_features)`. The
    caller is responsible for adding this to the base module's weight tensor.
    """
    if "lora_down.weight" not in group:
        raise KeyError(f"LoRA group missing lora_down.weight; keys: {list(group)}")
    down = group["lora_down.weight"]  # (rank * N_splits, in_features)
    alpha_scale = float(group.get("alpha_scale", mx.array(1.0)).item())

    # Detect split: presence of `lora_up.blocks.{i}.weight`
    split_keys = sorted(k for k in group if k.startswith("lora_up.blocks."))
    if split_keys:
        n_splits = len(split_keys)
        rank = down.shape[0] // n_splits
        assert down.shape[0] == n_splits * rank, (
            f"down shape {down.shape} doesn't divide into {n_splits} splits"
        )
        delta_parts = []
        for i, sk in enumerate(split_keys):
            up_i = group[sk]  # (out_per_split, rank)
            down_i = down[i * rank : (i + 1) * rank]  # (rank, in_features)
            # PT W = up @ down. Same shape: (out, rank) @ (rank, in) = (out, in)
            delta_i = up_i @ down_i  # (out_per_split, in_features)
            delta_parts.append(delta_i)
        delta = mx.concatenate(delta_parts, axis=0)  # (sum out_per_split, in_features)
    else:
        if "lora_up.weight" not in group:
            raise KeyError(f"LoRA group missing lora_up.weight; keys: {list(group)}")
        up = group["lora_up.weight"]  # (out_features, rank)
        delta = up @ down  # (out_features, in_features)

    return (multiplier * alpha_scale) * delta


def _is_quantized_affine_weight(weight: mx.array) -> bool:
    dtype = getattr(weight, "dtype", None)
    dtype_text = str(getattr(dtype, "name", dtype) or dtype or "")
    shape = getattr(weight, "shape", ())
    return "uint32" in dtype_text and len(shape) == 2


def _module_at_path(root: Any, path: str) -> Any:
    obj = root
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _set_module_at_path(root: Any, path: str, module: Any) -> None:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    setattr(parent, parts[-1], module)


def _merge_quantized_linear_lora(
    mx_model: Any,
    module_path: str,
    group: dict[str, mx.array],
    *,
    multiplier: float,
    bits: int = 4,
    group_size: int = 64,
) -> None:
    from importlib import import_module

    from backend.engine.common.model.base import _mlx_affine_infer_bits_and_group_size
    from mlx.utils import tree_flatten

    nn = import_module("mlx.nn")
    mod = _module_at_path(mx_model, module_path)
    if not hasattr(mod, "weight") or not hasattr(mod, "scales"):
        raise RuntimeError(
            f"LongCat LoRA target {module_path!r} is {type(mod).__name__}; "
            "expected a linear layer with weight/scales."
        )
    if not _is_quantized_affine_weight(mod.weight):
        raise RuntimeError(f"LongCat LoRA target {module_path!r} is not affine-quantized")

    delta = compute_merged_delta(group, multiplier=multiplier).astype(mx.float32)
    bits_eff, gs = _mlx_affine_infer_bits_and_group_size(
        mod.weight,
        mod.scales,
        dense_weight_shape=tuple(delta.shape),
        weight_key=f"{module_path}.weight",
        bundle_affine_bits=bits,
    )
    dense_w = mx.dequantize(mod.weight, mod.scales, mod.biases, gs, bits_eff)
    dense_updated = dense_w.astype(mx.float32) + delta

    mod_params = dict(tree_flatten(mod.parameters()))
    bias_param = mod_params.get("bias")
    in_features = int(dense_updated.shape[1])
    out_features = int(dense_updated.shape[0])
    linear = nn.Linear(in_features, out_features, bias=bias_param is not None)
    linear.weight = dense_updated.astype(dense_w.dtype)
    if bias_param is not None:
        linear.bias = bias_param
    q_linear = linear.to_quantized(bits=bits_eff, group_size=gs)
    _set_module_at_path(mx_model, module_path, q_linear)


def merge_lora_into_model(
    mx_model,
    lora_state_dict: dict[str, mx.array],
    multiplier: float = 1.0,
    target_prefix: str = "",
    *,
    quant_bits: int | None = None,
    quant_group_size: int = 64,
) -> dict[str, list[str]]:
    """Pre-merge a LoRA into the MLX model's base weights in-place.

    Args:
        mx_model: an nn.Module whose `parameters()` includes the LoRA target weights.
        lora_state_dict: the loaded safetensors dict from `dmd_lora.safetensors`.
        multiplier: per-LoRA strength multiplier (PT's `lora.multiplier`, default 1.0).
        target_prefix: optional path prefix prepended to each decoded module_path
            before lookup in `mx_model.parameters()`. Use `""` for a flat model;
            use `"dit"` if your wrapper nests the DiT under `self.dit`.

    Returns:
        Dict with `applied` (list of merged module paths) and `unmapped`
        (LoRA keys whose target module wasn't found in the MLX model).
    """
    from mlx.utils import tree_flatten, tree_unflatten

    grouped = group_lora_tensors(lora_state_dict)

    # Snapshot existing params for in-place update
    params = dict(tree_flatten(mx_model.parameters()))

    applied: list[str] = []
    unmapped: list[str] = []
    quantized_paths: list[str] = []
    dense_updated = False

    for module_path, group in grouped.items():
        weight_key = (
            f"{target_prefix}.{module_path}.weight" if target_prefix else f"{module_path}.weight"
        )
        if weight_key not in params:
            unmapped.append(module_path)
            continue
        base_w = params[weight_key]
        if _is_quantized_affine_weight(base_w):
            if quant_bits is None:
                raise RuntimeError(
                    f"LongCat cfg_step_lora cannot merge into quantized layer {module_path!r} "
                    "without quant_bits (pre-quantized bundle)."
                )
            quantized_paths.append(module_path)
            continue
        delta = compute_merged_delta(group, multiplier=multiplier).astype(base_w.dtype)
        if delta.shape != base_w.shape:
            raise ValueError(
                f"shape mismatch merging {module_path}: "
                f"delta {tuple(delta.shape)} vs base {tuple(base_w.shape)}"
            )
        params[weight_key] = base_w + delta
        applied.append(module_path)
        dense_updated = True

    if dense_updated:
        mx_model.update(tree_unflatten(list(params.items())))

    for module_path in quantized_paths:
        _merge_quantized_linear_lora(
            mx_model,
            module_path if not target_prefix else f"{target_prefix}.{module_path}",
            grouped[module_path],
            multiplier=multiplier,
            bits=int(quant_bits or 4),
            group_size=quant_group_size,
        )
        applied.append(module_path)
    mx.eval(mx_model.parameters())  # materialize the merged tensors

    return {"applied": applied, "unmapped": unmapped}


def list_lora_targets(lora_state_dict: dict[str, mx.array]) -> list[str]:
    """Diagnostic: list all module paths the LoRA targets. Useful for
    pre-flight validation against the MLX model's parameter tree.
    """
    return sorted(group_lora_tensors(lora_state_dict).keys())
