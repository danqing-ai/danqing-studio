"""Load MLX affine-quantized checkpoints into QuantizedLinear (no dequantize)."""

from __future__ import annotations

from typing import Any

from backend.engine.common.model.base import _assign_param_tensor, _mlx_affine_infer_bits_and_group_size

_QUANT_AFFINE_FIELDS = frozenset({"scales", "biases"})


def _resolve_module_attr(root: Any, key: str) -> tuple[Any, str]:
    parts = key.split(".")
    if len(parts) < 2:
        raise RuntimeError(f"Invalid quantized module key: {key!r}")
    obj: Any = root
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj, parts[-1]


def _assign_quant_affine_field(root: Any, key: str, tensor: Any) -> None:
    """Assign MLX affine-quant ``scales`` / ``biases`` without slice corruption.

    ``nn.quantize`` initializes these buffers as float32, but checkpoints store
    bfloat16. In-place ``param[:] = bf16_tensor`` corrupts values on Metal; replace
    the module attribute instead (matches ``nn.Module.load_weights`` behavior).
    """
    import mlx.core as mx

    parent, attr = _resolve_module_attr(root, key)
    if attr not in _QUANT_AFFINE_FIELDS:
        raise RuntimeError(f"Expected scales/biases key, got {key!r}")
    value = mx.array(tensor, dtype=mx.bfloat16)
    if tuple(getattr(parent, attr).shape) != tuple(value.shape):
        raise RuntimeError(
            f"{key}: shape mismatch {tuple(getattr(parent, attr).shape)} vs {tuple(value.shape)}"
        )
    setattr(parent, attr, value)


def collect_affine_quant_bases(weight_dict: dict[str, Any]) -> set[str]:
    bases: set[str] = set()
    for key in weight_dict:
        if not key.endswith(".scales"):
            continue
        base = key[:-7]
        if f"{base}.weight" in weight_dict:
            bases.add(base)
    return bases


def apply_quantized_skeleton(
    model: Any,
    bases: set[str],
    *,
    bits: int,
    group_size: int,
) -> None:
    """Convert selected ``nn.Linear`` layers to ``QuantizedLinear`` before weight assign."""
    from importlib import import_module

    nn = import_module("mlx.nn")

    if isinstance(model, nn.Module):
        def _predicate(path: str, module: Any) -> bool:
            if path not in bases:
                return False
            if not isinstance(module, nn.Linear):
                raise RuntimeError(
                    f"Affine-quant checkpoint base {path!r} does not map to nn.Linear "
                    f"(got {type(module).__name__})"
                )
            in_features = int(module.weight.shape[-1])
            if in_features % group_size != 0:
                raise RuntimeError(
                    f"Cannot quantize {path!r} for inference: in_features={in_features} "
                    f"is not divisible by group_size={group_size}"
                )
            return True

        nn.quantize(model, group_size=group_size, bits=bits, class_predicate=_predicate)
        return

    for base in sorted(bases):
        _quantize_linear_at_base(model, base, bits=bits, group_size=group_size)


def _quantize_linear_at_base(root: Any, base: str, *, bits: int, group_size: int) -> None:
    from importlib import import_module

    nn = import_module("mlx.nn")
    obj: Any = root
    for part in base.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    if not isinstance(obj, nn.Linear):
        raise RuntimeError(
            f"Affine-quant checkpoint base {base!r} does not map to nn.Linear "
            f"(got {type(obj).__name__})"
        )
    in_features = int(obj.weight.shape[-1])
    if in_features % group_size != 0:
        raise RuntimeError(
            f"Cannot quantize {base!r} for inference: in_features={in_features} "
            f"is not divisible by group_size={group_size}"
        )
    obj.to_quantized(bits=bits, group_size=group_size)


def load_weights_quantized_inference(
    model: Any,
    weights: list[tuple[str, Any]],
    *,
    strict: bool,
    ctx: Any,
    bundle_affine_bits: int | None,
    bits: int,
    group_size: int = 64,
    module_root: Any | None = None,
) -> tuple[list[str], list[str]]:
    """Load affine-quantized weights for low-VRAM inference (packed tensors, no dequantize)."""
    if hasattr(model, "_build_param_map"):
        model._build_param_map()
    elif not hasattr(model, "_param_map"):
        from backend.engine.common.codecs.vae.decoder import _collect_nn_params

        model._param_map = {}
        _collect_nn_params(model, "", model._param_map)

    weight_dict = dict(weights)
    if bundle_affine_bits is not None and int(bundle_affine_bits) != int(bits):
        raise RuntimeError(
            f"Registry/runtime bits={bits} conflicts with bundle metadata "
            f"quantization_level={bundle_affine_bits}."
        )

    sanitize = getattr(model, "sanitize", None)
    weight_dict = sanitize(weight_dict) if callable(sanitize) else weight_dict
    bases = collect_affine_quant_bases(weight_dict)
    if not bases:
        raise RuntimeError(
            "Quantized inference requires MLX affine Linear checkpoints "
            "(*.weight + *.scales); none found after sanitize()."
        )

    scales_map: dict[str, dict[str, Any]] = {}
    for base in bases:
        scales_map[base] = {
            "weight": weight_dict[f"{base}.weight"],
            "scales": weight_dict[f"{base}.scales"],
            "biases": weight_dict.get(f"{base}.biases"),
        }
        weight_key = f"{base}.weight"
        group = scales_map[base]
        dense_shape: tuple[int, int] | None = None
        p = model._param_map.get(weight_key)
        if p is not None and hasattr(p, "shape") and len(p.shape) == 2:
            dense_shape = (int(p.shape[0]), int(p.shape[1]))
        inferred_bits, inferred_gs = _mlx_affine_infer_bits_and_group_size(
            group["weight"],
            group["scales"],
            dense_weight_shape=dense_shape,
            weight_key=weight_key,
            bundle_affine_bits=bundle_affine_bits or bits,
        )
        if inferred_bits != bits:
            raise RuntimeError(
                f"{weight_key}: checkpoint implies {inferred_bits}-bit packing, "
                f"expected {bits}-bit quantized inference."
            )
        if inferred_gs != group_size:
            raise RuntimeError(
                f"{weight_key}: inferred group_size={inferred_gs}, expected {group_size}."
            )

    skeleton_root = module_root
    if skeleton_root is None:
        from importlib import import_module

        nn = import_module("mlx.nn")
        skeleton_root = model if isinstance(model, nn.Module) else model
    apply_quantized_skeleton(skeleton_root, bases, bits=bits, group_size=group_size)
    if hasattr(model, "_build_param_map"):
        model._build_param_map()
    else:
        from backend.engine.common.codecs.vae.decoder import _collect_nn_params

        model._param_map = {}
        _collect_nn_params(model, "", model._param_map)

    for param_key in model._param_map:
        if not param_key.endswith(".bias"):
            continue
        base = param_key[:-5]
        if base not in scales_map:
            continue
        if param_key in weight_dict:
            continue
        raise RuntimeError(
            f"Quantized checkpoint is missing dense bias tensor {param_key!r} (base {base!r}). "
            "Re-convert from the fp16/bf16 source with a fixed converter."
        )

    loaded: list[str] = []
    skipped: list[str] = []
    for key, tensor in weight_dict.items():
        field = key.rsplit(".", 1)[-1]
        if field in _QUANT_AFFINE_FIELDS:
            base = key[: -(len(field) + 1)]
            if base not in bases:
                skipped.append(key)
                continue
            try:
                parent, attr = _resolve_module_attr(skeleton_root, key)
            except (AttributeError, IndexError, RuntimeError) as exc:
                skipped.append(f"{key} module_missing: {exc}")
                continue
            current = getattr(parent, attr, None)
            if current is None:
                skipped.append(f"{key} module_missing: {attr!r} is None")
                continue
            if tuple(current.shape) != tuple(tensor.shape):
                skipped.append(f"{key} shape_mismatch: {current.shape} vs {tensor.shape}")
                continue
            _assign_quant_affine_field(skeleton_root, key, tensor)
            loaded.append(key)
            continue

        if key not in model._param_map:
            skipped.append(key)
            continue
        param = model._param_map[key]
        if tuple(param.shape) != tuple(tensor.shape):
            skipped.append(f"{key} shape_mismatch: {param.shape} vs {tensor.shape}")
            continue
        _assign_param_tensor(param, tensor)
        loaded.append(key)

    loaded_set = set(loaded)
    missing = [
        k
        for k in model._param_map
        if k not in loaded_set and k.rsplit(".", 1)[-1] not in _QUANT_AFFINE_FIELDS
    ]
    missing_bases = sorted(
        base
        for base in bases
        if f"{base}.scales" not in loaded_set
        or (
            f"{base}.biases" in weight_dict
            and f"{base}.biases" not in loaded_set
        )
    )
    if missing:
        preview = missing[:40]
        more = f" (+{len(missing) - 40} more)" if len(missing) > 40 else ""
        mismatches = [s for s in skipped if isinstance(s, str) and "shape_mismatch" in s]
        mm_note = (
            f" Shape mismatches (first): {mismatches[:20]!r}"
            + (f" (+{len(mismatches) - 20} more)" if len(mismatches) > 20 else "")
            if mismatches
            else ""
        )
        raise RuntimeError(
            f"Quantized weight load failed: {len(missing)} model parameter(s) missing or "
            f"shape mismatch. First keys: {preview!r}{more}.{mm_note}"
        )
    if missing_bases:
        preview = missing_bases[:40]
        more = f" (+{len(missing_bases) - 40} more)" if len(missing_bases) > 40 else ""
        raise RuntimeError(
            f"Quantized weight load failed: {len(missing_bases)} affine layer(s) missing "
            f"scales/biases assignment. First bases: {preview!r}{more}"
        )

    if strict:
        extras = [s for s in skipped if isinstance(s, str) and "shape_mismatch" not in s]
        if extras:
            prev = extras[:40]
            more = f" (+{len(extras) - 40} more)" if len(extras) > 40 else ""
            raise RuntimeError(
                f"Quantized weight load strict mode: {len(extras)} checkpoint key(s) unused. "
                f"First keys: {prev!r}{more}"
            )

    if hasattr(model, "_build_param_map"):
        model._build_param_map()
    elif hasattr(model, "_param_map"):
        from backend.engine.common.codecs.vae.decoder import _collect_nn_params

        model._param_map = {}
        _collect_nn_params(model, "", model._param_map)

    return loaded, skipped
