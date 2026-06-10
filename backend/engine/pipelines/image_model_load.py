"""Shared image DiT load path — ImagePipeline and v3 FamilyPlugin backbones."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from backend.engine._transformer_registry import get_transformer_class
from backend.engine.config.model_configs import (
    apply_image_bundle_config_merger,
    apply_image_registry_config_overrides,
)
from backend.engine.contracts import (
    local_bundle_root,
    resolve_version_block,
)
from backend.engine.cache import ModelCache
from backend.engine.common.bundle.quant_inference import (
    WeightInferenceMode,
    estimate_dit_cache_size_gb,
    resolve_inference_weight_mode,
)


def image_model_cache_key(
    entry: Any,
    version_key: str | None,
    inference_mode: WeightInferenceMode | None = None,
) -> str:
    suffix = inference_mode.cache_suffix() if inference_mode is not None else ""
    if not suffix:
        suffix = ":dense"
    return f"image:{entry.id}:{version_key or 'default'}{suffix}"


def load_image_transformer(
    *,
    ctx: Any,
    family: str,
    config: Any,
    entry: Any,
    version_key: str | None,
    project_root: Path,
    model_cache: ModelCache | None = None,
    allow_cache: bool = True,
    on_log: Callable[[str, str], None] | None = None,
) -> Any | None:
    """Load image transformer weights from bundle (registry-driven, no family branches)."""
    bundle_root = local_bundle_root(project_root, entry, version_key)
    apply_image_registry_config_overrides(entry, config)
    apply_image_bundle_config_merger(config, bundle_root)

    tp = (bundle_root / "transformer") if bundle_root else None
    if tp is None or not tp.exists():
        return None

    weights: dict[str, Any] = {}
    for shard in sorted(tp.glob("*.safetensors")):
        weights.update(ctx.load_weights(str(shard)))

    from backend.engine.common.bundle.safetensors_affine_quant import read_bundle_affine_bits_if_quantized

    bundle_affine_bits = read_bundle_affine_bits_if_quantized(weights, tp)
    inference_mode = resolve_inference_weight_mode(
        entry,
        version_key,
        ctx,
        weight_keys=frozenset(weights.keys()),
        bundle_affine_bits=bundle_affine_bits,
    )

    cache_key = image_model_cache_key(entry, version_key, inference_mode)
    if allow_cache and model_cache is not None:
        cached = model_cache.get(cache_key)
        if cached is not None:
            return cached

    trans_cls = get_transformer_class(family)
    model = trans_cls(config, ctx)

    model.load_weights(
        list(weights.items()),
        strict=False,
        ctx=ctx,
        bundle_affine_bits=bundle_affine_bits,
        inference_mode=inference_mode,
    )
    setattr(model, "_dq_inference_mode", inference_mode)
    ctx.eval(*[p for _, p in model.parameters()])

    if on_log is not None:
        from backend.engine.common.bundle.weights import parse_size_gb

        ver = resolve_version_block(entry, version_key)
        size_str = str((ver or {}).get("size") or getattr(entry, "raw", {}).get("size") or "10GB")
        disk_gb = parse_size_gb(size_str)
        est_gb = estimate_dit_cache_size_gb(disk_gb, inference_mode)
        on_log(
            "info",
            f"[load_transformer] {inference_mode.log_label()} "
            f"disk_gb={disk_gb:.1f} cache_est_gb={est_gb:.1f}",
        )

    if allow_cache and model_cache is not None:
        from backend.engine.common.bundle.weights import parse_size_gb

        ver = resolve_version_block(entry, version_key)
        size_str = ""
        if ver:
            size_str = str(ver.get("size") or "")
        if not size_str:
            raw = getattr(entry, "raw", {}) or {}
            size_str = str(raw.get("size") or "10GB")
        disk_gb = parse_size_gb(size_str)
        model_cache.put(
            cache_key,
            model,
            estimate_dit_cache_size_gb(disk_gb, inference_mode),
        )
    return model
