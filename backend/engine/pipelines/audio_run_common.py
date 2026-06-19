"""Shared audio inference helpers (LoRA merge, generator load)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from backend.core.contracts import (
    AudioEditRequest,
    AudioGenerationRequest,
    parse_model_version,
)
from backend.engine._transformer_registry import merge_audio_lora_adapters
from backend.engine.pipelines.audio_model_load import load_audio_generator


def apply_audio_lora_adapters(
    pipeline: Any,
    *,
    family: str,
    generator: Any,
    request: AudioGenerationRequest | AudioEditRequest,
    base_model_id: str,
    entry: Any,
    on_log: Callable[..., None] | None,
) -> None:
    adapters = getattr(request, "adapters", None) or []
    if not adapters:
        return
    lora_support = False
    if entry is not None:
        ui = getattr(entry, "ui", None) or {}
        params = ui.get("parameters") if isinstance(ui, dict) else {}
        lora_support = bool((params or {}).get("lora_support"))
    if not lora_support:
        raise RuntimeError(
            f"Model {base_model_id!r} does not declare LoRA support; "
            "remove adapters from the request or use an LoRA-capable audio model."
        )
    dit = getattr(generator, "_dit", None)
    if dit is None:
        raise RuntimeError("ACE-Step generator has no loaded DiT; cannot merge LoRA adapters.")
    from backend.engine.runtime.mlx import MLXContext

    if not isinstance(pipeline.ctx, MLXContext):
        raise RuntimeError(
            "ACE-Step LoRA merge is only implemented on the MLX runtime; "
            f"current runtime is {type(pipeline.ctx).__name__}."
        )
    merge_audio_lora_adapters(
        family=family,
        model=dit,
        adapters=list(adapters),
        base_model_id=base_model_id,
        project_root=pipeline._project_root,
        registry=pipeline._registry,
        ctx=pipeline.ctx,
        on_log=on_log,
    )


def load_audio_generator_for_request(
    pipeline: Any,
    *,
    family: str,
    bundle_root: Path,
    entry: Any,
    version_key: str | None,
    request: AudioGenerationRequest | AudioEditRequest,
    on_log: Callable[..., None] | None = None,
) -> Any:
    from backend.engine.common.bundle.quant_inference import assert_quantized_dit_lora_compatible

    assert_quantized_dit_lora_compatible(entry, version_key, getattr(request, "adapters", None))
    allow_cache = not (getattr(request, "adapters", None) or [])
    generator = load_audio_generator(
        ctx=pipeline.ctx,
        family=family,
        bundle_root=bundle_root,
        entry=entry,
        version_key=version_key,
        model_cache=pipeline._cache if allow_cache else None,
    )
    base_model_id, _ = parse_model_version(getattr(request, "model", "") or "")
    apply_audio_lora_adapters(
        pipeline,
        family=family,
        generator=generator,
        request=request,
        base_model_id=base_model_id,
        entry=entry,
        on_log=on_log,
    )
    return generator
