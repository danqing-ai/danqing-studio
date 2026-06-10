"""Z-Image LoRA merge via shared MLX skeleton (with CUDA dispatch)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from backend.engine.common.bundle.lora_mlx import merge_lora_adapters_common
from backend.engine.families.z_image.weights import remap_zimage_lora_keys
from backend.engine.runtime._base import RuntimeContext


def _base_model_scope_key(value: str) -> str:
    return value.split(":", 1)[0].strip() if value else ""


def merge_z_image_lora_adapters(
    model: Any,
    adapters: Sequence[Any],
    *,
    base_model_id: str,
    project_root: Path,
    registry: Any,
    ctx: RuntimeContext,
    patch_size: int | None = None,
    on_log: Callable[[str, str], None] | None = None,
) -> None:
    ps = int(patch_size) if patch_size is not None else int(
        getattr(getattr(model, "config", None), "patch_size", 2) or 2
    )

    backend = getattr(ctx, "backend", "mlx")
    if backend == "cuda":
        import importlib
        lora_cuda = importlib.import_module("backend.engine.families.z_image.lora_cuda")
        lora_cuda.merge_z_image_lora_adapters_cuda(
            model=model,
            adapters=adapters,
            base_model_id=base_model_id,
            project_root=project_root,
            registry=registry,
            ctx=ctx,
            patch_size=ps,
            on_log=on_log,
        )
        return

    merge_lora_adapters_common(
        model=model,
        adapters=adapters,
        base_model_id=base_model_id,
        project_root=project_root,
        registry=registry,
        ctx=ctx,
        family_name="Z-Image",
        remap_groups=lambda weights: remap_zimage_lora_keys(
            weights, patch_size=max(1, ps)
        ),
        param_key_for_module=lambda module_name: f"{module_name}.weight",
        base_model_scope_key=_base_model_scope_key,
        on_log=on_log,
    )
    refresh = getattr(model, "_refresh_compiled_forward", None)
    if callable(refresh):
        refresh()
