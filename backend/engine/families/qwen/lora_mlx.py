"""Qwen Image LoRA merge via shared MLX skeleton."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from backend.engine.common.lora_mlx import merge_lora_adapters_common
from backend.engine.families.qwen.weights import remap_qwen_lora_keys
from backend.engine.runtime._base import RuntimeContext


def _base_model_scope_key(value: str) -> str:
    return value.split(":", 1)[0].strip() if value else ""


def merge_qwen_image_lora_adapters(
    model: Any,
    adapters: Sequence[Any],
    *,
    base_model_id: str,
    project_root: Path,
    registry: Any,
    ctx: RuntimeContext,
    on_log: Callable[[str, str], None] | None = None,
) -> None:
    merge_lora_adapters_common(
        model=model,
        adapters=adapters,
        base_model_id=base_model_id,
        project_root=project_root,
        registry=registry,
        ctx=ctx,
        family_name="Qwen Image",
        remap_groups=remap_qwen_lora_keys,
        param_key_for_module=lambda module_name: f"dit.{module_name}.weight",
        base_model_scope_key=_base_model_scope_key,
        on_log=on_log,
    )
