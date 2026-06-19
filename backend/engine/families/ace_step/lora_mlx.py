"""ACE-Step LoRA merge (MLX DiT)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from backend.engine.common.bundle.lora_mlx import merge_lora_adapters_common
from backend.engine.families.ace_step.weights import (
    ace_step_lora_scope_key,
    remap_ace_step_lora_keys,
)
from backend.engine.runtime._base import RuntimeContext


def merge_ace_step_lora_adapters(
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
        family_name="ACE-Step",
        remap_groups=remap_ace_step_lora_keys,
        param_key_for_module=lambda module_name: f"{module_name}.weight",
        base_model_scope_key=ace_step_lora_scope_key,
        on_log=on_log,
    )
