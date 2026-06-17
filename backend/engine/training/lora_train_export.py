"""Export helpers for LoRA training artifacts registered for inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx

from backend.engine.training.lora_train_runtime import LoraTrainRuntimeConfig


def is_dense_delta_adapter(weights: dict[str, Any]) -> bool:
    return any(str(k).endswith(".delta.weight") for k in weights)


def adapter_needs_dense_export(train_runtime: LoraTrainRuntimeConfig) -> bool:
    """DoRA and fuse_adapters require dense-delta export for inference merge."""
    return train_runtime.train_type == "dora" or train_runtime.fuse_adapters


def export_registered_adapter(
    *,
    adapter_dir: Path,
    train_module: Any,
    train_runtime: LoraTrainRuntimeConfig,
    base_model_id: str,
    final_path: Path,
    meta: dict[str, Any],
    save_adapter: Callable[[Path], None],
) -> Path:
    """Write the adapter artifact that inference will load and return its path."""
    if not final_path.is_file():
        save_adapter(final_path)

    if not adapter_needs_dense_export(train_runtime):
        return final_path

    from backend.engine.training.lora_layers import (
        collect_fused_adapter_deltas,
        load_lora_into_train_module,
    )

    load_lora_into_train_module(train_module, final_path, rank=train_runtime.lora_rank)
    export_path = adapter_dir / "registered_adapters.safetensors"
    fused = collect_fused_adapter_deltas(train_module)
    mx.save_safetensors(str(export_path), fused)
    export_meta = {
        **meta,
        "format": "dense_delta",
        "base_model": base_model_id,
        "train_type": train_runtime.train_type,
    }
    export_path.with_suffix(".json").write_text(
        json.dumps(export_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return export_path
