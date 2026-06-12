"""Shared LoRA training runtime config (memory, optimizer, resume)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from backend.engine.training.presets import (
    FLUX1_TRAIN_MIN_MEMORY_GB,
    QWEN_IMAGE_TRAIN_MIN_MEMORY_GB,
    Z_IMAGE_TRAIN_MIN_MEMORY_GB,
)

OptimizerName = Literal["adam", "adamw"]

_QLORA_MIN_MEMORY_GB: dict[int, dict[str, float]] = {
    4: {
        "flux1-dev": 36.0,
        "z-image": 34.0,
        "qwen-image": 36.0,
    },
    8: {
        "flux1-dev": 42.0,
        "z-image": 40.0,
        "qwen-image": 42.0,
    },
}


@dataclass(frozen=True)
class LoraTrainRuntimeConfig:
    iterations: int
    batch_size: int
    lora_rank: int
    lora_blocks: int
    learning_rate: float
    grad_accumulate: int
    warmup_steps: int
    progress_every: int
    progress_steps: int
    checkpoint_every: int
    guidance: float
    num_augmentations: int
    qlora_bits: int | None
    grad_checkpoint: bool
    lora_scale: float
    lora_dropout: float
    lora_module_keys: list[str] | None
    optimizer_name: OptimizerName
    val_split: float
    val_every: int
    compile_step: bool
    resume_from: str | None
    weight_decay: float


def _model_id(base_model: str) -> str:
    return (base_model or "").split(":", 1)[0].strip()


def train_min_memory_gb(base_model_id: str, *, qlora_bits: int | None = None) -> float:
    mid = _model_id(base_model_id)
    if qlora_bits in _QLORA_MIN_MEMORY_GB:
        table = _QLORA_MIN_MEMORY_GB[qlora_bits]
        if mid in table:
            return table[mid]
    if mid == "z-image":
        return Z_IMAGE_TRAIN_MIN_MEMORY_GB
    if mid == "qwen-image":
        return QWEN_IMAGE_TRAIN_MIN_MEMORY_GB
    return FLUX1_TRAIN_MIN_MEMORY_GB


def assert_training_memory(base_model_id: str, mem_gb: float, *, qlora_bits: int | None) -> None:
    min_mem = train_min_memory_gb(base_model_id, qlora_bits=qlora_bits)
    if mem_gb > 0 and mem_gb < min_mem - 2:
        hint = (
            f"Enable qlora_bits=4 or 8, or reduce resolution/lora_blocks."
            if qlora_bits is None
            else "Reduce resolution/lora_blocks, or enable grad_checkpoint."
        )
        raise RuntimeError(
            f"LoRA training for {base_model_id!r} requires ~{min_mem:.0f}GB unified memory "
            f"(detected {mem_gb:.0f}GB). {hint}"
        )


def parse_lora_train_runtime_config(cfg: dict[str, Any], *, defaults: dict[str, Any]) -> LoraTrainRuntimeConfig:
    merged = {**defaults, **cfg}
    qlora_raw = merged.get("qlora_bits")
    qlora_bits: int | None = None
    if qlora_raw is not None:
        qlora_bits = int(qlora_raw)
        if qlora_bits not in (4, 8):
            raise RuntimeError(f"qlora_bits must be 4 or 8 (got {qlora_bits})")

    module_keys = merged.get("lora_module_keys")
    if module_keys is not None and not isinstance(module_keys, list):
        raise RuntimeError("lora_module_keys must be a list of module name suffixes")

    resume_from = merged.get("resume_from")
    if resume_from is not None:
        resume_from = str(resume_from).strip() or None

    lora_rank = int(merged.get("lora_rank", 8))
    lora_scale_raw = merged.get("lora_scale")
    lora_scale = float(lora_rank if lora_scale_raw is None else lora_scale_raw)

    opt = str(merged.get("optimizer") or "adam").strip().lower()
    if opt not in ("adam", "adamw"):
        raise RuntimeError(f"optimizer must be 'adam' or 'adamw' (got {opt!r})")

    return LoraTrainRuntimeConfig(
        iterations=int(merged.get("iterations", 600)),
        batch_size=int(merged.get("batch_size", 1)),
        lora_rank=lora_rank,
        lora_blocks=int(merged.get("lora_blocks") if merged.get("lora_blocks") is not None else -1),
        learning_rate=float(merged.get("learning_rate") or 1e-4),
        grad_accumulate=int(merged.get("grad_accumulate") or 4),
        warmup_steps=int(merged.get("warmup_steps") or 100),
        progress_every=int(merged.get("progress_every") or 300),
        progress_steps=int(merged.get("progress_steps") or 20),
        checkpoint_every=int(merged.get("checkpoint_every") or 300),
        guidance=float(merged.get("guidance") or 4.0),
        num_augmentations=int(merged.get("num_augmentations") or 5),
        qlora_bits=qlora_bits,
        grad_checkpoint=bool(merged.get("grad_checkpoint") or False),
        lora_scale=lora_scale,
        lora_dropout=float(merged.get("lora_dropout") or 0.0),
        lora_module_keys=module_keys,
        optimizer_name=opt,  # type: ignore[assignment]
        val_split=float(merged.get("val_split") or 0.0),
        val_every=int(merged.get("val_every") or 100),
        compile_step=bool(merged.get("compile_step") or False),
        resume_from=resume_from,
        weight_decay=float(merged.get("weight_decay") or 0.01),
    )


def build_optimizer(
    train_module: nn.Module,
    *,
    name: OptimizerName,
    learning_rate: Any,
    weight_decay: float,
) -> optim.Optimizer:
    if name == "adamw":
        return optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    return optim.Adam(learning_rate=learning_rate)


def configure_mlx_training_memory() -> None:
    try:
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    except Exception:
        pass


def adapter_meta_path(adapter_path: Path) -> Path:
    return adapter_path.with_suffix(".json")


def optimizer_state_path(adapter_path: Path) -> Path:
    return adapter_path.with_name(f"{adapter_path.stem}_optimizer.npz")


def save_training_checkpoint(
    adapter_path: Path,
    train_module: nn.Module,
    optimizer: optim.Optimizer,
    *,
    rank: int,
    meta: dict[str, Any],
) -> None:
    from backend.engine.training.lora_layers import collect_lora_safetensors

    weights = collect_lora_safetensors(train_module, rank=rank)
    weights.pop("lora_rank", None)
    mx.save_safetensors(str(adapter_path), weights)
    adapter_meta_path(adapter_path).write_text(
        __import__("json").dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    flat = dict(tree_flatten(optimizer.state))
    if flat:
        mx.savez(str(optimizer_state_path(adapter_path)), **flat)


def load_training_checkpoint(
    adapter_path: Path,
    train_module: nn.Module,
    optimizer: optim.Optimizer,
    *,
    rank: int,
) -> int:
    from backend.engine.training.lora_layers import load_lora_into_train_module

    if not adapter_path.is_file():
        raise RuntimeError(f"Resume adapter not found: {adapter_path}")
    load_lora_into_train_module(train_module, adapter_path, rank=rank)
    meta_path = adapter_meta_path(adapter_path)
    start_iter = 0
    if meta_path.is_file():
        meta = __import__("json").loads(meta_path.read_text(encoding="utf-8"))
        start_iter = int(meta.get("iteration") or 0)

    opt_path = optimizer_state_path(adapter_path)
    if opt_path.is_file():
        loaded = mx.load(str(opt_path))
        state = tree_unflatten(list(loaded.items()))
        optimizer.state = state
        mx.eval(train_module.parameters(), optimizer.state)
    return start_iter


def split_train_val_indices(n_items: int, *, val_split: float) -> tuple[list[int], list[int]]:
    if n_items < 1:
        return [], []
    if val_split <= 0 or n_items < 4:
        return list(range(n_items)), []
    n_val = max(1, min(n_items - 2, int(round(n_items * val_split))))
    train_idx = list(range(n_items - n_val))
    val_idx = list(range(n_items - n_val, n_items))
    return train_idx, val_idx
