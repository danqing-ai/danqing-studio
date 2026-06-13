"""Training preset profiles (mlx-examples/flux dreambooth defaults)."""

from __future__ import annotations

from typing import Any

PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "iterations": 300,
        "lora_rank": 4,
        "lora_blocks": 8,
        "grad_accumulate": 1,
        "warmup_steps": 25,
        "progress_every": 150,
        "checkpoint_every": 150,
        "optimizer": "adam",
        "min_snr_gamma": 0.0,
    },
    "standard": {
        "iterations": 600,
        "lora_rank": 8,
        "grad_accumulate": 4,
        "progress_every": 300,
        "checkpoint_every": 300,
        "optimizer": "adamw",
        "grad_checkpoint": True,
        "min_snr_gamma": 5.0,
        "prior_loss_weight": 1.0,
        "val_split": 0.1,
        "val_every": 100,
    },
    "quality": {
        "iterations": 1200,
        "lora_rank": 8,
        "grad_accumulate": 8,
        "progress_every": 600,
        "checkpoint_every": 600,
    },
}

FLUX1_TRAIN_MIN_MEMORY_GB = 50.0
Z_IMAGE_TRAIN_MIN_MEMORY_GB = 48.0
QWEN_IMAGE_TRAIN_MIN_MEMORY_GB = 52.0
TRAINABLE_BASE_MODELS: frozenset[str] = frozenset({"flux1-dev", "z-image", "qwen-image"})

Z_IMAGE_PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "iterations": 400,
        "lora_rank": 8,
        "lora_blocks": 12,
        "grad_accumulate": 4,
        "progress_every": 200,
        "checkpoint_every": 200,
        "learning_rate": 1e-4,
        "guidance": 5.0,
        "optimizer": "adamw",
    },
    "standard": {
        "iterations": 800,
        "lora_rank": 16,
        "lora_blocks": 16,
        "grad_accumulate": 4,
        "progress_every": 400,
        "checkpoint_every": 400,
        "learning_rate": 1e-4,
        "guidance": 5.0,
        "optimizer": "adamw",
        "min_snr_gamma": 5.0,
        "prior_loss_weight": 1.0,
        "val_split": 0.1,
        "val_every": 100,
    },
    "quality": {
        "iterations": 1500,
        "lora_rank": 16,
        "lora_blocks": 24,
        "grad_accumulate": 8,
        "progress_every": 500,
        "checkpoint_every": 500,
        "learning_rate": 5e-5,
        "guidance": 5.0,
    },
}

QWEN_IMAGE_PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "iterations": 400,
        "lora_rank": 8,
        "lora_blocks": 12,
        "grad_accumulate": 4,
        "progress_every": 200,
        "checkpoint_every": 200,
        "learning_rate": 1e-4,
        "grad_checkpoint": True,
        "optimizer": "adamw",
    },
    "standard": {
        "iterations": 800,
        "lora_rank": 16,
        "lora_blocks": 16,
        "grad_accumulate": 4,
        "progress_every": 400,
        "checkpoint_every": 400,
        "learning_rate": 1e-4,
        "optimizer": "adamw",
        "grad_checkpoint": True,
        "min_snr_gamma": 5.0,
        "prior_loss_weight": 1.0,
        "val_split": 0.1,
        "val_every": 100,
    },
    "quality": {
        "iterations": 1500,
        "lora_rank": 16,
        "lora_blocks": 24,
        "grad_accumulate": 8,
        "progress_every": 500,
        "checkpoint_every": 500,
        "learning_rate": 5e-5,
    },
}


def train_min_memory_gb(base_model_id: str) -> float:
    mid = (base_model_id or "").split(":", 1)[0].strip()
    if mid == "z-image":
        return Z_IMAGE_TRAIN_MIN_MEMORY_GB
    if mid == "qwen-image":
        return QWEN_IMAGE_TRAIN_MIN_MEMORY_GB
    return FLUX1_TRAIN_MIN_MEMORY_GB


def resolve_preset(name: str | None, *, base_model: str = "flux1-dev") -> dict[str, Any]:
    key = (name or "standard").strip().lower()
    if key == "custom":
        return {}
    mid = (base_model or "").split(":", 1)[0].strip()
    if mid == "z-image":
        table = Z_IMAGE_PRESETS
    elif mid == "qwen-image":
        table = QWEN_IMAGE_PRESETS
    else:
        table = PRESETS
    if key not in table:
        raise ValueError(f"Unknown training preset {name!r}; choose quick|standard|quality|custom")
    return dict(table[key])


_TRAINING_REQUEST_EXCLUDE = frozenset(
    {
        "base_model",
        "dataset_id",
        "preset",
        "output_name",
        "auto_register",
        "priority",
        "metadata",
    }
)


def merge_training_request_config(request: Any, preset: dict[str, Any]) -> dict[str, Any]:
    """Merge preset with optional request overrides; explicit nulls do not wipe preset."""
    overrides = request.model_dump(exclude=set(_TRAINING_REQUEST_EXCLUDE), exclude_none=True)
    return {**preset, **overrides}
