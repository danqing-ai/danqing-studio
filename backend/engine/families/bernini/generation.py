"""Bernini-R video generation via mlx-video (Shape C family generator)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from backend.engine.config.model_configs import BerniniConfig
from backend.engine.families.bernini.generation_mlx import BerniniMlxGenerator


class BerniniGeneratorProto(Protocol):
    def load(self) -> None: ...

    def generate_and_save(
        self,
        *,
        prompt: str,
        output_path: str,
        width: int,
        height: int,
        num_frames: int,
        fps: float,
        seed: int,
        steps: int,
        guidance: float,
        step_distill: bool,
        image_path: str | None,
        on_log: Any | None,
    ) -> str: ...


def create_bernini_generator(
    ctx: Any,
    bundle_root: Path,
    *,
    config: BerniniConfig | None = None,
    entry: Any | None = None,
    version_key: str | None = None,
) -> BerniniGeneratorProto:
    backend = getattr(ctx, "backend", "mlx")
    if backend != "mlx":
        raise RuntimeError(
            f"Bernini-R requires MLX runtime (got {backend!r}). "
            "Install mlx-community weights on Apple Silicon."
        )
    if not bundle_root.is_dir():
        raise RuntimeError(f"Bernini-R bundle directory not found: {bundle_root}")
    validate_bernini_bundle(bundle_root)
    return BerniniMlxGenerator(
        ctx,
        bundle_root,
        config=config or BerniniConfig(),
        entry=entry,
        version_key=version_key,
    )


def validate_bernini_bundle(bundle_root: Path) -> None:
    required = ("config.json", "model.safetensors", "vae.safetensors", "t5_encoder.safetensors")
    missing = [name for name in required if not (bundle_root / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Bernini-R bundle at {bundle_root} is missing: {missing}. "
            "Install from ModelScope mlx-community/Bernini-R-1.3B-bf16."
        )


def validate_video_generation_params(
    *,
    entry: Any,
    config: Any,
    step_distill: bool,
) -> None:
    _ = entry, config, step_distill
