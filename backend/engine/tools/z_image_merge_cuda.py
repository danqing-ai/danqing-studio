"""Z-Image DiT merge — CUDA placeholder (MLX-only operation)."""

from __future__ import annotations

_MSG = (
    "Z-Image DiT weight merge is MLX-only (Apple Silicon). "
    "CUDA batch merge is not implemented."
)


def assert_z_image_merge_mlx() -> None:
    raise RuntimeError(_MSG)
