"""Z-Image Fun ControlNet Union — CUDA placeholder (structural guide is MLX-only)."""

from __future__ import annotations

_MSG = (
    "Z-Image ControlNet Union (structural_guide) is MLX-only today. "
    "CUDA support is planned in a unified batch with flux1 structural CUDA."
)


def assert_z_image_control_mlx() -> None:
    raise RuntimeError(_MSG)
