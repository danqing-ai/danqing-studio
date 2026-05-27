"""MLX cache release hook for memory_policy."""
from __future__ import annotations


def clear_mlx_cache() -> None:
    import mlx.core as mx

    mx.clear_cache()
