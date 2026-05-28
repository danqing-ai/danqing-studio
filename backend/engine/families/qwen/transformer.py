"""Qwen-Image DiT — 对外入口（MLX / CUDA dispatch）。"""
from __future__ import annotations

from typing import Any

from backend.engine.common._base import TransformerBase


class QwenImageTransformer(TransformerBase):
    """Qwen-Image DiT — selects MLX or CUDA implementation from ``RuntimeContext``."""

    def __init__(self, config: Any, ctx: Any):
        super().__init__()
        backend = getattr(ctx, "backend", "mlx")
        if backend == "mlx":
            from .transformer_mlx import QwenImageTransformer as _MLX

            self._inner = _MLX(config, ctx)
        elif backend == "cuda":
            from .transformer_cuda import QwenImageTransformerCuda

            self._inner = QwenImageTransformerCuda(config, ctx)
        else:
            raise RuntimeError(f"Unsupported backend for Qwen-Image: {backend!r}")
        self.ctx = self._inner.ctx
        self.config = self._inner.config
        self._param_map = getattr(self._inner, "_param_map", {})

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.forward(*args, **kwargs)

    def parameters(self):
        return self._inner.parameters()

    def load_weights(self, *args: Any, **kwargs: Any):
        return self._inner.load_weights(*args, **kwargs)

    def after_load_weights(self, bundle_root: str | None = None) -> None:
        self._inner.after_load_weights(bundle_root)

    @property
    def dit(self):
        return getattr(self._inner, "dit", None)

    def combine_cfg_noise(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.combine_cfg_noise(*args, **kwargs)

    def refine_cfg_noise(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.refine_cfg_noise(*args, **kwargs)
