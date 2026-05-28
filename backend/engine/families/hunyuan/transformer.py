"""HunyuanVideo-1.5 transformer — 对外入口（MLX / CUDA dispatch）。"""
from __future__ import annotations

from typing import Any

from backend.engine.common._base import TransformerBase


class HunyuanVideoTransformer(TransformerBase):
    """Hunyuan Video DiT — selects MLX or CUDA implementation from ``RuntimeContext``."""

    def __init__(self, config: Any, ctx: Any):
        super().__init__()
        backend = getattr(ctx, "backend", "mlx")
        if backend == "mlx":
            from .transformer_mlx import HunyuanVideoTransformer as _MLX

            self._inner = _MLX(config, ctx)
        elif backend == "cuda":
            from .transformer_cuda import HunyuanVideoTransformerCuda

            self._inner = HunyuanVideoTransformerCuda(config, ctx)
        else:
            raise RuntimeError(f"Unsupported backend for Hunyuan Video: {backend!r}")
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
        if hasattr(self._inner, "after_load_weights"):
            self._inner.after_load_weights(bundle_root)

    def _build_param_map(self):
        if hasattr(self._inner, "_build_param_map"):
            self._inner._build_param_map()
            self._param_map = getattr(self._inner, "_param_map", {})

    def combine_cfg_noise(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.combine_cfg_noise(*args, **kwargs)

    def refine_cfg_noise(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.refine_cfg_noise(*args, **kwargs)

    def prepare_conditioning(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.prepare_conditioning(*args, **kwargs)

    def before_denoise(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.before_denoise(*args, **kwargs)


__all__ = ["HunyuanVideoTransformer"]
