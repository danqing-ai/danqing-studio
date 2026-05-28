"""Z-Image Transformer — 对外入口（MLX / CUDA dispatch）。"""
from __future__ import annotations

from typing import Any

from backend.engine.common._base import TransformerBase


class ZImageTransformer(TransformerBase):
    """Z-Image DiT — MLX 或 CUDA ``RuntimeContext`` 实现（见 ``transformer_mlx``）。"""

    def __init__(self, config: Any, ctx: Any):
        super().__init__()
        backend = getattr(ctx, "backend", "mlx")
        if backend == "mlx":
            from .transformer_mlx import ZImageTransformer as _Impl
        elif backend == "cuda":
            from .transformer_cuda import ZImageTransformerCuda as _Impl
        else:
            raise RuntimeError(f"Unsupported backend for Z-Image: {backend!r}")
        self._inner = _Impl(config, ctx)
        self.config = self._inner.config
        self.ctx = self._inner.ctx
        self._param_map = self._inner._param_map

    def __getattr__(self, name: str) -> Any:
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.forward(*args, **kwargs)

    def load_weights(self, *args: Any, **kwargs: Any):
        out = self._inner.load_weights(*args, **kwargs)
        self._param_map = self._inner._param_map
        return out

    def combine_cfg_noise(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.combine_cfg_noise(*args, **kwargs)

    def refine_cfg_noise(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.refine_cfg_noise(*args, **kwargs)

    def forward_cfg(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.forward_cfg(*args, **kwargs)

    def before_denoise(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.before_denoise(*args, **kwargs)

    def after_load_weights(self, bundle_root: str | None = None) -> None:
        self._inner.after_load_weights(bundle_root)
