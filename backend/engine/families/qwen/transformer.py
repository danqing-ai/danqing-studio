"""Qwen-Image DiT — 对外入口（MLX 实现见 ``transformer_mlx``；CUDA 尚未接入）。"""
from __future__ import annotations

from typing import Any

from backend.engine.common._base import TransformerBase


class QwenImageTransformer(TransformerBase):
    """Qwen-Image DiT — MLX-only today; fail loud on CUDA until ``transformer_cuda`` exists."""

    def __init__(self, config: Any, ctx: Any):
        super().__init__()
        backend = getattr(ctx, "backend", "mlx")
        if backend != "mlx":
            raise RuntimeError(
                "Qwen-Image DiT has no CUDA implementation in this build; "
                "use MLX runtime or add transformer_cuda."
            )
        from .transformer_mlx import QwenImageTransformer as _MLX

        self._inner = _MLX(config, ctx)
        self.ctx = self._inner.ctx
        self.config = self._inner.config
        self._param_map = self._inner._param_map
        self.dit = self._inner.dit

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.forward(*args, **kwargs)

    def parameters(self):
        return self._inner.parameters()

    def load_weights(self, *args: Any, **kwargs: Any):
        return self._inner.load_weights(*args, **kwargs)

    def after_load_weights(self, bundle_root: str | None = None) -> None:
        self._inner.after_load_weights(bundle_root)
