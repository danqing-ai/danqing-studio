"""DiT dispatch stem — shared backend selection and ``_inner`` delegation (≥2 families)."""
from __future__ import annotations

from typing import Any, Type

from backend.engine.common._base import TransformerBase


def dispatch_dit_implementation(
    config: Any,
    ctx: Any,
    *,
    mlx_cls: Type[Any],
    cuda_cls: Type[Any] | None = None,
    unavailable_product: str | None = None,
    **factory_kwargs: Any,
) -> Any:
    """Instantiate MLX or CUDA DiT; fail loud when CUDA is requested but not provided."""
    backend = getattr(ctx, "backend", "mlx")
    if backend == "mlx":
        return mlx_cls(config, ctx, **factory_kwargs)
    if backend == "cuda":
        if cuda_cls is not None:
            return cuda_cls(config, ctx, **factory_kwargs)
        if unavailable_product is None:
            raise RuntimeError(f"CUDA DiT is not implemented for backend={backend!r}")
        from backend.engine.common.dit_cuda_unavailable import raise_cuda_dit_unavailable

        raise_cuda_dit_unavailable(unavailable_product)
    raise RuntimeError(f"Unsupported DiT backend: {backend!r}")


class DelegatingDiTStem(TransformerBase):
    """Thin stem: ``_inner`` holds MLX/CUDA implementation; forwards unknown attrs."""

    _inner: Any

    def __init__(
        self,
        config: Any,
        ctx: Any,
        *,
        mlx_cls: Type[Any],
        cuda_cls: Type[Any] | None = None,
        unavailable_product: str | None = None,
        **factory_kwargs: Any,
    ) -> None:
        super().__init__()
        self._inner = dispatch_dit_implementation(
            config,
            ctx,
            mlx_cls=mlx_cls,
            cuda_cls=cuda_cls,
            unavailable_product=unavailable_product,
            **factory_kwargs,
        )
        self.ctx = self._inner.ctx
        self.config = self._inner.config
        self._param_map = getattr(self._inner, "_param_map", {})

    def __getattr__(self, name: str) -> Any:
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.forward(*args, **kwargs)

    def parameters(self):
        return self._inner.parameters()

    def load_weights(self, *args: Any, **kwargs: Any) -> Any:
        out = self._inner.load_weights(*args, **kwargs)
        self._param_map = getattr(self._inner, "_param_map", {})
        return out

    def after_load_weights(self, bundle_root: str | None = None) -> None:
        if hasattr(self._inner, "after_load_weights"):
            self._inner.after_load_weights(bundle_root)

    def sanitize(self, weights: dict) -> dict:
        if hasattr(self._inner, "sanitize"):
            return self._inner.sanitize(weights)
        return weights

    def combine_cfg_noise(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.combine_cfg_noise(*args, **kwargs)

    def refine_cfg_noise(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.refine_cfg_noise(*args, **kwargs)

    def _build_param_map(self) -> None:
        if hasattr(self._inner, "_build_param_map"):
            self._inner._build_param_map()
            self._param_map = getattr(self._inner, "_param_map", {})
