"""
ACE-Step Transformer — 对外入口；MLX / CUDA 实现分别见 ``transformer_mlx`` / ``transformer_cuda``。
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

from backend.engine.common.model.base import TransformerBase


class AceStepTransformer(TransformerBase):
    """ACE-Step DiT decoder — dual-backend dispatcher via RuntimeContext.

    Constructed by ``AudioPipeline`` / ``AudioSession`` with the current ``RuntimeContext``
    so that the appropriate MLX or CUDA implementation is selected.
    """

    def __init__(self, ctx: Any, **config: Any):
        super().__init__()
        self._ctx = ctx
        backend = getattr(ctx, "backend", "mlx")

        if backend == "mlx":
            from .transformer_mlx import AceStepDiTMLX
            self._model = AceStepDiTMLX(**config)
        elif backend == "cuda":
            from .transformer_cuda import AceStepDiTCuda
            self._model = AceStepDiTCuda(**config)
        else:
            raise RuntimeError(f"Unsupported backend: {backend}")

        self._backend = backend
        self._build_param_map()

    # ------------------------------------------------------------------
    # TransformerBase interface
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: Any,
        timestep: Any,
        timestep_r: Optional[Any] = None,
        encoder_hidden_states: Optional[Any] = None,
        context_latents: Optional[Any] = None,
        cache: Any = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        if timestep_r is None:
            timestep_r = timestep
        return self._model(
            hidden_states=hidden_states,
            timestep=timestep,
            timestep_r=timestep_r,
            encoder_hidden_states=encoder_hidden_states,
            context_latents=context_latents,
            cache=cache,
            use_cache=use_cache,
        )

    def parameters(self):
        return self._model.parameters()

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        """Strip ``decoder.`` prefix from checkpoint keys."""
        remapped: dict[str, Any] = {}
        for key, tensor in weights.items():
            nk = key[len("decoder."):] if key.startswith("decoder.") else key
            remapped[nk] = tensor
        return remapped

    def _build_param_map(self):
        from backend.engine.common.codecs.vae.decoder import _collect_nn_params

        self._param_map = {}
        _collect_nn_params(self._model, "", self._param_map)
