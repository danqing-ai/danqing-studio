"""ControlNet / structural guide / FLUX Fill — runtime contract (MLX today; CUDA unified batch TBD).

Flip ``CONTROLNET_DECLARED_BACKENDS`` and implement ``families/flux1/transformer_cuda.py``
when the CUDA batch lands; pipeline hooks call :func:`require_controlnet_runtime`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

# Registry ``backends`` for controlnets category rows until CUDA batch is ready.
CONTROLNET_DECLARED_BACKENDS: tuple[str, ...] = ("mlx",)
CONTROLNET_CUDA_BATCH_PLANNED = True

if TYPE_CHECKING:
    from backend.engine.runtime._base import RuntimeContext


def controlnet_host_backends_available() -> tuple[str, ...]:
    from backend.engine.platform import PlatformInfo

    detected = set(PlatformInfo.detect())
    return tuple(b for b in CONTROLNET_DECLARED_BACKENDS if b in detected)


def controlnet_runtime_available() -> bool:
    return len(controlnet_host_backends_available()) > 0


def require_controlnet_runtime(ctx: RuntimeContext, *, feature: str) -> None:
    """Fail loud when this host or RuntimeContext cannot run ControlNet paths today."""
    from backend.engine.platform import PlatformInfo
    from backend.engine.runtime.mlx import MLXContext

    detected = PlatformInfo.detect()
    if not controlnet_runtime_available():
        raise RuntimeError(
            f"FLUX ControlNet ({feature}) requires host backends {CONTROLNET_DECLARED_BACKENDS}; "
            f"detected={detected}. "
            "CUDA support is planned in a unified engine batch "
            "(see backend/engine/common/controlnet_runtime.py)."
        )
    if not isinstance(ctx, MLXContext):
        raise RuntimeError(
            f"FLUX ControlNet ({feature}) is MLX-only until the unified CUDA batch; "
            f"current runtime={type(ctx).__name__}. "
            "Placeholder: backend/engine/families/flux1/transformer_cuda.py + CudaContext paths."
        )
