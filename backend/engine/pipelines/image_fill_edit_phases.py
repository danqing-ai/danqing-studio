"""FLUX Fill retouch/extend — phased session helpers."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from backend.core.contracts import ExecutionContext, ImageEditRequest
from backend.engine.contracts.runtime_contracts import FamilyRuntimeContract
from backend.engine.sessions._context import MediaRunContext, ResolvedRun

PhaseCmFactory = Callable[[str], AbstractContextManager[Any]]


@dataclass
class ImageFillEditRunContext(MediaRunContext):
    """State for Fill retouch/extend (encode → schedule → denoise → persist)."""

    pipeline: Any
    request: ImageEditRequest
    exec_ctx: ExecutionContext
    entry: Any
    config: Any
    runtime_contract: FamilyRuntimeContract
    family: str
    model_key: str
    version_key: str | None
    bundle_root: Path | None
    model: Any
    extra_cond: dict[str, Any]
    txt_embeds: Any
    neg_embeds: Any
    txt_attn_mask: Any
    neg_attn_mask: Any
    pooled_embeds: Any
    neg_pooled_embeds: Any
    encoder_type: str
    scheduler: Any
    timesteps: list[Any]
    sigmas: Any
    sched_ts: Any
    timestep_embed_schedule: list[float] | None
    semantics: Any
    latents: Any
    w: int
    h: int
    lh: int
    lw: int
    seed: int
    steps: int
    guidance: float
    flux_unpack: Callable[..., Any]
    flux_pack: Callable[..., Any]
    preview_mode: str
    preview_interval: int
    preview_max_edge: int
    preview_state: dict[str, Any]
    on_progress: Callable | None = None
    on_log: Callable | None = None

    def session_infer(self, **_ignored: Any) -> Any | None:
        return execute_image_fill_edit_denoise(self)


def build_image_fill_edit_context(
    pipeline: Any,
    request: ImageEditRequest,
    ctx_exec: ExecutionContext,
    *,
    resolved: ResolvedRun,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
    phase_cm: PhaseCmFactory | None = None,
) -> ImageFillEditRunContext | None:
    """Prepare Fill edit through encode + schedule (flux-fill-controlnet today)."""
    if request.operation not in ("retouch", "extend"):
        raise RuntimeError(
            f"build_image_fill_edit_context requires retouch|extend, got {request.operation!r}"
        )
    from backend.engine.families.flux1.fill_edit import build_flux1_fill_edit_context

    return build_flux1_fill_edit_context(
        pipeline,
        request,
        ctx_exec,
        resolved=resolved,
        on_progress=on_progress,
        on_log=on_log,
        phase_cm=phase_cm,
    )


def execute_image_fill_edit_denoise(ctx: ImageFillEditRunContext) -> Any | None:
    """Run Fill denoise via ``DiffusionParadigm``."""
    from backend.engine.families.flux1.fill_edit import execute_flux1_fill_edit_denoise

    return execute_flux1_fill_edit_denoise(ctx)


def persist_image_fill_edit(
    ctx: ImageFillEditRunContext,
    latents: Any,
) -> tuple[str, dict[str, Any]] | None:
    """Decode Fill latents and write work_dir asset metadata."""
    from backend.engine.families.flux1.fill_edit import persist_flux1_fill_edit

    return persist_flux1_fill_edit(ctx, latents)


