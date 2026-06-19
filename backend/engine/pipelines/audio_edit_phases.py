"""Audio edit phased helpers (``AudioSession``)."""

from __future__ import annotations

import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List

from backend.core.contracts import (
    AudioEditRequest,
    EngineResult,
    ExecutionContext,
    LogEvent,
    ProgressEvent,
)
from backend.engine._transformer_registry import get_audio_edit_handler
from backend.engine.config.model_configs import get_config_class
from backend.engine.families._audio_backbone import plugin_audio_generator_if_ready
from backend.engine.inference.audio_edit import run_audio_edit_handler
from backend.engine.pipelines.audio_run_common import load_audio_generator_for_request
from backend.engine.pipelines.audio_persist import (
    ACE_STEP_SAMPLE_RATE,
    persist_audio_edit_assets,
    quality_log_message,
    raise_if_cancelled,
    save_audio_waveform,
)
from backend.engine.protocols.plugin import FamilyPlugin, ParadigmKind
from backend.engine.sessions._context import MediaRunContext, ResolvedRun, require_resolved_bundle

PhaseCmFactory = Callable[[str], AbstractContextManager[Any]]


@dataclass
class AudioEditRunContext(MediaRunContext):
    """State for one audio edit run (prepare → infer → persist)."""

    pipeline: Any
    request: AudioEditRequest
    exec_ctx: ExecutionContext
    model_id: str
    version_key: str | None
    entry: Any
    bundle_root: Path
    config: Any
    family: str
    generator: Any
    handler: Callable[..., Any]
    src_path: Path
    paradigm: ParadigmKind
    t0: float

    def session_infer(self, **_ignored: Any) -> tuple[list[str], list[float], Any, Any]:
        return execute_audio_edit_infer(self)


def build_audio_edit_run_context(
    pipeline: Any,
    request: AudioEditRequest,
    exec_ctx: ExecutionContext,
    *,
    resolved: ResolvedRun,
    t0: float,
    phase_cm: PhaseCmFactory | None = None,
    plugin: FamilyPlugin | None = None,
    paradigm: ParadigmKind = "flow_matching",
) -> AudioEditRunContext:
    phase_cm = phase_cm or (lambda _name: nullcontext())
    bundle_root = require_resolved_bundle(resolved)
    entry = resolved.registry_entry
    model_id = resolved.model_id
    version_key = resolved.version_key
    family = resolved.family_id
    handler = get_audio_edit_handler(family, request.operation)
    config = get_config_class(family)()

    src_path = exec_ctx.asset_store.get_file_path(request.source_asset_id)
    if not src_path.is_file():
        raise RuntimeError(f"Source audio asset not found: {request.source_asset_id!r}")

    with phase_cm("prepare"):
        generator = plugin_audio_generator_if_ready(plugin)
        if generator is None:

            def _on_log(level: str, message: str) -> None:
                exec_ctx.on_log(LogEvent(level=level, message=message))

            generator = load_audio_generator_for_request(
                pipeline,
                family=family,
                bundle_root=bundle_root,
                entry=entry,
                version_key=version_key,
                request=request,
                on_log=_on_log,
            )

    return AudioEditRunContext(
        pipeline=pipeline,
        request=request,
        exec_ctx=exec_ctx,
        model_id=model_id,
        version_key=version_key,
        entry=entry,
        bundle_root=bundle_root,
        config=config,
        family=family,
        generator=generator,
        handler=handler,
        src_path=src_path,
        paradigm=paradigm,
        t0=t0,
    )


def execute_audio_edit_infer(
    ctx: AudioEditRunContext,
) -> tuple[list[str], list[float], Any, Any]:
    """Run registry edit handler and write waveforms to disk."""
    pipeline = ctx.pipeline
    exec_ctx = ctx.exec_ctx

    def raise_if_cancelled() -> None:
        raise_if_cancelled(exec_ctx)

    def on_progress(progress: float, step: int, total: int, message: str) -> None:
        exec_ctx.on_progress(
            ProgressEvent(progress=progress, step=step, total=total, message=message)
        )

    def on_log(level: str, message: str) -> None:
        exec_ctx.on_log(LogEvent(level=level, message=message))

    batch = run_audio_edit_handler(
        handler=ctx.handler,
        paradigm=ctx.paradigm,
        exec_ctx=exec_ctx,
        generator=ctx.generator,
        request=ctx.request,
        config=ctx.config,
        bundle_root=ctx.bundle_root,
        source_path=ctx.src_path,
        raise_if_cancelled=raise_if_cancelled,
        on_progress=on_progress,
        on_log=on_log,
    )

    output_paths: List[str] = []
    output_durations: List[float] = []
    quality = batch.quality
    for i, waveform in enumerate(batch.waveforms):
        q_msg = quality_log_message(quality) if quality is not None else None
        if q_msg:
            exec_ctx.on_log(LogEvent(level="info", message=q_msg))
        batch_seed = batch.seed + i
        out_path = save_audio_waveform(
            ctx.pipeline._project_root,
            waveform,
            ctx.model_id,
            batch_seed,
            family=f"{ctx.family}_{ctx.request.operation}",
            sample_rate=ACE_STEP_SAMPLE_RATE,
        )
        n_samples = int(waveform.shape[0]) if hasattr(waveform, "shape") else 0
        dur_written = n_samples / ACE_STEP_SAMPLE_RATE if n_samples else 0.0
        output_paths.append(str(out_path))
        output_durations.append(dur_written)

    return output_paths, output_durations, batch, quality


def persist_audio_edit(
    ctx: AudioEditRunContext,
    output_paths: list[str],
    output_durations: list[float],
    batch: Any,
    quality: Any,
) -> EngineResult:
    elapsed = time.monotonic() - ctx.t0
    asset_ids = persist_audio_edit_assets(
        ctx.pipeline._asset_store,
        output_paths,
        ctx.request,
        ctx.model_id,
        elapsed,
        ctx.exec_ctx.task_id,
        output_durations,
        quality=quality,
    )

    result_meta: dict[str, Any] = {
        "model": ctx.model_id,
        "operation": ctx.request.operation,
        "source_asset_id": ctx.request.source_asset_id,
        "source_fidelity": ctx.request.source_fidelity,
        "seed": batch.seed,
    }
    if batch.lyrics != "[Instrumental]":
        result_meta["lyrics"] = batch.lyrics
        result_meta["vocal_language"] = batch.vocal_lang
    if ctx.request.duration is not None:
        result_meta["duration"] = float(max(10, min(600, int(ctx.request.duration))))
    if ctx.request.bpm is not None:
        result_meta["bpm"] = ctx.request.bpm
    if ctx.request.key_scale:
        result_meta["key_scale"] = ctx.request.key_scale
    if ctx.request.time_signature:
        result_meta["time_signature"] = ctx.request.time_signature
    if quality is not None:
        result_meta.update(quality.as_metadata())

    return EngineResult(
        primary_asset_id=asset_ids[0] if asset_ids else "",
        asset_ids=asset_ids,
        output_paths=output_paths,
        metadata=result_meta,
    )

