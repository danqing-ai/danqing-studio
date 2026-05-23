"""
MusicPipeline — audio request → family generator → WAV → asset save.

Dispatches by registry ``family`` (ACE-Step, HeartMuLa, …) on MLX or CUDA.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List

import numpy as np
import soundfile as sf

from backend.core.contracts import (
    AudioGenerationRequest,
    EngineResult,
    ExecutionContext,
    LogEvent,
    ProgressEvent,
    parse_model_version,
)
from backend.engine.common.cache import ModelCache
from backend.engine.common.pipeline_registry import (
    local_bundle_root as _local_bundle_root_fn,
    registry_scalar_default as _registry_scalar_default_fn,
    resolve_project_path as _resolve_project_path_fn,
    resolve_version_block as _resolve_version_block_fn,
)
from backend.engine.config.model_configs import (
    AceStepConfig,
    HeartMulaConfig,
    get_config_class,
)
from backend.engine.families.ace_step.generation import (
    AceStepLyricsCapture,
    create_ace_step_generator,
    lyrics_capture_log_message,
    lyrics_capture_metadata,
    prepare_music_request,
    write_lyrics_sidecar,
)
from backend.engine.families.heartmula.generation import (
    FRAME_RATE as HEARTMULA_FRAME_RATE,
    SAMPLE_RATE as HEARTMULA_SAMPLE_RATE,
    create_heartmula_generator,
    prepare_heartmula_request,
)
from backend.engine.runtime._base import RuntimeContext

logger = logging.getLogger(__name__)

_ACE_STEP_SAMPLE_RATE = 48_000


class MusicPipeline:
    """Audio generation — registry ``family`` on MLX or CUDA."""

    def __init__(
        self,
        ctx: RuntimeContext,
        model_registry: Any,
        asset_store: Any,
        model_cache: ModelCache | None = None,
        project_root: Path | None = None,
    ):
        self.ctx = ctx
        self._registry = model_registry
        self._asset_store = asset_store
        self._cache = model_cache
        self._project_root = project_root or Path.cwd()
        self._generator: Any = None
        self._generator_key: str | None = None

    def _resolve_path(self, local_path: str) -> Path:
        return _resolve_project_path_fn(self._project_root, local_path)

    @staticmethod
    def _registry_scalar_default(entry, key: str, fallback):
        return _registry_scalar_default_fn(entry, key, fallback)

    def _resolve_version_block(self, entry, version_key: str | None) -> dict | None:
        return _resolve_version_block_fn(entry, version_key)

    def _local_bundle_root(self, entry, version_key: str | None) -> Path | None:
        return _local_bundle_root_fn(self._project_root, entry, version_key)

    def _get_generator(self, family: str, bundle_root: Path):
        key = f"{family}:{bundle_root}"
        if self._generator is None or self._generator_key != key:
            if family == "ace_step":
                gen = create_ace_step_generator(self.ctx, bundle_root)
            elif family == "heartmula":
                gen = create_heartmula_generator(self.ctx, bundle_root)
            else:
                raise RuntimeError(f"MusicPipeline: unsupported audio family {family!r}")
            gen.load()
            self._generator = gen
            self._generator_key = key
        return self._generator

    @staticmethod
    def _raise_if_cancelled(exec_ctx: ExecutionContext) -> None:
        if exec_ctx.cancel_token.is_cancelled():
            raise asyncio.CancelledError()

    def run(
        self,
        request: AudioGenerationRequest,
        exec_ctx: ExecutionContext,
    ) -> EngineResult:
        self._raise_if_cancelled(exec_ctx)
        t0 = time.monotonic()
        model_id, version_key = parse_model_version(request.model)
        entry = self._registry.get(model_id)
        if entry is None:
            raise RuntimeError(f"Model {model_id!r} not found in registry")

        bundle_root = self._local_bundle_root(entry, version_key)
        if bundle_root is None or not bundle_root.is_dir():
            raise RuntimeError(f"Model bundle not found for {request.model!r}")

        family = entry.family
        cfg_cls = get_config_class(family)
        if family == "ace_step":
            return self._run_ace_step(
                request, exec_ctx, model_id, entry, bundle_root, cfg_cls(), t0
            )
        if family == "heartmula":
            return self._run_heartmula(
                request, exec_ctx, model_id, entry, bundle_root, cfg_cls(), t0
            )
        raise RuntimeError(f"MusicPipeline: unknown audio family {family!r}")

    def _run_heartmula(
        self,
        request: AudioGenerationRequest,
        exec_ctx: ExecutionContext,
        model_id: str,
        entry: Any,
        bundle_root: Path,
        config: HeartMulaConfig,
        t0: float,
    ) -> EngineResult:
        prepared = prepare_heartmula_request(request, config)
        for level, message in prepared.log_events:
            exec_ctx.on_log(LogEvent(level=level, message=message))

        exec_ctx.on_log(
            LogEvent(
                level="info",
                message=f"Loading HeartMuLa: {model_id} from {bundle_root}",
            )
        )
        generator = self._get_generator("heartmula", bundle_root)
        seed = (
            request.seed
            if request.seed is not None and request.seed >= 0
            else random.randint(0, 2**31 - 1)
        )
        n = max(request.n, 1)
        exec_ctx.on_log(
            LogEvent(
                level="info",
                message=(
                    f"Generating {n} track(s): duration={prepared.duration}s, "
                    f"cfg={prepared.cfg_scale}, temperature={prepared.temperature}, "
                    f"topk={prepared.topk}, seed={seed}"
                ),
            )
        )
        if request.negative_prompt:
            exec_ctx.on_log(
                LogEvent(
                    level="warning",
                    message="negative_prompt is not used by HeartMuLa (ignored)",
                )
            )

        output_paths: List[str] = []
        output_durations: List[float] = []
        for i in range(n):
            self._raise_if_cancelled(exec_ctx)
            batch_seed = seed + i
            exec_ctx.on_progress(
                ProgressEvent(
                    progress=(i + 0.1) / n,
                    step=i + 1,
                    total=n,
                    message=f"Generating music {i + 1}/{n}",
                )
            )
            t_gen = time.monotonic()
            max_lm_frames = max(1, int(prepared.duration * HEARTMULA_FRAME_RATE))

            def _lm_progress(done: int, total: int) -> None:
                self._raise_if_cancelled(exec_ctx)
                frac = done / total if total else 0.0
                exec_ctx.on_progress(
                    ProgressEvent(
                        progress=(i + 0.12 + 0.76 * frac) / n,
                        step=i + 1,
                        total=n,
                        message=(
                            f"HeartMuLa LM {done}/{total} 帧 "
                            f"(~{done / HEARTMULA_FRAME_RATE:.0f}s / "
                            f"{prepared.duration:.0f}s)"
                        ),
                    )
                )
                if done <= 1 or done % 10 == 0 or done >= total:
                    exec_ctx.on_log(
                        LogEvent(
                            level="info",
                            message=(
                                f"HeartMuLa LM 进度 {done}/{total} 帧 "
                                f"(~{done / HEARTMULA_FRAME_RATE:.1f}s 音频)"
                            ),
                        )
                    )

            exec_ctx.on_log(
                LogEvent(
                    level="info",
                    message=(
                        f"HeartMuLa 开始逐帧生成（最多 {max_lm_frames} 帧，"
                        f"约 {prepared.duration:.0f}s；遇 Audio EOS 会提前结束）"
                    ),
                )
            )
            try:
                waveform = generator.generate_waveform(
                    tags=prepared.tags,
                    lyrics=prepared.lyrics,
                    duration=prepared.duration,
                    temperature=prepared.temperature,
                    topk=prepared.topk,
                    cfg_scale=prepared.cfg_scale,
                    codec_steps=prepared.codec_steps,
                    codec_guidance=prepared.codec_guidance,
                    seed=batch_seed,
                    progress_callback=_lm_progress,
                )
            except Exception as exc:
                logger.exception(
                    "HeartMuLa generation failed (%d/%d, seed=%s)",
                    i + 1,
                    n,
                    batch_seed,
                )
                exec_ctx.on_log(
                    LogEvent(
                        level="error",
                        message=f"HeartMuLa failed ({i + 1}/{n}): {exc}",
                    )
                )
                raise
            self._raise_if_cancelled(exec_ctx)

            gen_s = time.monotonic() - t_gen
            frames = getattr(generator, "last_frame_count", 0)
            eos_early = getattr(generator, "last_eos_early", False)
            exec_ctx.on_log(
                LogEvent(
                    level="info",
                    message=(
                        f"HeartMuLa done ({i + 1}/{n}): {gen_s:.1f}s, "
                        f"frames={frames}, eos_early={eos_early}"
                    ),
                )
            )
            if eos_early:
                exec_ctx.on_log(
                    LogEvent(
                        level="info",
                        message="模型提前结束（Audio EOS）；实际时长可能短于请求时长",
                    )
                )

            exec_ctx.on_progress(
                ProgressEvent(
                    progress=(i + 0.9) / n,
                    step=i + 1,
                    total=n,
                    message=f"Saving audio {i + 1}/{n}",
                )
            )
            out_path = self._save_audio(
                waveform,
                model_id,
                batch_seed,
                family="heartmula",
                sample_rate=HEARTMULA_SAMPLE_RATE,
            )
            n_samples = int(waveform.shape[0]) if hasattr(waveform, "shape") else 0
            dur_written = n_samples / HEARTMULA_SAMPLE_RATE if n_samples else 0.0
            exec_ctx.on_log(
                LogEvent(
                    level="success",
                    message=(
                        f"Saved {i + 1}/{n}: {out_path.name} "
                        f"({dur_written:.1f}s, {out_path.stat().st_size // 1024}KB)"
                    ),
                )
            )
            output_paths.append(str(out_path))
            output_durations.append(dur_written)

        self._raise_if_cancelled(exec_ctx)
        exec_ctx.on_progress(
            ProgressEvent(progress=1.0, step=n, total=n, message="Complete")
        )
        elapsed = time.monotonic() - t0
        asset_ids = self._persist_assets(
            output_paths,
            request,
            model_id,
            elapsed,
            exec_ctx.task_id,
            output_durations,
        )
        return EngineResult(
            primary_asset_id=asset_ids[0] if asset_ids else "",
            asset_ids=asset_ids,
            output_paths=output_paths,
            metadata={
                "model": model_id,
                "seed": seed,
                "duration_seconds": prepared.duration,
                "cfg_scale": prepared.cfg_scale,
                "temperature": prepared.temperature,
            },
        )

    def _run_ace_step(
        self,
        request: AudioGenerationRequest,
        exec_ctx: ExecutionContext,
        model_id: str,
        entry: Any,
        bundle_root: Path,
        config: AceStepConfig,
        t0: float,
    ) -> EngineResult:
        prepared = prepare_music_request(request, config, bundle_root)
        for level, message in prepared.log_events:
            exec_ctx.on_log(LogEvent(level=level, message=message))

        exec_ctx.on_log(
            LogEvent(
                level="info",
                message=f"Loading ACE-Step model: {model_id} from {bundle_root}",
            )
        )
        generator = self._get_generator("ace_step", bundle_root)

        steps = prepared.steps
        lyrics = prepared.lyrics
        vocal_lang = prepared.vocal_language
        shift = prepared.shift
        guidance = request.guidance if request.guidance is not None else 3.0
        duration = float(request.duration or 30)
        seed = (
            request.seed
            if request.seed is not None and request.seed >= 0
            else random.randint(0, 2**31 - 1)
        )
        n = max(request.n, 1)

        exec_ctx.on_log(
            LogEvent(
                level="info",
                message=(
                    f"Generating {n} audio(s): duration={duration}s, steps={steps}, "
                    f"guidance={guidance} (turbo ignores CFG), seed={seed}, "
                    f"instrumental={request.instrumental}, vocal_language={vocal_lang}"
                ),
            )
        )
        if request.negative_prompt:
            exec_ctx.on_log(
                LogEvent(
                    level="warning",
                    message="negative_prompt is not used by ACE-Step yet (ignored)",
                )
            )

        output_paths: List[str] = []
        output_durations: List[float] = []
        lyrics_capture: AceStepLyricsCapture | None = None
        for i in range(n):
            self._raise_if_cancelled(exec_ctx)
            batch_seed = seed + i
            exec_ctx.on_progress(
                ProgressEvent(
                    progress=(i + 0.1) / n,
                    step=i + 1,
                    total=n,
                    message=f"Generating audio {i + 1}/{n}",
                )
            )

            exec_ctx.on_log(
                LogEvent(
                    level="info",
                    message=f"Running ACE-Step diffusion ({i + 1}/{n}, seed={batch_seed})...",
                )
            )
            t_gen = time.monotonic()
            try:
                waveform = generator.generate_waveform(
                    prompt=prepared.effective_prompt or request.prompt or "",
                    lyrics=lyrics,
                    vocal_language=vocal_lang,
                    duration=duration,
                    steps=steps,
                    guidance=guidance,
                    seed=batch_seed,
                    bpm=request.bpm,
                    key_scale=request.key_scale or "",
                    time_signature=request.time_signature or "",
                    shift=shift,
                )
            except Exception as exc:
                logger.exception(
                    "ACE-Step generation failed (item %d/%d, seed=%s)",
                    i + 1,
                    n,
                    batch_seed,
                )
                exec_ctx.on_log(
                    LogEvent(
                        level="error",
                        message=f"ACE-Step diffusion failed ({i + 1}/{n}): {exc}",
                    )
                )
                raise
            self._raise_if_cancelled(exec_ctx)

            gen_s = time.monotonic() - t_gen
            latent_frames = getattr(generator, "last_latent_frames", 0)
            hum_ratio = getattr(generator, "last_hum_ratio", 0.0)
            mains_acf = getattr(generator, "last_mains_acf", 0.0)
            decode_mode = getattr(generator, "last_decode_mode", "")
            latent_cos = getattr(generator, "last_latent_cos", 0.0)
            latent_diff = getattr(generator, "last_latent_diff_mean", 0.0)
            lm_expanded = getattr(generator, "last_lm_expanded", False)
            cap = getattr(generator, "last_lyrics_capture", None)
            if cap is not None:
                lyrics_capture = cap
                if i == 0:
                    log_msg = lyrics_capture_log_message(cap)
                    if log_msg:
                        exec_ctx.on_log(LogEvent(level="info", message=log_msg))
            nominal_frames = max(128, int(round(duration * 48_000 / 1920)))
            if latent_frames == nominal_frames - 1 and nominal_frames % 2 == 1:
                exec_ctx.on_log(
                    LogEvent(
                        level="info",
                        message=(
                            f"ACE-Step latent length {nominal_frames}→{latent_frames} "
                            "(even frame count avoids turbo DiT collapse on long clips)"
                        ),
                    )
                )
            exec_ctx.on_log(
                LogEvent(
                    level="info",
                    message=(
                        f"ACE-Step inference done ({i + 1}/{n}): {gen_s:.1f}s, "
                        f"latent_frames={latent_frames}, decode={decode_mode}, "
                        f"lm_expanded={lm_expanded}, latent_cos={latent_cos:.4f}, "
                        f"latent_diff={latent_diff:.4f}, mains_acf={mains_acf:.3f}"
                    ),
                )
            )
            if mains_acf > 0.4 or hum_ratio > 0.25:
                exec_ctx.on_log(
                    LogEvent(
                        level="warning",
                        message=(
                            f"Audio {i + 1}/{n} may contain mains hum "
                            f"(mains_acf={mains_acf:.3f}); try another seed or a descriptive prompt"
                        ),
                    )
                )

            exec_ctx.on_progress(
                ProgressEvent(
                    progress=(i + 0.9) / n,
                    step=i + 1,
                    total=n,
                    message=f"Saving audio {i + 1}/{n}",
                )
            )

            out_path = self._save_audio(
                waveform,
                model_id,
                batch_seed,
                family="ace_step",
                sample_rate=_ACE_STEP_SAMPLE_RATE,
            )
            if lyrics_capture is not None:
                sidecar = write_lyrics_sidecar(out_path, lyrics_capture.lyrics_effective)
                if sidecar is not None:
                    exec_ctx.on_log(
                        LogEvent(
                            level="info",
                            message=f"歌词已写入: {sidecar.name}",
                        )
                    )
            n_samples = int(waveform.shape[0]) if hasattr(waveform, "shape") else 0
            dur_written = n_samples / _ACE_STEP_SAMPLE_RATE if n_samples else 0.0
            exec_ctx.on_log(
                LogEvent(
                    level="success",
                    message=(
                        f"Saved audio {i + 1}/{n}: {out_path.name} "
                        f"({dur_written:.1f}s, {out_path.stat().st_size // 1024}KB)"
                    ),
                )
            )
            output_paths.append(str(out_path))
            output_durations.append(dur_written)

        self._raise_if_cancelled(exec_ctx)
        exec_ctx.on_progress(
            ProgressEvent(progress=1.0, step=n, total=n, message="Complete")
        )
        exec_ctx.on_log(
            LogEvent(
                level="success",
                message=f"ACE-Step complete: {len(output_paths)} file(s) in {time.monotonic() - t0:.1f}s",
            )
        )
        elapsed = time.monotonic() - t0
        asset_ids = self._persist_assets(
            output_paths,
            request,
            model_id,
            elapsed,
            exec_ctx.task_id,
            output_durations,
            lyrics_capture=lyrics_capture,
        )

        result_meta: dict[str, Any] = {
            "model": model_id,
            "seed": seed,
            "steps": steps,
            "guidance": guidance,
            "duration_seconds": duration,
        }
        if lyrics_capture is not None:
            result_meta.update(lyrics_capture_metadata(lyrics_capture))

        return EngineResult(
            primary_asset_id=asset_ids[0] if asset_ids else "",
            asset_ids=asset_ids,
            output_paths=output_paths,
            metadata=result_meta,
        )

    def _save_audio(
        self,
        waveform: Any,
        model_id: str,
        seed: int,
        *,
        family: str,
        sample_rate: int,
    ) -> Path:
        out_dir = self._project_root / "outputs" / "audio"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_id = model_id.replace("/", "_")
        fname = f"{family}_{safe_id}_{ts}_{seed}.wav"
        out_path = out_dir / fname

        wf = np.array(waveform) if not isinstance(waveform, np.ndarray) else waveform
        if wf.ndim == 3:
            wf = wf[0]
        if wf.ndim == 1:
            wf = wf[:, None]

        sf.write(str(out_path), wf, sample_rate)
        return out_path

    def _persist_assets(
        self,
        paths: List[str],
        request: AudioGenerationRequest,
        model_id: str,
        elapsed: float,
        task_id: str,
        durations: List[float] | None = None,
        lyrics_capture: AceStepLyricsCapture | None = None,
    ) -> List[str]:
        ids = []
        fmt = (request.audio_format or "wav").lower()
        mime = "audio/mpeg" if fmt == "mp3" else f"audio/{fmt}"
        for idx, p in enumerate(paths):
            dur = None
            if durations and idx < len(durations):
                dur = durations[idx]
            asset_meta: dict[str, Any] = {
                "model": model_id,
                "prompt": request.prompt,
                "duration_seconds": dur if dur is not None else request.duration,
                "format": fmt,
                "elapsed_seconds": elapsed,
                "output_path": str(p),
            }
            if lyrics_capture is not None:
                asset_meta.update(lyrics_capture_metadata(lyrics_capture))
                sidecar = Path(p).with_name(f"{Path(p).stem}_lyrics.txt")
                if sidecar.is_file():
                    asset_meta["lyrics_sidecar"] = str(sidecar)
            aid = self._asset_store.create_from_file(
                Path(p),
                kind="audio",
                mime_type=mime,
                source_task_id=task_id,
                metadata=asset_meta,
                source_action="create",
            )
            ids.append(aid)
        return ids
