"""
MusicPipeline — audio request → ACE-Step → VAE decode → asset save.

Dispatches by runtime: **MLX** (native DiT + VAE on Apple Silicon) or **CUDA** (PyTorch).
"""
from __future__ import annotations

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
from backend.engine.config.model_configs import AceStepConfig, get_config_class
from backend.engine.families.ace_step.generation import (
    AceStepLyricsCapture,
    create_ace_step_generator,
    lyrics_capture_log_message,
    lyrics_capture_metadata,
    prepare_music_request,
    write_lyrics_sidecar,
)
from backend.engine.runtime._base import RuntimeContext

logger = logging.getLogger(__name__)


class MusicPipeline:
    """Audio generation — ACE-Step via MLX (Apple Silicon) or CUDA PyTorch."""

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

    def _resolve_path(self, local_path: str) -> Path:
        return _resolve_project_path_fn(self._project_root, local_path)

    @staticmethod
    def _registry_scalar_default(entry, key: str, fallback):
        return _registry_scalar_default_fn(entry, key, fallback)

    def _resolve_version_block(self, entry, version_key: str | None) -> dict | None:
        return _resolve_version_block_fn(entry, version_key)

    def _local_bundle_root(self, entry, version_key: str | None) -> Path | None:
        return _local_bundle_root_fn(self._project_root, entry, version_key)

    def _get_generator(self, bundle_root: Path):
        if self._generator is None:
            gen = create_ace_step_generator(self.ctx, bundle_root)
            gen.load()
            self._generator = gen
        return self._generator

    def run(
        self,
        request: AudioGenerationRequest,
        exec_ctx: ExecutionContext,
    ) -> EngineResult:
        t0 = time.monotonic()
        model_id, version_key = parse_model_version(request.model)
        entry = self._registry.get(model_id)
        if entry is None:
            raise RuntimeError(f"Model {model_id!r} not found in registry")

        bundle_root = self._local_bundle_root(entry, version_key)
        if bundle_root is None or not bundle_root.is_dir():
            raise RuntimeError(f"Model bundle not found for {request.model!r}")

        cfg_cls = get_config_class(entry.family)
        if cfg_cls is not AceStepConfig:
            raise RuntimeError(
                f"MusicPipeline expects AceStepConfig for family {entry.family!r}, got {cfg_cls!r}"
            )
        config = cfg_cls()

        prepared = prepare_music_request(request, config, bundle_root)
        for level, message in prepared.log_events:
            exec_ctx.on_log(LogEvent(level=level, message=message))

        exec_ctx.on_log(
            LogEvent(
                level="info",
                message=f"Loading ACE-Step model: {model_id} from {bundle_root}",
            )
        )
        generator = self._get_generator(bundle_root)

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

            out_path = self._save_audio(waveform, model_id, batch_seed)
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
            dur_written = n_samples / 48_000.0 if n_samples else 0.0
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

    def _save_audio(self, waveform: Any, model_id: str, seed: int) -> Path:
        out_dir = self._project_root / "outputs" / "audio"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"ace_step_{model_id.replace('/', '_')}_{ts}_{seed}.wav"
        out_path = out_dir / fname

        wf = np.array(waveform) if not isinstance(waveform, np.ndarray) else waveform
        if wf.ndim == 3:
            wf = wf[0]
        if wf.ndim == 1:
            wf = wf[:, None]

        sf.write(str(out_path), wf, 48_000)
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
