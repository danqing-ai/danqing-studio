"""
HeartMuLa generation — shared helpers + public Pipeline entry.

Pipeline and engine must import from this module only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Protocol, Tuple

import numpy as np

from backend.core.contracts import AudioGenerationRequest
from backend.engine.config.model_configs import HeartMulaConfig

SAMPLE_RATE = 48_000
FRAME_RATE = 12.5


def prompt_to_tags(prompt: str) -> str:
    """Map UI ``prompt`` to HeartMuLa ``<tag>…</tag>`` caption (comma-separated styles)."""
    text = (prompt or "").strip()
    if not text:
        return "pop, melodic, high quality"
    if text.startswith("<tag>"):
        return text
    return text


def duration_to_max_frames(duration: float) -> int:
    return max(1, int(float(duration) * FRAME_RATE))


def estimate_hf_noise_ratio(wf: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """High-frequency energy share — elevated values often indicate codec hiss/static."""
    mono = np.asarray(wf, dtype=np.float64).reshape(-1)
    if mono.size < sample_rate // 4:
        return 0.0
    spec = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(mono.size, 1.0 / sample_rate)
    total = float(np.sum(spec) + 1e-12)
    hf = float(np.sum(spec[freqs >= 8_000.0]))
    return hf / total


def _clamp_float(value: Optional[float], default: float, lo: float, hi: float) -> float:
    if value is None:
        return float(default)
    return max(lo, min(hi, float(value)))


def _clamp_int(value: Optional[int], default: int, lo: int, hi: int) -> int:
    if value is None:
        return int(default)
    return max(lo, min(hi, int(value)))


@dataclass
class HeartMulaPreparedRequest:
    tags: str
    lyrics: str
    duration: float
    temperature: float
    topk: int
    cfg_scale: float
    codec_steps: int
    codec_guidance: float
    long_form_temperature: float
    long_form_topk: int
    log_events: List[Tuple[str, str]] = field(default_factory=list)


class _HeartMulaGeneratorProto(Protocol):
    def load(self) -> None: ...
    def generate_waveform(self, **kwargs: Any) -> Any: ...


def create_heartmula_generator(ctx: Any, bundle_root: Path) -> _HeartMulaGeneratorProto:
    backend = getattr(ctx, "backend", "mlx")
    if backend == "mlx":
        from backend.engine.families.heartmula.generation_mlx import HeartMulaMlxGenerator

        return HeartMulaMlxGenerator(ctx, bundle_root)
    if backend == "cuda":
        from backend.engine.families.heartmula.generation_cuda import HeartMulaCudaGenerator

        return HeartMulaCudaGenerator(ctx, bundle_root)
    raise RuntimeError(
        f"HeartMuLa audio requires mlx or cuda runtime (got {backend!r})"
    )


def prepare_heartmula_request(
    request: AudioGenerationRequest,
    config: HeartMulaConfig,
) -> HeartMulaPreparedRequest:
    """Resolve tags/lyrics/duration from contract (no family branches in Pipeline)."""
    events: List[Tuple[str, str]] = []
    lyrics = (request.lyrics or "").strip()
    if request.instrumental or not lyrics:
        lyrics = ""
        if not request.instrumental and not (request.lyrics or "").strip():
            events.append(
                (
                    "warning",
                    "未填写歌词：将仅按风格标签生成（无人声段落）。"
                    "要人声请在「歌词」中填写 [Verse]/[Chorus] 等结构。",
                )
            )
    else:
        preview = lyrics.replace("\n", " ")[:80]
        events.append(("info", f"歌词已启用（约 {len(lyrics)} 字）: {preview!r}…"))

    tags = prompt_to_tags(request.prompt or "")
    duration = float(request.duration or config.default_duration_seconds)
    if duration > config.max_duration_seconds:
        duration = float(config.max_duration_seconds)
        events.append(
            (
                "warning",
                f"时长已限制为 {config.max_duration_seconds}s（HeartMuLa 上限）",
            )
        )
    elif duration > 60.0:
        est_frames = duration_to_max_frames(duration)
        events.append(
            (
                "warning",
                f"时长 {duration:.0f}s 较长（约 {est_frames} 帧自回归），MLX 上 LM 耗时会显著增加；"
                f"Codec 将自动分块解码（~30s/块 overlap-add）。上限 {config.max_duration_seconds:.0f}s，"
                "建议先 30–90s 试听后加长。",
            )
        )

    cfg = _clamp_float(
        request.guidance,
        config.default_cfg_scale,
        config.cfg_scale_min,
        config.cfg_scale_max,
    )
    temperature = _clamp_float(
        request.temperature,
        config.default_temperature,
        config.temperature_min,
        config.temperature_max,
    )
    topk = _clamp_int(
        request.top_k,
        config.default_topk,
        config.topk_min,
        config.topk_max,
    )
    codec_steps = _clamp_int(
        request.codec_steps,
        config.codec_ode_steps,
        config.codec_ode_steps_min,
        config.codec_ode_steps_max,
    )
    codec_guidance = _clamp_float(
        request.codec_guidance,
        config.codec_guidance_scale,
        config.codec_guidance_min,
        config.codec_guidance_max,
    )

    lf_temp = _clamp_float(
        request.long_form_temperature,
        config.long_form_temperature,
        config.long_form_temperature_min,
        config.long_form_temperature_max,
    )
    lf_topk = _clamp_int(
        request.long_form_topk,
        config.long_form_topk,
        config.long_form_topk_min,
        config.long_form_topk_max,
    )

    return HeartMulaPreparedRequest(
        tags=tags,
        lyrics=lyrics,
        duration=duration,
        temperature=temperature,
        topk=topk,
        cfg_scale=cfg,
        codec_steps=codec_steps,
        codec_guidance=codec_guidance,
        long_form_temperature=lf_temp,
        long_form_topk=lf_topk,
        log_events=events,
    )
