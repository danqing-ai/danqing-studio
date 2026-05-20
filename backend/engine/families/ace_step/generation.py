"""
ACE-Step generation — shared helpers + public Pipeline entry.

Pipeline and engine must import from this module only, not from ``generation_*`` internals.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Protocol, Tuple

import numpy as np

from backend.core.contracts import AudioGenerationRequest
from backend.engine.config.model_configs import AceStepConfig
from backend.engine.families.ace_step.lm_format import is_instrumental_lyrics
from backend.engine.families.ace_step.vocal_prompt import apply_vocal_type_to_prompt

SFT_GEN_PROMPT = """# Instruction
{instruction}

# Caption
{caption}

# Metas
{metadata}<|endoftext|>"""

DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"

SAMPLE_RATE = 48_000
LATENT_HOP_SAMPLES = 1920
VAE_TILE_CHUNK_SIZE = 32
VAE_TILE_OVERLAP = 8


def duration_to_latent_frames(duration: float) -> int:
    return max(128, int(round(float(duration) * SAMPLE_RATE / LATENT_HOP_SAMPLES)))


def snap_latent_frames_for_inference(frames: int) -> int:
    """Prefer even latent length (odd turbo lengths can collapse to silence/hum)."""
    if frames % 2 == 1:
        return max(128, frames - 1)
    return frames


def estimate_hum_ratio(wf: np.ndarray) -> float:
    mono = wf.mean(axis=1) if wf.ndim == 2 else wf
    seg = mono[: min(len(mono), SAMPLE_RATE * 5)]
    if len(seg) < SAMPLE_RATE:
        return 0.0
    spec = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(len(seg), 1.0 / SAMPLE_RATE)
    low = float(np.sum(spec[(freqs >= 50) & (freqs < 120)] ** 2))
    mid = float(np.sum(spec[(freqs >= 200) & (freqs < 4000)] ** 2))
    return low / (mid + 1e-12)


def estimate_mains_correlation(wf: np.ndarray, mains_hz: float = 50.0) -> float:
    mono = wf.mean(axis=1) if wf.ndim == 2 else wf
    n = min(len(mono), SAMPLE_RATE * 10)
    if n < SAMPLE_RATE:
        return 0.0
    seg = mono[:n]
    acf = np.correlate(seg, seg, mode="full")
    acf = acf[acf.size // 2 :]
    if acf[0] < 1e-12:
        return 0.0
    lag = int(round(SAMPLE_RATE / mains_hz))
    if lag >= len(acf):
        return 0.0
    return float(acf[lag] / acf[0])


def lyrics_looks_chinese(lyrics: str) -> bool:
    if not lyrics:
        return False
    cjk = len(re.findall(r"[\u4e00-\u9fff]", lyrics))
    latin = len(re.findall(r"[A-Za-z]", lyrics))
    return cjk >= 4 and cjk >= latin


def resolve_vocal_language(lyrics: str, vocal_language: str) -> str:
    """Pick vocal language for ``# Languages`` block; auto-detect when UI leaves it empty."""
    explicit = (vocal_language or "").strip().lower()
    if explicit and explicit not in ("auto", "detect", "automatic"):
        return explicit
    if lyrics_looks_chinese(lyrics):
        return "zh"
    return "en"


def vocal_language_mismatch_warning(lyrics: str, lang: str) -> Optional[str]:
    if is_instrumental_lyrics(lyrics):
        return None
    lang_l = (lang or "").strip().lower()
    if lyrics_looks_chinese(lyrics) and lang_l.startswith("en"):
        return (
            "歌词为中文但人声语言为 en；条件不一致，演唱可能极弱或像器乐。"
            "请将「人声语言」设为 zh（中文）。"
        )
    return None


def format_lyrics(lyrics: str, language: str) -> str:
    return f"# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>"


LYRIC_TOKEN_MAX_LENGTH = 2048


def warn_lyrics_char_heuristic(lyrics: str) -> Optional[str]:
    """Rough pre-check before tokenization (Chinese-heavy text ~1–2 chars/token)."""
    if is_instrumental_lyrics(lyrics):
        return None
    n = len(lyrics)
    if n >= 3500:
        return (
            f"歌词约 {n} 字，很可能超过模型 {LYRIC_TOKEN_MAX_LENGTH} token 上限，"
            "后段可能被截断而唱不出来；建议缩短或分段生成。"
        )
    if n >= 2200:
        return (
            f"歌词约 {n} 字，接近 token 上限，后段对齐可能变弱；"
            "可精简段落或拆成多次生成。"
        )
    return None


def warn_lyrics_token_truncation(
    token_count: int,
    *,
    max_length: int = LYRIC_TOKEN_MAX_LENGTH,
) -> Optional[str]:
    """Warn when lyric conditioning will be truncated (later verses often not sung)."""
    if token_count >= max_length:
        return (
            f"歌词编码已达 {token_count} token（上限 {max_length}），尾部已被截断，"
            "后段歌词很可能唱不出来。请缩短歌词或分段多次生成。"
        )
    if token_count >= int(max_length * 0.85):
        return (
            f"歌词较长（约 {token_count} token，上限 {max_length}），"
            "后段对齐可能变弱，建议精简或只保留关键段落。"
        )
    return None


def warn_weak_vocal_lyrics(lyrics: str) -> Optional[str]:
    """Return a user-facing warning when lyrics look unlikely to produce audible vocals."""
    if is_instrumental_lyrics(lyrics):
        return None
    lower = lyrics.lower()
    score_tags = (
        "cinematic",
        "instrumental",
        "guzheng",
        "war drum",
        "orchestral",
        "soundtrack",
        "bgm",
        "配乐",
        "史诗",
        "管弦",
        "古筝",
        "战鼓",
    )
    vocal_tags = (
        "vocal",
        "sing",
        "singer",
        "voice",
        "人声",
        "演唱",
        "歌声",
        "合唱",
        "主唱",
    )
    if any(k in lower for k in score_tags) and not any(k in lower for k in vocal_tags):
        return (
            "歌词/段落标签偏电影配乐或器乐（如 cinematic、guzheng、war drums），"
            "未强调演唱；成片可能几乎听不到人声。建议在歌词中写清可唱词句，"
            "并在描述中加入「清晰人声/主唱演唱」。"
        )
    bracket_tags = re.findall(r"\[[^\]]+\]", lyrics)
    singable_lines = [
        ln.strip()
        for ln in lyrics.splitlines()
        if ln.strip() and not ln.strip().startswith("[")
    ]
    singable_chars = sum(len(ln) for ln in singable_lines)
    if len(bracket_tags) >= 4 and singable_chars < 80:
        return (
            "歌词以 [Intro]/[Verse] 等制作标签为主、可唱正文很少；"
            "请在各段下补充具体歌词句子，否则模型易生成纯配乐。"
        )
    return None


def ensure_vocal_caption_hint(caption: str, lyrics: str, language: str = "en") -> str:
    """Nudge DiT caption toward lead vocals when non-instrumental lyrics are present."""
    cap = (caption or "").strip()
    if is_instrumental_lyrics(lyrics):
        return cap
    lower = cap.lower()
    lang = (language or "en").lower()
    vocal_in_cap = any(
        k in lower
        for k in (
            "vocal", "singing", "singer", "voice", "人声", "演唱", "歌声", "主唱",
            "male", "female", "choir", "chorus", "duet", "男声", "女声", "合唱", "对唱",
        )
    )
    if vocal_in_cap:
        return cap
    if lang.startswith("zh"):
        suffix = " 清晰的中文主唱人声，按所给歌词演唱，人声位于前景。"
        default = "带清晰中文主唱人声演唱的歌曲，人声位于前景。"
    else:
        suffix = " Clear lead vocals singing the provided lyrics, vocals in the front."
        default = "Song with clear lead vocals singing the provided lyrics."
    return (cap + suffix) if cap else default


def latent_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    va = np.asarray(a, dtype=np.float32).reshape(1, -1)
    vb = np.asarray(b, dtype=np.float32).reshape(1, -1)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom < 1e-12:
        return 1.0
    return float(np.sum(va * vb) / denom)


def latent_diff_mean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))))


def latents_collapsed_to_silence(
    latents: Any,
    src_latents: Any,
    *,
    cos_threshold: float = 0.95,
    diff_threshold: float = 0.10,
) -> tuple[bool, float, float]:
    cos = latent_cosine_similarity(latents, src_latents)
    diff = latent_diff_mean(latents, src_latents)
    return cos >= cos_threshold and diff < diff_threshold, cos, diff


def diffusion_retry_seed(base_seed: int, attempt: int) -> int:
    if attempt <= 0:
        return int(base_seed) & 0x7FFFFFFF
    mixed = int(base_seed) ^ (0x9E3779B9 * (attempt + 1))
    mixed = (mixed * 1_103_515_245 + 12_345) & 0x7FFFFFFF
    return mixed or 1


def resolve_dit_bundle(bundle_root: Path) -> Path:
    root = Path(bundle_root)
    if (root / "model.safetensors").is_file() or (root / "model.safetensors.index.json").is_file():
        return root
    for sub_name in (
        "acestep-v15-xl-sft",
        "acestep-v15-sft",
        "acestep-v15-turbo",
        "acestep-v15-base",
    ):
        sub = root / sub_name
        if sub.is_dir() and (sub / "model.safetensors").is_file():
            return sub
    raise RuntimeError(
        f"No ACE-Step DiT checkpoint (model.safetensors) under {root}. "
        "Expected weights at bundle root or in acestep-v15-*/ subdirectory."
    )


def resolve_silence_latent_path(bundle_root: Path, dit_bundle: Path) -> Path:
    for candidate in (dit_bundle / "silence_latent.pt", bundle_root / "silence_latent.pt"):
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        f"silence_latent.pt not found under {bundle_root} or {dit_bundle}"
    )


def resolve_silence_latent_npy_path(pt_path: Path) -> Path:
    return pt_path.with_suffix(".npy")


def load_silence_latent_numpy(pt_path: Path) -> np.ndarray:
    """Load silence latent for MLX (``.npy`` cache written by CUDA loader)."""
    npy_path = resolve_silence_latent_npy_path(pt_path)
    if not npy_path.is_file():
        raise RuntimeError(
            f"MLX ACE-Step requires {npy_path.name} next to {pt_path.name}. "
            "Run once on CUDA backend or: "
            "python -c \"...\" to convert silence_latent.pt to .npy"
        )
    return np.asarray(np.load(npy_path), dtype=np.float32)


def format_metadata(
    *,
    duration: float,
    bpm: Optional[int] = None,
    key_scale: str = "",
    time_signature: str = "",
) -> str:
    bpm_s = bpm if bpm is not None else "N/A"
    ts_s = time_signature if time_signature else "N/A"
    key_s = key_scale if key_scale else "N/A"
    dur_s = f"{int(round(duration))} seconds"
    return (
        f"- bpm: {bpm_s}\n"
        f"- timesignature: {ts_s}\n"
        f"- keyscale: {key_s}\n"
        f"- duration: {dur_s}\n"
    )


def normalize_waveform(wf: np.ndarray) -> np.ndarray:
    if wf.ndim == 2:
        wf = wf - np.mean(wf, axis=0, keepdims=True)
    else:
        wf = wf - float(np.mean(wf))
    peak = float(np.abs(wf).max())
    rms = float(np.sqrt(np.mean(wf**2)))
    if peak < 1e-5 or rms < 1e-6:
        raise RuntimeError(
            f"ACE-Step VAE decode produced near-silent audio (peak={peak:.2e}, rms={rms:.2e})"
        )
    if peak > 1.0:
        wf = wf / peak
    target_amp = 10.0 ** (-1.0 / 20.0)
    peak = float(np.abs(wf).max())
    if peak > 1e-8:
        wf = wf * (target_amp / peak)
    return wf

# --- Public API ---

class _AceStepGeneratorProto(Protocol):
    def load(self) -> None: ...
    def generate_waveform(self, **kwargs: Any) -> Any: ...
    @property
    def model_config(self) -> Any: ...


@dataclass
class AceStepPreparedRequest:
    """Registry-driven inference knobs for one audio generation request."""

    lyrics: str
    vocal_language: str
    effective_prompt: str
    steps: int
    shift: float
    is_turbo: bool
    log_events: List[Tuple[str, str]] = field(default_factory=list)


def create_ace_step_generator(ctx: Any, bundle_root: Path) -> _AceStepGeneratorProto:
    backend = getattr(ctx, "backend", "mlx")
    if backend == "mlx":
        from backend.engine.families.ace_step.generation_mlx import AceStepMlxGenerator

        return AceStepMlxGenerator(ctx, bundle_root)
    if backend == "cuda":
        from backend.engine.families.ace_step.generation_cuda import AceStepCudaGenerator

        return AceStepCudaGenerator(ctx, bundle_root)
    raise RuntimeError(
        f"ACE-Step audio requires mlx or cuda runtime (got {backend!r})"
    )


def resolve_bundle_is_turbo(bundle_root: Path) -> bool:
    """Read ``is_turbo`` from installed DiT ``config.json`` (authoritative over static defaults)."""
    dit = resolve_dit_bundle(bundle_root)
    cfg_path = dit / "config.json"
    if not cfg_path.is_file():
        return False
    with open(cfg_path, encoding="utf-8") as f:
        raw = json.load(f)
    return bool(raw.get("is_turbo", False))


def prepare_music_request(
    request: AudioGenerationRequest,
    config: AceStepConfig,
    bundle_root: Path,
) -> AceStepPreparedRequest:
    """Resolve lyrics/language/steps/shift from contract + registry config (no family branches in Pipeline)."""
    events: List[Tuple[str, str]] = []
    lyrics = (request.lyrics or "").strip()
    vocal_lang = resolve_vocal_language(lyrics, request.vocal_language or "")

    if request.instrumental:
        lyrics = "[Instrumental]"
    elif not lyrics:
        lyrics = "[Instrumental]"
        events.append(
            (
                "warning",
                "未填写歌词且未勾选纯器乐：将按纯音乐生成（无人声）。"
                "要人声请在「歌词」中填写内容，并关闭「纯器乐」。",
            )
        )
    elif not is_instrumental_lyrics(lyrics):
        preview = lyrics.replace("\n", " ")[:80]
        events.append(
            ("info", f"人声歌词已启用（约 {len(lyrics)} 字）: {preview!r}…"),
        )
        weak = warn_weak_vocal_lyrics(lyrics)
        if weak:
            events.append(("warning", weak))
        long_hint = warn_lyrics_char_heuristic(lyrics)
        if long_hint:
            events.append(("warning", long_hint))
        mismatch = vocal_language_mismatch_warning(lyrics, vocal_lang)
        if mismatch:
            events.append(("warning", mismatch))
        if not (request.vocal_language or "").strip():
            events.append(
                ("info", f"人声语言未指定，已根据歌词自动设为 {vocal_lang!r}"),
            )

    effective_prompt, vocal_tpl_log = apply_vocal_type_to_prompt(
        request.prompt or "",
        request.vocal_type or "",
        language=vocal_lang,
        instrumental=request.instrumental or is_instrumental_lyrics(lyrics),
    )
    if vocal_tpl_log:
        events.append(("info", vocal_tpl_log))

    is_turbo = resolve_bundle_is_turbo(bundle_root)
    steps = request.steps or config.default_infer_steps
    shift = float(config.default_shift)
    if is_turbo:
        steps = min(int(steps), int(config.turbo_infer_steps))
        shift = float(config.turbo_shift)
        events.append(
            (
                "info",
                f"ACE-Step turbo DiT: {steps} diffusion steps, shift={shift} "
                f"({getattr(bundle_root, 'name', bundle_root)})",
            )
        )

    return AceStepPreparedRequest(
        lyrics=lyrics,
        vocal_language=vocal_lang,
        effective_prompt=effective_prompt,
        steps=steps,
        shift=shift,
        is_turbo=is_turbo,
        log_events=events,
    )