"""Shared ACE-Step 5Hz LM prompt parsing (MLX + PyTorch formatters)."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

DEFAULT_LM_REWRITE_INSTRUCTION = (
    "Format the user's input into a more detailed and specific musical description:"
)
DEFAULT_LM_SIMPLE_INSTRUCTION = (
    "Generate audio semantic tokens based on the given conditions:"
)

THINK_START_RE = re.compile(r"<think>\s*", re.IGNORECASE)
THINK_END_RE = re.compile(r"</think>", re.IGNORECASE)
AUDIO_CODE_RE = re.compile(r"<\|audio_code_\d+\|>")
AUDIO_CODE_NUM_RE = re.compile(r"<\|audio_code_(\d+)\|>")


@dataclass
class LmFormatResult:
    caption: str
    lyrics: str
    bpm: Optional[int] = None
    duration: Optional[int] = None
    keyscale: str = ""
    timesignature: str = ""
    language: str = ""
    audio_codes: str = ""
    audio_code_indices: tuple[int, ...] = ()


def resolve_lm_dir(
    bundle_root: Path,
    *,
    preferred: Optional[Path] = None,
) -> Optional[Path]:
    if preferred is not None and (preferred / "model.safetensors").is_file():
        return preferred
    root = Path(bundle_root)
    for name in ("acestep-5Hz-lm-1.7B", "acestep-5Hz-lm-0.6B", "acestep-5Hz-lm-4B"):
        candidate = root / name
        if (candidate / "model.safetensors").is_file():
            return candidate
    return None


def parse_reasoning_block(text: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if not text.strip():
        return metadata

    lines = text.split("\n")
    current_key: Optional[str] = None
    value_lines: list[str] = []

    def flush() -> None:
        nonlocal current_key, value_lines
        if not current_key or not value_lines:
            current_key = None
            value_lines = []
            return
        value = "\n".join(value_lines).strip()
        if current_key == "bpm":
            try:
                metadata["bpm"] = int(float(value.split()[0]))
            except (ValueError, IndexError):
                metadata["bpm"] = value
        elif current_key == "caption":
            metadata["caption"] = value
        elif current_key == "duration":
            try:
                metadata["duration"] = int(float(re.findall(r"\d+", value)[0]))
            except (IndexError, ValueError):
                metadata["duration"] = value
        elif current_key == "keyscale":
            metadata["keyscale"] = value
        elif current_key == "timesignature":
            metadata["timesignature"] = value
        elif current_key == "language":
            metadata["language"] = value
        elif current_key == "genres":
            metadata["genres"] = value
        current_key = None
        value_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("<"):
            continue
        if not line[:1].isspace() and ":" in line:
            flush()
            key, _, rest = line.partition(":")
            current_key = key.strip().lower()
            if rest.strip():
                value_lines.append(rest.strip())
        elif line.startswith((" ", "\t")) and current_key:
            value_lines.append(line)
    flush()
    return metadata


def extract_think_body(output_text: str) -> tuple[str, str]:
    text = output_text
    code_match = AUDIO_CODE_RE.search(text)
    if code_match:
        text = text[: code_match.start()]

    start_m = THINK_START_RE.search(text)
    if start_m:
        body = text[start_m.end() :]
        end_m = THINK_END_RE.search(body)
        if end_m:
            return body[: end_m.start()].strip(), body[end_m.end() :].strip()
        return body.strip(), ""

    end_m = THINK_END_RE.search(text)
    if end_m:
        return text[: end_m.start()].strip(), text[end_m.end() :].strip()
    return text.strip(), ""


def parse_lm_output(output_text: str) -> dict[str, Any]:
    reasoning, tail = extract_think_body(output_text)
    meta = parse_reasoning_block(reasoning)
    codes = extract_audio_codes(output_text)
    if codes:
        meta["audio_codes"] = codes
        meta["audio_code_indices"] = list(parse_audio_code_indices(codes))
    if tail:
        meta["lyrics_tail"] = normalize_lyrics_body(tail)
    return meta


def extract_audio_codes(text: str) -> str:
    matches = AUDIO_CODE_RE.findall(text or "")
    return "".join(matches)


def parse_audio_code_indices(text: str) -> tuple[int, ...]:
    return tuple(int(v) for v in AUDIO_CODE_NUM_RE.findall(text or ""))


_LYRIC_HEADER_RE = re.compile(r"^#\s*Lyri[c|cs]?\s*", re.IGNORECASE | re.MULTILINE)
_FORMAT_LYRIC_PREFIX_RE = re.compile(
    r"^#\s*Languages\s*\n.*?\n\n#\s*Lyric\s*",
    re.IGNORECASE | re.DOTALL,
)
_SECTION_HEADER_LINE_RE = re.compile(r"^\s*\[(.+?)\]\s*$")
_DROP_SECTION_KEYS = frozenset({"start", "end"})


def normalize_lyrics_body(text: str) -> str:
    """Strip conditioning wrappers / ``# Lyric`` headers for checks and DiT input."""
    t = (text or "").strip()
    if not t:
        return ""
    t = _FORMAT_LYRIC_PREFIX_RE.sub("", t)
    t = _LYRIC_HEADER_RE.sub("", t)
    t = re.sub(r"<\|endoftext\|>\s*$", "", t).strip()
    return t


def compact_vocal_lyrics_structure(text: str) -> str:
    """Drop empty scaffolding sections (e.g. bare ``[intro]``) before LM / DiT conditioning.

    Keeps singable lines intact; removes ``[start]``/``[end]`` and section headers with no lyrics.
    """
    t = normalize_lyrics_body(text)
    if not t or is_instrumental_lyrics(t):
        return t

    sections: list[tuple[str, list[str]]] = []
    cur_label = ""
    cur_lines: list[str] = []

    for raw in t.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        sec = _SECTION_HEADER_LINE_RE.match(stripped)
        if sec:
            if cur_label or cur_lines:
                sections.append((cur_label, cur_lines))
            key_norm = re.sub(r"[\s\d\-_]+", "", sec.group(1).strip().lower())
            if key_norm in _DROP_SECTION_KEYS:
                cur_label = ""
                cur_lines = []
            else:
                cur_label = sec.group(1).strip()
                cur_lines = []
            continue
        cur_lines.append(stripped)

    if cur_label or cur_lines:
        sections.append((cur_label, cur_lines))

    kept = [(label, lines) for label, lines in sections if lines]
    if not kept:
        return t

    out: list[str] = []
    for label, lines in kept:
        if label:
            out.append(f"[{label}]")
        out.extend(lines)
    return "\n".join(out)


def extract_lyrics_after_thinking(output_text: str) -> str:
    end_m = THINK_END_RE.search(output_text or "")
    if not end_m:
        return ""
    tail = output_text[end_m.end() :]
    tail = AUDIO_CODE_RE.sub("", tail)
    tail = re.sub(r"<\|im_end\|>\s*$", "", tail).strip()
    return normalize_lyrics_body(tail)


def extract_lm_generated_lyrics(output_text: str, *, prefilled_think_end: bool = False) -> str:
    """Parse lyrics from ``generate_constrained_*`` output (new tokens only)."""
    text = (output_text or "").strip()
    if not text:
        return ""
    if not prefilled_think_end and THINK_END_RE.search(text):
        return extract_lyrics_after_thinking(text)
    text = AUDIO_CODE_RE.sub("", text)
    text = re.sub(r"<\|im_end\|>\s*$", "", text).strip()
    return normalize_lyrics_body(text)


def is_instrumental_lyrics(text: str) -> bool:
    """True when lyrics text requests no vocals."""
    t = normalize_lyrics_body(text).lower()
    if not t:
        return True
    if t in ("[instrumental]", "instrumental", "[inst]"):
        return True
    return bool(re.fullmatch(r"\[?instrumental\]?", t))


def vocal_lyrics_required_error() -> str:
    return (
        "ACE-Step 人声曲目需要歌词：请在歌词框填写内容，或使用 Composer 中的「AI 生成歌词」。"
        "5Hz LM 仅格式化 caption/元数据，不再从描述自动生成歌词。"
    )


def parse_lm_understand_output(
    output_text: str,
    *,
    instrumental: bool,
) -> tuple[dict[str, Any], str]:
    meta = parse_lm_output(output_text)
    lyrics = extract_lm_generated_lyrics(output_text)
    if not lyrics:
        lyrics = normalize_lyrics_body(str(meta.get("lyrics_tail") or ""))
    if lyrics:
        meta["lyrics_tail"] = lyrics
    elif instrumental:
        meta["lyrics_tail"] = "[Instrumental]"
    return meta, lyrics


def format_sample_understand_config(
    *,
    duration: Optional[float],
    user_metadata: Optional[dict[str, Any]],
    metadata_only: bool,
) -> Any:
    """Phase-1 config for ``format_sample`` (rewrite metadata; optional lyrics stop)."""
    from backend.engine.families.ace_step.lm.constrained_generate import (
        ConstrainedGenerationConfig,
        compute_max_new_tokens,
    )

    max_new = compute_max_new_tokens(
        target_duration=duration,
        generation_phase="understand",
    )
    max_new = min(max_new, 1200 if not metadata_only else 512)
    return ConstrainedGenerationConfig(
        target_duration=duration,
        generation_phase="understand",
        user_metadata=user_metadata,
        stop_at_reasoning=metadata_only,
        skip_genres=False,
        max_new_tokens=max_new,
    )


def build_lm_format_result(
    meta: dict[str, Any],
    *,
    caption_in: str,
    lyrics_in: str,
    duration: Optional[float],
    bpm: Optional[int],
    keyscale: str,
    timesignature: str,
    language: str,
    want_vocals: bool = False,
    preserve_user_lyrics: bool = False,
) -> LmFormatResult:
    out_caption = str(meta.get("caption") or caption_in)
    lm_lyrics = normalize_lyrics_body(str(meta.get("lyrics_tail") or ""))
    lyrics_in_norm = normalize_lyrics_body(lyrics_in)
    user_wants_vocals = want_vocals or not is_instrumental_lyrics(lyrics_in)
    if (
        preserve_user_lyrics
        and lyrics_in_norm
        and not is_instrumental_lyrics(lyrics_in_norm)
    ):
        out_lyrics = lyrics_in_norm
    elif user_wants_vocals and is_instrumental_lyrics(lm_lyrics):
        if lyrics_in_norm and not is_instrumental_lyrics(lyrics_in_norm):
            out_lyrics = lyrics_in_norm
        else:
            out_lyrics = ""
    elif lm_lyrics and not (
        preserve_user_lyrics
        and lyrics_in_norm
        and not is_instrumental_lyrics(lyrics_in_norm)
    ):
        out_lyrics = lm_lyrics
    elif lyrics_in_norm:
        out_lyrics = lyrics_in_norm
    else:
        out_lyrics = ""
    out_bpm = meta.get("bpm", bpm)
    if isinstance(out_bpm, str):
        try:
            out_bpm = int(float(out_bpm))
        except ValueError:
            out_bpm = bpm
    out_duration = meta.get("duration")
    if isinstance(out_duration, str):
        try:
            out_duration = int(float(re.findall(r"\d+", out_duration)[0]))
        except (IndexError, ValueError):
            out_duration = None
    if out_duration is None and duration is not None:
        out_duration = int(round(duration))

    code_str = str(meta.get("audio_codes") or "")
    code_indices = tuple(meta.get("audio_code_indices") or parse_audio_code_indices(code_str))

    return LmFormatResult(
        caption=out_caption,
        lyrics=out_lyrics,
        bpm=out_bpm if isinstance(out_bpm, int) else None,
        duration=out_duration if isinstance(out_duration, int) else None,
        keyscale=str(meta.get("keyscale") or keyscale or ""),
        timesignature=str(meta.get("timesignature") or timesignature or ""),
        language=str(meta.get("language") or language or "en"),
        audio_codes=code_str,
        audio_code_indices=code_indices,
    )


def lm_planner_codes_enabled() -> bool:
    """Phase-2 audio codes after planner metadata (upstream ``llm_dit`` path)."""
    return os.environ.get("ACESTEP_LM_CODES", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


_COT_FIELD_ORDER = (
    "bpm",
    "caption",
    "duration",
    "keyscale",
    "language",
    "timesignature",
)


def format_metadata_as_cot(metadata: dict[str, Any]) -> str:
    """Format planner metadata as ``<think>`` block (upstream YAML-style)."""
    lines: list[str] = []
    for key in _COT_FIELD_ORDER:
        if key not in metadata or metadata[key] is None:
            continue
        value = metadata[key]
        if key == "timesignature" and isinstance(value, str) and value.endswith("/4"):
            value = value.split("/")[0]
        if isinstance(value, str) and value.isdigit():
            value = int(value)
        lines.append(f"{key}: {value}")
    body = "\n".join(lines)
    if not body:
        return "<think>\n\n</think>"
    return f"<think>\n{body}\n</think>"


def build_codes_phase_prompt(
    tokenizer: Any,
    *,
    caption: str,
    lyrics: str,
    cot_text: str,
    instruction: str = DEFAULT_LM_SIMPLE_INSTRUCTION,
) -> str:
    """Build phase-2 prompt: caption/lyrics user turn + pre-filled CoT + open assistant for codes."""
    user_prompt = f"# Caption\n{caption}\n\n# Lyric\n{lyrics}\n"
    formatted = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": f"# Instruction\n{instruction}\n\n"},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return formatted + cot_text + "\n\n"


def build_codes_uncond_prompt(
    tokenizer: Any,
    *,
    negative_prompt: str = "NO USER INPUT",
    instruction: str = DEFAULT_LM_SIMPLE_INSTRUCTION,
) -> str:
    """Unconditional prompt for codes-phase CFG (upstream training-aligned layout)."""
    neg = (negative_prompt or "").strip()
    user_prompt = neg if neg and neg != "NO USER INPUT" else "NO USER INPUT"
    cot_for_prompt = "<think>\n\n</think>"
    formatted = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": f"# Instruction\n{instruction}\n\n"},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return formatted + cot_for_prompt + "\n\n"


def default_lm_codes_cfg_scale() -> float:
    raw = os.environ.get("ACESTEP_LM_CFG_SCALE", "2.0").strip()
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 2.0


def run_planner_codes_phase(
    *,
    generate_fn: Callable[[str, Any], str],
    tokenizer: Any,
    result: LmFormatResult,
    duration_hint: Optional[float] = None,
) -> LmFormatResult:
    """Phase 2: constrained audio codes from planner metadata (upstream ``llm_dit``)."""
    if not lm_planner_codes_enabled():
        return result

    from backend.engine.families.ace_step.lm.constrained_generate import (
        ConstrainedGenerationConfig,
    )

    metadata = {
        "bpm": result.bpm,
        "caption": result.caption,
        "duration": result.duration,
        "keyscale": result.keyscale,
        "language": result.language,
        "timesignature": result.timesignature,
    }
    metadata = {k: v for k, v in metadata.items() if v not in (None, "")}

    target = duration_hint
    if (target is None or target <= 0) and result.duration:
        target = float(result.duration)

    cot_text = format_metadata_as_cot(metadata)
    prompt = build_codes_phase_prompt(
        tokenizer,
        caption=result.caption,
        lyrics=result.lyrics,
        cot_text=cot_text,
    )
    uncond_prompt = build_codes_uncond_prompt(tokenizer)
    cfg_scale = default_lm_codes_cfg_scale()
    output_text = generate_fn(
        prompt,
        ConstrainedGenerationConfig(
            target_duration=target,
            generation_phase="codes",
            user_metadata=None,
            stop_at_reasoning=False,
            skip_genres=True,
            skip_caption=True,
            skip_language=True,
            cfg_scale=cfg_scale,
            uncond_prompt=uncond_prompt if cfg_scale > 1.0 else "",
        ),
    )
    codes = extract_audio_codes(output_text)
    indices = parse_audio_code_indices(codes)
    if not indices:
        raise RuntimeError(
            "ACE-Step LM planner codes phase produced no audio codes "
            f"(target_duration={target!r})"
        )
    return replace(result, audio_codes=codes, audio_code_indices=indices)
