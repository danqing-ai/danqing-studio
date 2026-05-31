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
DEFAULT_LM_INSPIRED_INSTRUCTION = (
    "Expand the user's input into a more detailed and specific musical description:"
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


def build_audio_codes_tensor(
    indices: tuple[int, ...] | list[int],
    *,
    device: Any,
    max_codes: Optional[int] = None,
) -> Any:
    """Build upstream DiT ``audio_codes`` indices tensor ``[1, T, 1]``."""
    import torch

    use = list(indices)
    if max_codes is not None and max_codes > 0:
        use = use[:max_codes]
    if not use:
        raise ValueError("audio_code indices must be non-empty")
    tensor = torch.tensor(use, dtype=torch.long, device=device)
    return tensor.reshape(1, -1, 1)


_LYRIC_HEADER_RE = re.compile(r"^#\s*Lyri[c|cs]?\s*", re.IGNORECASE | re.MULTILINE)
_FORMAT_LYRIC_PREFIX_RE = re.compile(
    r"^#\s*Languages\s*\n.*?\n\n#\s*Lyric\s*",
    re.IGNORECASE | re.DOTALL,
)


def normalize_lyrics_body(text: str) -> str:
    """Strip conditioning wrappers / ``# Lyric`` headers for checks and DiT input."""
    t = (text or "").strip()
    if not t:
        return ""
    t = _FORMAT_LYRIC_PREFIX_RE.sub("", t)
    t = _LYRIC_HEADER_RE.sub("", t)
    t = re.sub(r"<\|endoftext\|>\s*$", "", t).strip()
    return t


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
        "ACE-Step 5Hz LM did not produce vocal lyrics for this prompt. "
        "Add lyrics in the lyrics field, or describe lead vocals explicitly in the prompt."
    )


def build_inspiration_user_content(
    query: str,
    *,
    instrumental: bool,
    vocal_language: str = "",
    force_vocals: bool = False,
) -> str:
    """User turn for inspiration / create_sample — steer LM away from [Instrumental]."""
    query_in = (query or "").strip() or "NO USER INPUT"
    if instrumental:
        return f"{query_in}\n\n[Instrumental]"
    lang = (vocal_language or "en").strip()
    if lang.lower().startswith("zh"):
        vocal_hint = (
            "请生成完整的中文演唱歌词，包含 [Verse]、[Chorus] 等结构标签。"
            "必须有人声演唱，不要使用 [Instrumental]。"
        )
        if force_vocals:
            vocal_hint = (
                "重要：这是一首必须有人声演唱的歌曲，不是纯音乐。"
                "请写出完整、可演唱的中文歌词（含 [Verse]、[Chorus]），"
                "禁止输出 [Instrumental] 或纯器乐描述。"
            )
    else:
        vocal_hint = (
            f"Generate full sung lyrics in {lang} with [Verse] and [Chorus] structure tags. "
            "Include clear lead vocals; do not use [Instrumental]."
        )
        if force_vocals:
            vocal_hint = (
                f"IMPORTANT: This track MUST have lead vocals in {lang}, not instrumental. "
                "Write complete singable lyrics with [Verse] and [Chorus]. "
                "Do NOT output [Instrumental]."
            )
    return f"{query_in}\n\n{vocal_hint}"


def inspiration_understand_config(
    *,
    duration: Optional[float],
    user_metadata: Optional[dict[str, Any]],
) -> Any:
    """Constrained config for inspiration phase-1 (metadata + lyrics)."""
    from backend.engine.families.ace_step.constrained_generate import (
        ConstrainedGenerationConfig,
        compute_max_new_tokens,
    )

    max_new = compute_max_new_tokens(
        target_duration=duration,
        generation_phase="understand",
    )
    max_new = min(max_new, 1200)
    return ConstrainedGenerationConfig(
        target_duration=duration,
        generation_phase="understand",
        user_metadata=user_metadata,
        stop_at_reasoning=False,
        skip_genres=False,
        max_new_tokens=max_new,
    )


def parse_inspiration_understand_output(
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


def synthesize_fallback_vocal_lyrics(caption: str, *, vocal_language: str) -> str:
    """Minimal singable placeholder when 5Hz LM cannot produce lyrics."""
    theme = normalize_lyrics_body(caption) or "这首歌"
    lang = (vocal_language or "en").strip().lower()
    if lang.startswith("zh"):
        return (
            f"[Verse 1]\n{theme}\n在风里轻轻唱\n\n"
            f"[Chorus]\n{theme}\n让我们一起唱"
        )
    return (
        f"[Verse 1]\n{theme}\nSinging through the night\n\n"
        f"[Chorus]\n{theme}\nSing it out loud"
    )


def build_lyrics_only_prompt(
    tokenizer: Any,
    *,
    caption: str,
    lyrics_placeholder: str,
    cot_text: str,
    instruction: str = DEFAULT_LM_INSPIRED_INSTRUCTION,
) -> str:
    """Prefill planner CoT then continue with free-form vocal lyrics."""
    user_prompt = f"# Caption\n{caption}\n\n# Lyric\n{lyrics_placeholder}\n"
    formatted = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": f"# Instruction\n{instruction}\n\n"},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return formatted + cot_text + "\n\n"


def run_lyrics_only_fallback(
    generate_fn: Callable[[str, Any], str],
    tokenizer: Any,
    *,
    caption: str,
    metadata: dict[str, Any],
    vocal_language: str,
    duration: Optional[float] = None,
) -> str:
    """Second pass: metadata is fixed; ask LM to continue with vocal lyrics only."""
    from backend.engine.families.ace_step.constrained_generate import (
        ConstrainedGenerationConfig,
    )

    cot = format_metadata_as_cot(metadata)
    lang = (vocal_language or "en").strip().lower()
    if lang.startswith("zh"):
        placeholder = (
            "请在此写出完整的中文演唱歌词（含 [Verse]、[Chorus]），"
            "不要写 [Instrumental]。"
        )
    else:
        placeholder = (
            "Write complete sung lyrics with [Verse] and [Chorus] here. "
            "Do not use [Instrumental]."
        )
    prompt = build_lyrics_only_prompt(
        tokenizer,
        caption=caption,
        lyrics_placeholder=placeholder,
        cot_text=cot,
    )
    output_text = generate_fn(
        prompt,
        ConstrainedGenerationConfig(
            target_duration=duration,
            generation_phase="understand",
            use_constrained_decoding=False,
            stop_at_reasoning=False,
            max_new_tokens=800,
        ),
    )
    lyrics = extract_lm_generated_lyrics(output_text, prefilled_think_end=True)
    if not lyrics:
        lyrics = normalize_lyrics_body(str(parse_lm_output(output_text).get("lyrics_tail") or ""))
    return lyrics


def ensure_vocal_lyrics_for_inspiration(
    generate_fn: Callable[[str, Any], str],
    tokenizer: Any,
    chat_prompt: Callable[[str, str], str],
    *,
    query_in: str,
    vocal_language: str,
    duration: Optional[float],
    user_metadata: Optional[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    """Run understand (+ retry / lyrics-only fallback) until singable lyrics exist."""
    last_meta: dict[str, Any] = {"caption": query_in}
    understand_cfg = inspiration_understand_config(
        duration=duration,
        user_metadata=user_metadata,
    )

    for force in (False, True):
        user_content = build_inspiration_user_content(
            query_in,
            instrumental=False,
            vocal_language=vocal_language,
            force_vocals=force,
        )
        prompt = chat_prompt(DEFAULT_LM_INSPIRED_INSTRUCTION, user_content)
        output_text = generate_fn(prompt, understand_cfg)
        meta, lyrics = parse_inspiration_understand_output(output_text, instrumental=False)
        last_meta = meta
        if lyrics and not is_instrumental_lyrics(lyrics):
            return meta, lyrics

    caption = str(last_meta.get("caption") or query_in)
    fallback_meta = {
        k: v
        for k, v in {
            "bpm": last_meta.get("bpm"),
            "caption": caption,
            "duration": last_meta.get("duration"),
            "keyscale": last_meta.get("keyscale"),
            "language": last_meta.get("language") or vocal_language,
            "timesignature": last_meta.get("timesignature"),
        }.items()
        if v not in (None, "")
    }
    lyrics = run_lyrics_only_fallback(
        generate_fn,
        tokenizer,
        caption=caption,
        metadata=fallback_meta,
        vocal_language=vocal_language,
        duration=duration,
    )
    if lyrics and not is_instrumental_lyrics(lyrics):
        merged = dict(last_meta)
        merged["lyrics_tail"] = lyrics
        merged["caption"] = caption
        return merged, lyrics

    placeholder = synthesize_fallback_vocal_lyrics(caption, vocal_language=vocal_language)
    import logging

    logging.getLogger(__name__).warning(
        "ACE-Step LM: using minimal placeholder lyrics for %r (5Hz LM returned no vocals)",
        query_in[:80],
    )
    merged = dict(last_meta)
    merged["lyrics_tail"] = placeholder
    merged["caption"] = caption
    merged["lyrics_fallback"] = True
    return merged, placeholder


def format_sample_understand_config(
    *,
    duration: Optional[float],
    user_metadata: Optional[dict[str, Any]],
    metadata_only: bool,
) -> Any:
    """Phase-1 config for ``format_sample`` (rewrite metadata; optional lyrics stop)."""
    from backend.engine.families.ace_step.constrained_generate import (
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

    from backend.engine.families.ace_step.constrained_generate import (
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
