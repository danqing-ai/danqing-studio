"""Shared ACE-Step 5Hz LM prompt parsing (MLX + PyTorch formatters)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

DEFAULT_LM_REWRITE_INSTRUCTION = (
    "Format the user's input into a more detailed and specific musical description:"
)

THINK_START_RE = re.compile(r"<think>\s*", re.IGNORECASE)
THINK_END_RE = re.compile(r"</think>", re.IGNORECASE)
AUDIO_CODE_RE = re.compile(r"<\|audio_code_\d+\|>")


@dataclass
class LmFormatResult:
    caption: str
    lyrics: str
    bpm: Optional[int] = None
    duration: Optional[int] = None
    keyscale: str = ""
    timesignature: str = ""
    language: str = ""


def resolve_lm_dir(bundle_root: Path) -> Optional[Path]:
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
    if tail:
        tail = re.sub(r"^#\s*Lyri[c|cs]?\s*\n", "", tail, flags=re.IGNORECASE)
        tail = re.sub(r"<\|im_end\|>\s*$", "", tail).strip()
        if tail:
            meta["lyrics_tail"] = tail
    return meta


def is_instrumental_lyrics(text: str) -> bool:
    """True when lyrics text requests no vocals."""
    t = (text or "").strip().lower()
    if not t:
        return True
    if t in ("[instrumental]", "instrumental", "[inst]"):
        return True
    return bool(re.fullmatch(r"\[?instrumental\]?", t))


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
) -> LmFormatResult:
    out_caption = str(meta.get("caption") or caption_in)
    lm_lyrics = str(meta.get("lyrics_tail") or "").strip()
    user_wants_vocals = not is_instrumental_lyrics(lyrics_in)
    if user_wants_vocals and is_instrumental_lyrics(lm_lyrics):
        # 5Hz LM often emits [Instrumental] even when user supplied lyrics — keep user text.
        out_lyrics = lyrics_in
    elif lm_lyrics:
        out_lyrics = lm_lyrics
    else:
        out_lyrics = lyrics_in
    out_bpm = meta.get("bpm", bpm)
    if isinstance(out_bpm, str):
        try:
            out_bpm = int(float(out_bpm))
        except ValueError:
            out_bpm = bpm
    out_duration = int(round(duration)) if duration is not None else None
    return LmFormatResult(
        caption=out_caption,
        lyrics=out_lyrics,
        bpm=out_bpm if isinstance(out_bpm, int) else None,
        duration=out_duration if isinstance(out_duration, int) else None,
        keyscale=str(meta.get("keyscale") or keyscale or ""),
        timesignature=str(meta.get("timesignature") or timesignature or ""),
        language=str(meta.get("language") or language or "en"),
    )
