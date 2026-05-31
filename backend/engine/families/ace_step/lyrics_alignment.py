"""Lyrics structure → coarse timestamp hints (no DiT cross-attn dependency)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

_SECTION_RE = re.compile(r"^\s*\[(.+?)\]\s*$")
_SINGABLE_LINE_RE = re.compile(r"^\s*(?!\[).+")


@dataclass(frozen=True)
class LyricsAlignmentSegment:
    label: str
    start_sec: float
    end_sec: float
    text: str


@dataclass(frozen=True)
class LyricsAlignmentResult:
    mode: str
    segments: tuple[LyricsAlignmentSegment, ...]

    def as_metadata(self) -> dict[str, Any]:
        return {
            "lyrics_alignment": self.mode,
            "lyrics_segments": [
                {
                    "label": s.label,
                    "start_sec": round(s.start_sec, 2),
                    "end_sec": round(s.end_sec, 2),
                    "text": s.text[:200],
                }
                for s in self.segments
            ],
        }


def estimate_lyrics_alignment(
    lyrics: str,
    *,
    duration_sec: float,
    instrumental: bool = False,
) -> LyricsAlignmentResult:
    """Evenly allocate non-empty lyric lines across track duration by section."""
    if instrumental:
        return LyricsAlignmentResult(mode="instrumental", segments=())

    text = (lyrics or "").strip()
    if not text:
        return LyricsAlignmentResult(mode="empty", segments=())

    sections: list[tuple[str, list[str]]] = []
    current_label = "body"
    current_lines: list[str] = []

    for raw in text.splitlines():
        line = raw.rstrip()
        sec = _SECTION_RE.match(line)
        if sec:
            if current_lines:
                sections.append((current_label, current_lines))
            current_label = sec.group(1).strip() or "section"
            current_lines = []
            continue
        if _SINGABLE_LINE_RE.match(line) and line.strip():
            current_lines.append(line.strip())
    if current_lines:
        sections.append((current_label, current_lines))

    singable = [(label, lines) for label, lines in sections if lines]
    if not singable:
        return LyricsAlignmentResult(mode="conditioning_only", segments=())

    total_lines = sum(len(lines) for _, lines in singable)
    if total_lines <= 0 or duration_sec <= 0:
        return LyricsAlignmentResult(mode="conditioning_only", segments=())

    sec_duration = duration_sec / max(len(singable), 1)
    segments: list[LyricsAlignmentSegment] = []
    t = 0.0
    for label, lines in singable:
        block_end = min(duration_sec, t + sec_duration)
        line_dur = (block_end - t) / max(len(lines), 1)
        for i, ln in enumerate(lines):
            start = t + i * line_dur
            end = min(block_end, start + line_dur)
            segments.append(
                LyricsAlignmentSegment(
                    label=label,
                    start_sec=start,
                    end_sec=end,
                    text=ln,
                )
            )
        t = block_end

    return LyricsAlignmentResult(mode="structure_estimate", segments=tuple(segments))


def format_lrc(alignment: LyricsAlignmentResult) -> Optional[str]:
    if not alignment.segments:
        return None
    lines: list[str] = []
    for seg in alignment.segments:
        mm = int(seg.start_sec // 60)
        ss = seg.start_sec % 60
        ts = f"[{mm:02d}:{ss:05.2f}]"
        lines.append(f"{ts}{seg.text}")
    return "\n".join(lines) + "\n"
