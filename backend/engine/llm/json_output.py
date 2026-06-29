"""Extract and validate JSON objects from local LLM text responses."""
from __future__ import annotations

import json
import re


def extract_json_object(text: str) -> dict:
    """Parse a single JSON object from model output; fail loud on invalid JSON."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("LLM response is empty; expected a JSON object")

    cleaned = _strip_markdown_fence(raw)
    start = cleaned.find("{")
    if start > 0:
        cleaned = cleaned[start:]

    last_error: json.JSONDecodeError | None = None
    for candidate in _json_candidates(cleaned):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(parsed, dict):
            raise ValueError(
                f"LLM JSON root must be an object, got {type(parsed).__name__}"
            )
        return parsed

    snippet = cleaned[:280].replace("\n", " ")
    detail = f": {last_error.msg}" if last_error else ""
    raise ValueError(f"LLM response is not valid JSON{detail}: {snippet}")


def _json_candidates(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        c = (candidate or "").strip()
        if c and c not in seen:
            seen.add(c)
            out.append(c)

    add(text)
    balanced = _extract_balanced_object(text)
    add(balanced)
    add(_extract_braced_object(text))
    return out


def _extract_balanced_object(text: str) -> str:
    raw = text or ""
    start = raw.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : idx + 1]
    return ""


def _strip_markdown_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_braced_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return ""
    return text[start : end + 1]
