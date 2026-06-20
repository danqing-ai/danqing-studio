"""Parse human-readable model size strings to GB (aligned with frontend memoryHint)."""

from __future__ import annotations

import re
from typing import Optional


def parse_human_size_to_gb(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    text = re.sub(r"[,~≈]", "", text)
    text = re.sub(r"\s+", "", text)
    match = re.match(r"([\d.]+)\s*(tb|t|gb|g|mb|m)?", text)
    if not match:
        return None
    try:
        amount = float(match.group(1))
    except ValueError:
        return None
    if not (amount > 0):
        return None
    unit = match.group(2) or "gb"
    if unit in ("tb", "t"):
        amount *= 1024
    elif unit in ("mb", "m"):
        amount /= 1024
    return amount
