"""L1 output integrity gate — file exists and is a readable image."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class IntegrityResult:
    ok: bool
    reason: str = ""
    width: int = 0
    height: int = 0
    bytes: int = 0


def check_output_image_integrity(
    path: str | Path,
    *,
    min_bytes: int = 4096,
    min_side: int = 64,
    expected_width: int | None = None,
    expected_height: int | None = None,
    size_tolerance: int = 8,
    min_channel_std: float = 8.0,
) -> IntegrityResult:
    path = Path(path)
    if not path.is_file():
        return IntegrityResult(False, "missing_file")
    size = path.stat().st_size
    if size < min_bytes:
        return IntegrityResult(False, f"tiny_file(bytes={size})", bytes=size)

    try:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            arr = np.asarray(rgb, dtype=np.uint8)
    except Exception as exc:
        return IntegrityResult(False, f"load_error:{exc!r}", bytes=size)

    if arr.ndim != 3 or arr.shape[-1] != 3:
        return IntegrityResult(False, "invalid_channels", bytes=size)

    h, w = int(arr.shape[0]), int(arr.shape[1])
    if w < min_side or h < min_side:
        return IntegrityResult(False, f"too_small({w}x{h})", width=w, height=h, bytes=size)

    if expected_width is not None and abs(w - expected_width) > size_tolerance:
        return IntegrityResult(
            False,
            f"width_mismatch({w}!={expected_width})",
            width=w,
            height=h,
            bytes=size,
        )
    if expected_height is not None and abs(h - expected_height) > size_tolerance:
        return IntegrityResult(
            False,
            f"height_mismatch({h}!={expected_height})",
            width=w,
            height=h,
            bytes=size,
        )

    if arr.size > 0:
        flat = arr.reshape(-1, 3)
        if np.all(flat == 0) or np.all(flat == 255):
            return IntegrityResult(False, "degenerate_solid", width=w, height=h, bytes=size)
        channel_std = float(arr.reshape(-1, 3).astype(np.float32).std(axis=0).mean())
        if channel_std < min_channel_std:
            return IntegrityResult(
                False,
                f"low_variance(std={channel_std:.1f})",
                width=w,
                height=h,
                bytes=size,
            )

    return IntegrityResult(True, "", width=w, height=h, bytes=size)
