"""Real-ESRGAN weight key remap (mlx-community body.N → body_N)."""
from __future__ import annotations

import re
from typing import Any


_BODY_IDX = re.compile(r"^body\.(\d+)\.(.*)$")


def remap_esrgan_weights(flat: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in flat.items():
        m = _BODY_IDX.match(key)
        if m:
            out[f"body_{m.group(1)}.{m.group(2)}"] = val
        else:
            out[key] = val
    return out
