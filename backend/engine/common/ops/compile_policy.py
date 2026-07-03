"""MLX compile policy — schedule/chip-aware ``use_mlx_compile`` resolution."""
from __future__ import annotations

import platform
import subprocess
from typing import Any


def _apple_chip_tier() -> str:
    """Return coarse Apple Silicon tier: base | pro | max | unknown."""
    try:
        out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return "unknown"
    low = out.lower()
    if "ultra" in low or " max" in low:
        return "max"
    if " pro" in low:
        return "pro"
    if "m1" in low or "m2" in low or "m3" in low or "m4" in low or "m5" in low:
        return "base"
    return "unknown"


def teacache_blocks_compile(mode: str | None) -> bool:
    m = str(mode or "none").strip().lower()
    if m in ("", "none", "off", "false", "0", "auto"):
        return False
    return True


def resolve_use_mlx_compile(
    config: Any,
    *,
    num_steps: int,
    backend: str | None,
    teacache_mode: str | None = None,
) -> bool:
    """Resolve whether DiT forward should use ``ctx.compile`` for this run."""
    if backend != "mlx":
        return False
    if teacache_blocks_compile(teacache_mode):
        return False

    explicit = getattr(config, "use_mlx_compile", False)
    step_distill = getattr(config, "use_mlx_compile_step_distill", False)
    auto = getattr(config, "use_mlx_compile_auto", True)

    if num_steps <= 8 and step_distill:
        return True
    if explicit:
        return True
    if not auto:
        return False

    # Auto: prefer compile on longer schedules where dispatch overhead matters.
    if num_steps <= 8:
        return bool(step_distill)
    tier = _apple_chip_tier()
    if tier == "max":
        # M-series Max/Ultra: full-graph compile can fight dynamic gates; still OK without TeaCache.
        return True
    if tier in ("pro", "base"):
        return True
    return platform.machine() == "arm64"
