"""LeMiCa step schedule — Z-Image-Turbo training-free step cache (MLX)."""
from __future__ import annotations

# Ref: z-image-turbo-mlx / UnicomAI LeMiCa
LEMICA_SCHEDULES: dict[int, dict[str, tuple[int, ...]]] = {
    8: {
        "slow": (0, 1, 2, 3, 5, 6, 7),
        "medium": (0, 1, 2, 4, 5, 7),
        "fast": (0, 1, 2, 5, 7),
    },
    9: {
        "slow": (0, 1, 2, 3, 5, 7, 8),
        "medium": (0, 1, 2, 4, 6, 8),
        "fast": (0, 1, 2, 5, 8),
    },
    28: {
        "slow": tuple(range(22)) + (24, 26, 27),
        "medium": tuple(range(20)) + (22, 24, 26, 27),
        "fast": tuple(range(18)) + (20, 22, 25, 27),
    },
}


def normalize_lemica_mode(mode: str | None) -> str:
    return str(mode or "none").strip().lower()


def lemica_enabled(mode: str | None) -> bool:
    m = normalize_lemica_mode(mode)
    return m not in ("", "none", "off")


def lemica_compute_steps(mode: str, num_steps: int) -> tuple[bool, ...] | None:
    m = normalize_lemica_mode(mode)
    if not lemica_enabled(m):
        return None
    table = LEMICA_SCHEDULES.get(int(num_steps))
    if table is None:
        for key in sorted(LEMICA_SCHEDULES.keys(), reverse=True):
            if num_steps <= key:
                table = LEMICA_SCHEDULES[key]
                break
    if table is None:
        return None
    steps = table.get(m)
    if steps is None:
        raise RuntimeError(f"unknown lemica_mode={m!r} for num_steps={num_steps}")
    step_set = set(steps)
    return tuple(i in step_set for i in range(num_steps))
