"""Link Wan base encoders into Bernini-R install directories."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from backend.services.wan_distill_shared import link_wan_distill_shared_assets

_BERNINI_ENCODER_VARIANT: dict[str, str] = {
    "bernini_r_14b": "turbo_i2v_720p",
    "bernini_r_1.3b": "turbo_t2v_480p_1.3b",
}


def link_bernini_shared_assets(
    *,
    bernini_root: Path,
    variant: str,
    resolve_local_path: Callable[[str, str], Path],
) -> None:
    """Symlink Wan shared encoders into Bernini install root."""
    mapped = _BERNINI_ENCODER_VARIANT.get(str(variant))
    if mapped is None:
        known = ", ".join(sorted(_BERNINI_ENCODER_VARIANT))
        raise RuntimeError(f"Unknown bernini_variant {variant!r} (known: {known}).")
    link_wan_distill_shared_assets(
        distill_root=bernini_root,
        variant=mapped,
        resolve_local_path=resolve_local_path,
    )
