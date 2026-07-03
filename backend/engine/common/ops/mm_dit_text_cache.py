"""Static text-conditioning cache helpers for joint MM-DiT families (FLUX.1, etc.)."""
from __future__ import annotations

from typing import Any


def text_cache_key(*parts: Any) -> tuple[Any, ...]:
    out: list[Any] = []
    for p in parts:
        if p is None:
            out.append(None)
        elif hasattr(p, "shape"):
            out.append(tuple(int(x) for x in p.shape))
            out.append(id(p))
        else:
            out.append(p)
    return tuple(out)


def invalidate_mm_dit_text_cache(model: Any) -> None:
    for attr in (
        "_mm_text_cache_key",
        "_mm_cached_encoder_hidden",
        "_mm_cached_pooled_contrib",
        "_mm_cached_rotary_emb",
        "_mm_cached_txt_len",
        "_mm_cached_hw",
    ):
        if hasattr(model, attr):
            setattr(model, attr, None)
