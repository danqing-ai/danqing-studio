"""Cross-run text embedding cache (B15-style) for encode phase."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


def _cache_key(
    *,
    encoder_id: str,
    prompt: str,
    negative_prompt: str | None,
    guidance: float,
) -> str:
    neg = negative_prompt or ""
    raw = f"{encoder_id}\0{prompt}\0{neg}\0{guidance:.6g}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class TextEmbeddingCache:
    """In-memory prompt → embedding tuple cache (training-free, encode phase only)."""

    max_entries: int = 32
    _store: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)

    def get(
        self,
        *,
        encoder_id: str,
        prompt: str,
        negative_prompt: str | None,
        guidance: float,
    ) -> tuple[Any, ...] | None:
        key = _cache_key(
            encoder_id=encoder_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance=guidance,
        )
        return self._store.get(key)

    def put(
        self,
        *,
        encoder_id: str,
        prompt: str,
        negative_prompt: str | None,
        guidance: float,
        value: tuple[Any, ...],
    ) -> None:
        key = _cache_key(
            encoder_id=encoder_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance=guidance,
        )
        if key in self._store:
            return
        if len(self._order) >= max(1, int(self.max_entries)):
            oldest = self._order.pop(0)
            self._store.pop(oldest, None)
        self._store[key] = value
        self._order.append(key)

    def clear(self) -> None:
        self._store.clear()
        self._order.clear()


_GLOBAL_TEXT_EMBEDDING_CACHE = TextEmbeddingCache()


def global_text_embedding_cache() -> TextEmbeddingCache:
    return _GLOBAL_TEXT_EMBEDDING_CACHE
