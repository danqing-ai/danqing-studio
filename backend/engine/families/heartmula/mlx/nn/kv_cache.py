"""KV cache for autoregressive generation (pre-allocated, no per-step realloc)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import mlx.core as mx

CacheState = Union[Tuple[mx.array, mx.array], "KVLayerCache"]


@dataclass
class KVLayerCache:
    """Single-layer K/V buffer with running length (in-place append)."""

    k: mx.array
    v: mx.array
    length: int = 0

    @classmethod
    def create(
        cls,
        *,
        batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> "KVLayerCache":
        shape = (batch_size, max_seq_len, n_kv_heads, head_dim)
        return cls(k=mx.zeros(shape, dtype=dtype), v=mx.zeros(shape, dtype=dtype), length=0)

    def as_tuple(self) -> Optional[Tuple[mx.array, mx.array]]:
        if self.length <= 0:
            return None
        return self.k[:, : self.length], self.v[:, : self.length]

    def append(
        self,
        k_new: mx.array,
        v_new: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        new_len = int(k_new.shape[1])
        start = self.length
        end = start + new_len
        max_len = int(self.k.shape[1])
        if end > max_len:
            raise RuntimeError(
                f"KV cache overflow: need {end} positions, max_seq_len={max_len}"
            )
        self.k[:, start:end] = k_new
        self.v[:, start:end] = v_new
        self.length = end
        return self.k[:, :end], self.v[:, :end]

    def reset(self) -> None:
        self.length = 0


class KVCache:
    """Per-layer pre-allocated caches for a transformer stack."""

    def __init__(
        self,
        *,
        batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        n_layers: int,
        dtype: mx.Dtype = mx.bfloat16,
    ):
        self.max_seq_len = max_seq_len
        self.layers: List[KVLayerCache] = [
            KVLayerCache.create(
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                dtype=dtype,
            )
            for _ in range(n_layers)
        ]

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()

    def __getitem__(self, index: int) -> KVLayerCache:
        """Layer access (compat with legacy per-layer cache lists)."""
        return self.layers[index]

    def __len__(self) -> int:
        return len(self.layers)


# Legacy aliases kept for imports elsewhere
RotatingKVCache = KVCache
HierarchicalKVCache = KVCache
