"""Backbone transformer for HeartMuLa (LLaMA-3B)."""

from typing import Optional, Tuple, List

import mlx.core as mx
import mlx.nn as nn

from backend.engine.families.heartmula.mlx.nn.transformer import (
    RMSNorm,
    build_llama_transformer_layers,
    forward_llama_transformer_stack,
    llama_stack_from_config,
    setup_llama_kv_cache,
)
from backend.engine.families.heartmula.mlx.nn.kv_cache import KVCache


class HeartMuLaBackbone(nn.Module):
    """Backbone transformer for HeartMuLa.

    This is a LLaMA-style transformer that processes the combined
    text and audio sequence to predict the first codebook (codebook 0).
    """

    def __init__(
        self,
        dim: int = 3072,
        n_heads: int = 24,
        n_kv_heads: int = 8,
        n_layers: int = 28,
        hidden_dim: int = 8192,
        max_seq_len: int = 8192,
        norm_eps: float = 1e-5,
        rope_base: float = 500000.0,
    ):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_layers = n_layers
        self.head_dim = dim // n_heads

        self.layers = build_llama_transformer_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            norm_eps=norm_eps,
            rope_base=rope_base,
        )
        self.norm = RMSNorm(dim, eps=norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache | List] = None,
    ) -> Tuple[mx.array, KVCache | List]:
        return forward_llama_transformer_stack(
            layers=self.layers,
            norm=self.norm,
            hidden_states=hidden_states,
            mask=mask,
            cache=cache,
        )

    def setup_cache(self, batch_size: int, max_seq_len: int) -> KVCache:
        return setup_llama_kv_cache(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            n_layers=self.n_layers,
        )

    @classmethod
    def from_config(cls, config: dict) -> "HeartMuLaBackbone":
        return llama_stack_from_config(cls, config)
