"""Backbone transformer for HeartMuLa (LLaMA-3B)."""

from typing import Optional, Tuple, List

import mlx.core as mx
import mlx.nn as nn

from backend.engine.common.attention import build_causal_with_offset_bias
from backend.engine.families.heartmula.mlx.nn.transformer import (
    RMSNorm,
    LlamaTransformerBlock,
)
from backend.engine.families.heartmula.mlx.nn.kv_cache import KVCache, KVLayerCache


class HeartMuLaBackbone(nn.Module):
    """Backbone transformer for HeartMuLa.

    This is a LLaMA-style transformer that processes the combined
    text and audio sequence to predict the first codebook (codebook 0).

    Args:
        dim: Model dimension.
        n_heads: Number of attention heads.
        n_kv_heads: Number of key/value heads (for GQA).
        n_layers: Number of transformer layers.
        hidden_dim: MLP hidden dimension.
        max_seq_len: Maximum sequence length.
        norm_eps: Epsilon for RMSNorm.
        rope_base: Base for RoPE.
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

        # Transformer layers
        self.layers = [
            LlamaTransformerBlock(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                hidden_dim=hidden_dim,
                max_seq_len=max_seq_len,
                norm_eps=norm_eps,
                rope_base=rope_base,
            )
            for _ in range(n_layers)
        ]

        # Output normalization
        self.norm = RMSNorm(dim, eps=norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache | List] = None,
    ) -> Tuple[mx.array, KVCache | List]:
        """Forward pass through backbone.

        Args:
            hidden_states: Input embeddings of shape (batch, seq_len, dim).
            mask: Optional attention mask.
            cache: Optional list of KV caches for each layer.

        Returns:
            Tuple of (output, new_caches).
        """
        offset = 0
        if isinstance(cache, KVCache):
            offset = cache.layers[0].length
        elif cache is not None and cache[0] is not None:
            if isinstance(cache[0], KVLayerCache):
                offset = cache[0].length
            else:
                offset = cache[0][0].shape[1]

        # Create causal mask if not provided
        if mask is None:
            seq_len = hidden_states.shape[1]
            mask = build_causal_with_offset_bias(mx, seq_len, offset, dtype=mx.float32)

        # Process through layers
        new_caches = []
        x = hidden_states

        for i, layer in enumerate(self.layers):
            if isinstance(cache, KVCache):
                layer_cache = cache.layers[i]
            elif cache is not None:
                layer_cache = cache[i]
            else:
                layer_cache = None
            x, new_cache = layer(x, mask=mask, cache=layer_cache, offset=offset)
            new_caches.append(new_cache)

        # Final norm
        x = self.norm(x)

        if isinstance(cache, KVCache):
            return x, cache
        return x, new_caches

    def setup_cache(
        self,
        batch_size: int,
        max_seq_len: int,
    ) -> KVCache:
        """Setup KV cache for generation.

        Args:
            batch_size: Batch size.
            max_seq_len: Maximum sequence length.

        Returns:
            KVCache instance.
        """
        return KVCache(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            n_layers=self.n_layers,
        )

    @classmethod
    def from_config(cls, config: dict) -> "HeartMuLaBackbone":
        """Create backbone from configuration dictionary.

        Args:
            config: Configuration with dim, n_heads, etc.

        Returns:
            HeartMuLaBackbone instance.
        """
        return cls(
            dim=config["dim"],
            n_heads=config["n_heads"],
            n_kv_heads=config.get("n_kv_heads", config["n_heads"]),
            n_layers=config["n_layers"],
            hidden_dim=config.get("hidden_dim"),
            norm_eps=config.get("norm_eps", 1e-5),
            rope_base=config.get("rope_base", 10000.0),
        )
