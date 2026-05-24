"""HeartMuLa - Music Language Model."""

from functools import partial
from typing import Optional, Tuple, List, Union
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from backend.engine.common.mlx_runtime_fallback import (
    load_weights_dict,
    run_eval,
)
from backend.engine.families.heartmula.mlx.heartmula.configuration import HeartMuLaConfig
from backend.engine.families.heartmula.mlx.heartmula.backbone import HeartMuLaBackbone
from backend.engine.families.heartmula.mlx.heartmula.decoder import HeartMuLaDecoder
from backend.engine.families.heartmula.mlx.nn.kv_cache import KVCache


def _run_eval(*values) -> None:
    run_eval(None, *values)


# Compiled top-k sampling for faster inference
@mx.compile
def _sample_topk_compiled(logits: mx.array, topk: int, temperature: float) -> mx.array:
    """Compiled top-k sampling."""
    logits = logits / temperature
    topk = min(topk, logits.shape[-1])
    sorted_indices = mx.argsort(-logits, axis=-1)
    topk_indices = sorted_indices[:, :topk]
    topk_logits = mx.take_along_axis(logits, topk_indices, axis=-1)
    mask = mx.full(logits.shape, float("-inf"))
    mask = mx.put_along_axis(mask, topk_indices, topk_logits, axis=-1)
    log_probs = mx.log(mx.softmax(mask, axis=-1) + 1e-10)
    samples = mx.random.categorical(log_probs)
    return samples[:, None]


def sample_topk(logits: mx.array, topk: int, temperature: float) -> mx.array:
    """Sample from logits using top-k sampling (uses compiled version)."""
    return _sample_topk_compiled(logits, topk, temperature)


class HeartMuLa(nn.Module):
    """HeartMuLa: Music Language Model.

    Matches the PyTorch implementation exactly:
    - Combined token format: (batch, seq_len, num_codebooks+1) where last dim is text
    - Embeddings are summed across codebook dimension, not concatenated
    - CFG via batch doubling (conditional + unconditional in same batch)
    - Per-codebook audio_head weights
    """

    def __init__(self, config: HeartMuLaConfig):
        super().__init__()
        self.config = config

        backbone_cfg = config.backbone_config
        decoder_cfg = config.decoder_config

        self.backbone_dim = backbone_cfg["dim"]
        self.decoder_dim = decoder_cfg["dim"]
        self.num_codebooks = config.audio_num_codebooks
        self.audio_vocab_size = config.audio_vocab_size

        # Text embeddings
        self.text_embeddings = nn.Embedding(config.text_vocab_size, self.backbone_dim)

        # Audio embeddings: single embedding table for all codebooks
        # Size: (audio_vocab_size * num_codebooks, dim)
        # Access: token + codebook_idx * audio_vocab_size
        self.audio_embeddings = nn.Embedding(
            config.audio_vocab_size * config.audio_num_codebooks,
            self.backbone_dim,
        )

        # Unconditional text embedding for CFG
        self.unconditional_text_embedding = nn.Embedding(1, self.backbone_dim)

        # Backbone transformer (LLaMA-3B)
        self.backbone = HeartMuLaBackbone(
            dim=backbone_cfg["dim"],
            n_heads=backbone_cfg["n_heads"],
            n_kv_heads=backbone_cfg["n_kv_heads"],
            n_layers=backbone_cfg["n_layers"],
            hidden_dim=backbone_cfg["hidden_dim"],
            max_seq_len=config.max_seq_len,
            norm_eps=backbone_cfg["norm_eps"],
            rope_base=backbone_cfg["rope_base"],
        )

        # Projection from backbone to decoder
        self.projection = nn.Linear(self.backbone_dim, self.decoder_dim, bias=False)

        # Local decoder (LLaMA-300M)
        self.decoder = HeartMuLaDecoder(
            dim=decoder_cfg["dim"],
            n_heads=decoder_cfg["n_heads"],
            n_kv_heads=decoder_cfg["n_kv_heads"],
            n_layers=decoder_cfg["n_layers"],
            hidden_dim=decoder_cfg["hidden_dim"],
            max_seq_len=config.audio_num_codebooks,  # Decoder only sees num_codebooks positions
            norm_eps=decoder_cfg["norm_eps"],
            rope_base=decoder_cfg["rope_base"],
        )

        # Prediction heads
        # Codebook 0 is predicted by the backbone
        self.codebook0_head = nn.Linear(self.backbone_dim, config.audio_vocab_size, bias=False)

        # Codebooks 1-7 are predicted by the decoder
        # Shape: (num_codebooks-1, decoder_dim, audio_vocab_size)
        # In MLX, we store this as a flattened Linear and index into output
        self.audio_head = nn.Linear(
            self.decoder_dim,
            config.audio_vocab_size * (config.audio_num_codebooks - 1),
            bias=False,
        )

        # MuQ linear for audio conditioning
        self.muq_linear = nn.Linear(config.muq_dim, self.backbone_dim, bias=True)

        self._backbone_cache: KVCache | None = None
        self._decoder_cache: KVCache | None = None

    def setup_caches(self, batch_size: int, max_seq_len: int) -> None:
        """Pre-allocate KV caches for backbone (AR) and decoder (per-frame codebooks)."""
        self._backbone_cache = self.backbone.setup_cache(batch_size, max_seq_len)
        dec_max = int(self.num_codebooks) + 2
        self._decoder_cache = self.decoder.setup_cache(batch_size, dec_max)

    def reset_caches(self) -> None:
        if self._backbone_cache is not None:
            self._backbone_cache.reset()
        if self._decoder_cache is not None:
            self._decoder_cache.reset()

    def _codebook_logits(self, codebook_idx: int, hidden: mx.array) -> mx.array:
        """Logits for one decoder codebook (avoids full ``audio_head`` matmul)."""
        start = (codebook_idx - 1) * self.audio_vocab_size
        end = codebook_idx * self.audio_vocab_size
        w_slice = self.audio_head.weight[start:end]
        return hidden @ w_slice.T

    def _inject_muq(
        self,
        h: mx.array,
        continuous_segments: mx.array,
        starts: List[int],
        uncond_mask: Optional[mx.array],
    ) -> mx.array:
        continuous_segments = self.muq_linear(continuous_segments)
        if uncond_mask is not None:
            uncond_embed = self.unconditional_text_embedding(mx.zeros((1,), dtype=mx.int32))[0]
            mask_expanded = mx.broadcast_to(uncond_mask[:, None], continuous_segments.shape)
            continuous_segments = mx.where(
                mask_expanded,
                mx.broadcast_to(uncond_embed, continuous_segments.shape),
                continuous_segments,
            )
        rows = []
        for batch_idx, start in enumerate(starts):
            row = h[batch_idx]
            rows.append(
                mx.concatenate(
                    [row[:start], continuous_segments[batch_idx : batch_idx + 1], row[start + 1 :]],
                    axis=0,
                )
            )
        return mx.stack(rows, axis=0)

    def _embed_audio(self, codebook: int, tokens: mx.array) -> mx.array:
        """Embed audio tokens for a specific codebook.

        Args:
            codebook: Codebook index (0-7).
            tokens: Token IDs of shape (batch, seq_len) or (batch,).

        Returns:
            Embeddings of shape (batch, seq_len, dim) or (batch, dim).
        """
        return self.audio_embeddings(tokens + codebook * self.audio_vocab_size)

    def _embed_tokens(
        self,
        tokens: mx.array,
        uncond_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Embed combined tokens.

        Args:
            tokens: Combined tokens of shape (batch, seq_len, num_codebooks+1).
                    Last dimension is text, first num_codebooks are audio.
            uncond_mask: Boolean mask of shape (batch,) indicating unconditional samples.

        Returns:
            Embeddings of shape (batch, seq_len, num_codebooks+1, dim).
        """
        B, S, _ = tokens.shape

        # Text embeddings from last channel
        text_tokens = tokens[:, :, -1].astype(mx.int32)  # (B, S)
        text_embeds = self.text_embeddings(text_tokens)  # (B, S, dim)

        # Apply unconditional embedding for CFG
        if uncond_mask is not None:
            uncond_embed = self.unconditional_text_embedding(
                mx.zeros((1,), dtype=mx.int32)
            )  # (1, dim)
            # Expand mask: (B,) -> (B, 1, 1)
            mask_expanded = uncond_mask[:, None, None]
            # Replace text embeddings with unconditional for masked samples
            text_embeds = mx.where(
                mx.broadcast_to(mask_expanded, text_embeds.shape),
                uncond_embed,
                text_embeds,
            )

        text_embeds = text_embeds[:, :, None, :]  # (B, S, 1, dim)

        # Audio embeddings from first num_codebooks channels
        # tokens[:, :, :-1] shape: (B, S, num_codebooks)
        audio_tokens = tokens[:, :, :-1].astype(mx.int32)  # (B, S, num_codebooks)

        # Add codebook offsets: token + codebook_idx * vocab_size
        codebook_offsets = mx.arange(self.num_codebooks) * self.audio_vocab_size
        audio_tokens_offset = audio_tokens + codebook_offsets  # (B, S, num_codebooks)

        # Flatten for embedding lookup
        audio_tokens_flat = audio_tokens_offset.reshape(-1)  # (B * S * num_codebooks,)
        audio_embeds_flat = self.audio_embeddings(audio_tokens_flat)  # (B * S * num_codebooks, dim)
        audio_embeds = audio_embeds_flat.reshape(B, S, self.num_codebooks, -1)  # (B, S, num_codebooks, dim)

        # Concatenate: audio (num_codebooks) + text (1) -> (num_codebooks+1)
        embeds = mx.concatenate([audio_embeds, text_embeds], axis=2)  # (B, S, num_codebooks+1, dim)

        return embeds

    def generate_frame(
        self,
        tokens: mx.array,
        tokens_mask: mx.array,
        input_pos: mx.array,
        temperature: float,
        topk: int,
        cfg_scale: float,
        continuous_segments: Optional[mx.array] = None,
        starts: Optional[List[int]] = None,
    ) -> mx.array:
        """Generate a single audio frame (all codebooks).

        Args:
            tokens: Combined tokens of shape (batch, seq_len, num_codebooks+1).
            tokens_mask: Mask of shape (batch, seq_len, num_codebooks+1).
            input_pos: Position indices of shape (batch, seq_len).
            temperature: Sampling temperature.
            topk: Top-k sampling parameter.
            cfg_scale: Classifier-free guidance scale.
            continuous_segments: Optional MuQ embeddings.
            starts: Optional start positions for MuQ injection.

        Returns:
            Generated codes of shape (batch, num_codebooks).
            For CFG (batch=2), returns codes for conditional sample only (first half).
        """
        b, s, _ = tokens.shape

        # Determine unconditional mask for CFG
        uncond_mask = None
        if cfg_scale > 1.0 and b > 1:
            actual_B = b // 2
            # First half: conditional (False), Second half: unconditional (True)
            uncond_mask = mx.concatenate([
                mx.zeros((actual_B,), dtype=mx.bool_),
                mx.ones((actual_B,), dtype=mx.bool_),
            ])

        # Get embeddings: (B, S, num_codebooks+1, dim)
        embeds = self._embed_tokens(tokens, uncond_mask=uncond_mask)

        # Apply mask and sum across codebook dimension
        # tokens_mask: (B, S, num_codebooks+1) -> (B, S, num_codebooks+1, 1)
        masked_embeds = embeds * tokens_mask[:, :, :, None]
        h = mx.sum(masked_embeds, axis=2)  # (B, S, dim)

        if continuous_segments is not None and starts is not None:
            h = self._inject_muq(h, continuous_segments, starts, uncond_mask)

        # Run backbone
        h, self._backbone_cache = self.backbone(h, cache=self._backbone_cache)

        # Get last position for prediction
        last_h = h[:, -1, :]  # (B, dim)

        # Predict codebook 0
        c0_logits = self.codebook0_head(last_h)  # (B, vocab_size)

        # Apply CFG for codebook 0
        if cfg_scale > 1.0 and b > 1 and (b % 2 == 0):
            actual_B = b // 2
            cond_logits = c0_logits[:actual_B, :]
            uncond_logits = c0_logits[actual_B:, :]
            guided_logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
            c0_sample = sample_topk(guided_logits, topk, temperature)
            # Repeat for both branches to keep cache aligned
            c0_sample = mx.concatenate([c0_sample, c0_sample], axis=0)
        else:
            c0_sample = sample_topk(c0_logits, topk, temperature)

        c0_sample = c0_sample[:, 0]  # (B,)

        # Get codebook 0 embedding
        c0_embed = self._embed_audio(0, c0_sample)  # (B, dim)

        if self._decoder_cache is not None:
            self._decoder_cache.reset()

        # Initialize decoder with backbone output + c0 embedding
        curr_h = mx.stack([last_h, c0_embed], axis=1)  # (B, 2, dim)
        curr_sample = c0_sample[:, None]  # (B, 1)

        # Generate codebooks 1-7
        for i in range(1, self.num_codebooks):
            # Project and run decoder
            projected = self.projection(curr_h)
            decoder_h, self._decoder_cache = self.decoder(projected, cache=self._decoder_cache)

            ci_logits = self._codebook_logits(i, decoder_h[:, -1, :])

            # Apply CFG
            if cfg_scale > 1.0 and b > 1 and (b % 2 == 0):
                actual_B = b // 2
                cond_ci = ci_logits[:actual_B, :]
                uncond_ci = ci_logits[actual_B:, :]
                guided_ci = uncond_ci + (cond_ci - uncond_ci) * cfg_scale
                ci_sample = sample_topk(guided_ci, topk, temperature)
                ci_sample = mx.concatenate([ci_sample, ci_sample], axis=0)
            else:
                ci_sample = sample_topk(ci_logits, topk, temperature)

            ci_sample = ci_sample[:, 0]  # (B,)

            # Get embedding for next iteration
            ci_embed = self._embed_audio(i, ci_sample)  # (B, dim)
            curr_h = ci_embed[:, None, :]  # (B, 1, dim)
            curr_sample = mx.concatenate([curr_sample, ci_sample[:, None]], axis=1)

        return curr_sample  # (B, num_codebooks)

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        dtype: mx.Dtype = mx.bfloat16,
    ) -> "HeartMuLa":
        """Load a pretrained HeartMuLa model."""
        path = Path(path)

        # Load config
        config = HeartMuLaConfig.from_pretrained(path)

        # Create model
        model = cls(config)

        # Load weights
        weights_path = path / "model.safetensors"
        if weights_path.exists():
            weights = load_weights_dict(None, str(weights_path))
            weights = {k: v.astype(dtype) for k, v in weights.items()}
            model.load_weights(list(weights.items()))
            _run_eval(model.parameters())

        return model
