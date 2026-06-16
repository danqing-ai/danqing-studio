"""Z-Image Fun ControlNet Union — MLX control branches (hint injection)."""
from __future__ import annotations

from typing import Any

from backend.engine.config.model_configs import ZImageConfig
from backend.engine.families.z_image.transformer_mlx import (
    FeedForward,
    ZImageAttention,
    ZImageContextBlock,
    ZImageTransformerBlock,
)
from backend.engine.runtime._base import RuntimeContext
from backend.engine.common.ops.norm import apply_scale_shift, unpack_modulation_4way

CONTROL_LAYER_PLACES: tuple[int, ...] = tuple(range(0, 30, 2))
CONTROL_REFINER_PLACES: tuple[int, ...] = (0, 1)
CONTROL_IN_CHANNELS: int = 33


class ZImageControlBlock:
    """Control branch block — accumulates hints for main DiT layers."""

    def __init__(self, block_id: int, dim: int, n_heads: int, ctx: RuntimeContext, *, norm_eps: float = 1e-5):
        nn = ctx
        self.block_id = block_id
        self.dim = dim
        self.ctx = ctx
        self.before_proj = nn.Linear(dim, dim, bias=True) if block_id == 0 else None
        self.after_proj = nn.Linear(dim, dim, bias=True)
        self.attention = ZImageAttention(dim, n_heads, ctx, qk_norm=True, eps=norm_eps)
        self.feed_forward = FeedForward(dim, int(dim / 3 * 8), ctx)
        self.attn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.attn_norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.adaLN_modulation = [nn.Linear(min(dim, 256), 4 * dim, bias=True)]

    def _transformer_forward(self, x, attn_mask, freqs_cis, t_emb):
        ctx = self.ctx
        modulation = ctx.reshape(self.adaLN_modulation[0](t_emb), (-1, 1, 4 * self.dim))
        scale_msa, gate_msa, scale_mlp, gate_mlp = unpack_modulation_4way(modulation)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp
        gate_msa = ctx.tanh(gate_msa)
        gate_mlp = ctx.tanh(gate_mlp)

        normed = self.attn_norm1(x)
        attn_out = self.attention.forward(
            apply_scale_shift(normed, scale_msa, 0.0, add_one=False),
            attention_mask=attn_mask,
            freqs_cis=freqs_cis,
        )
        x = x + gate_msa * self.attn_norm2(attn_out)

        normed = self.ffn_norm1(x)
        ffn_out = self.feed_forward.forward(apply_scale_shift(normed, scale_mlp, 0.0, add_one=False))
        x = x + gate_mlp * self.ffn_norm2(ffn_out)
        return x

    def forward(self, c, x_main, attn_mask, freqs_cis, t_emb):
        ctx = self.ctx
        if self.block_id == 0:
            if self.before_proj is None:
                raise RuntimeError("ZImageControlBlock block_id=0 requires before_proj")
            control = self.before_proj(c) + x_main
            parts: list[Any] = []
        else:
            num_hints = int(c.shape[0])
            parts = [c[i] for i in range(num_hints - 1)]
            control = c[num_hints - 1]

        control = self._transformer_forward(control, attn_mask, freqs_cis, t_emb)
        c_skip = self.after_proj(control)
        parts.append(c_skip)
        parts.append(control)
        return ctx.stack(parts, axis=0)


class ZImageControlRuntime:
    """Optional control modules attached to ``ZImageDiTMLX`` at inference time."""

    def __init__(self, config: ZImageConfig, ctx: RuntimeContext):
        self.config = config
        self.ctx = ctx
        dim = config.dim
        n_heads = config.num_heads
        norm_eps = config.norm_eps
        patch_size = config.patch_size
        embed_in = patch_size * patch_size * 1 * CONTROL_IN_CHANNELS
        nn = ctx
        self.control_x_embedder = nn.Linear(embed_in, dim, bias=True)
        self.control_noise_refiner = [
            ZImageControlBlock(i, dim, n_heads, ctx, norm_eps=norm_eps)
            for i in range(len(CONTROL_REFINER_PLACES))
        ]
        self.control_layers = [
            ZImageControlBlock(i, dim, n_heads, ctx, norm_eps=norm_eps)
            for i in range(len(CONTROL_LAYER_PLACES))
        ]
        self._layer_place_to_idx = {place: idx for idx, place in enumerate(CONTROL_LAYER_PLACES)}
        self._refiner_place_to_idx = {place: idx for idx, place in enumerate(CONTROL_REFINER_PLACES)}
        self.context_scale: float = 0.75
        self._param_map: dict[str, Any] = {}
        self._build_param_map()

    def _build_param_map(self) -> None:
        self._param_map["control_x_embedder.weight"] = self.control_x_embedder.weight
        if getattr(self.control_x_embedder, "bias", None) is not None:
            self._param_map["control_x_embedder.bias"] = self.control_x_embedder.bias
        for i, block in enumerate(self.control_noise_refiner):
            prefix = f"control_noise_refiner.{i}"
            if block.before_proj is not None:
                self._param_map[f"{prefix}.before_proj.weight"] = block.before_proj.weight
                if block.before_proj.bias is not None:
                    self._param_map[f"{prefix}.before_proj.bias"] = block.before_proj.bias
            self._param_map[f"{prefix}.after_proj.weight"] = block.after_proj.weight
            if block.after_proj.bias is not None:
                self._param_map[f"{prefix}.after_proj.bias"] = block.after_proj.bias
            self._bind_block(prefix, block)
        for i, block in enumerate(self.control_layers):
            prefix = f"control_layers.{i}"
            if block.before_proj is not None:
                self._param_map[f"{prefix}.before_proj.weight"] = block.before_proj.weight
                if block.before_proj.bias is not None:
                    self._param_map[f"{prefix}.before_proj.bias"] = block.before_proj.bias
            self._param_map[f"{prefix}.after_proj.weight"] = block.after_proj.weight
            if block.after_proj.bias is not None:
                self._param_map[f"{prefix}.after_proj.bias"] = block.after_proj.bias
            self._bind_block(prefix, block)

    def _bind_block(self, prefix: str, block: ZImageControlBlock) -> None:
        attn = block.attention
        self._param_map[f"{prefix}.attention.to_q.weight"] = attn.to_q.weight
        self._param_map[f"{prefix}.attention.to_k.weight"] = attn.to_k.weight
        self._param_map[f"{prefix}.attention.to_v.weight"] = attn.to_v.weight
        self._param_map[f"{prefix}.attention.to_out.weight"] = attn.to_out.weight
        if attn.norm_q is not None:
            self._param_map[f"{prefix}.attention.norm_q.weight"] = attn.norm_q.weight
            self._param_map[f"{prefix}.attention.norm_k.weight"] = attn.norm_k.weight
        self._param_map[f"{prefix}.attention_norm1.weight"] = block.attn_norm1.weight
        self._param_map[f"{prefix}.attention_norm2.weight"] = block.attn_norm2.weight
        self._param_map[f"{prefix}.ffn_norm1.weight"] = block.ffn_norm1.weight
        self._param_map[f"{prefix}.ffn_norm2.weight"] = block.ffn_norm2.weight
        self._param_map[f"{prefix}.feed_forward.w1.weight"] = block.feed_forward.w1.weight
        self._param_map[f"{prefix}.feed_forward.w2.weight"] = block.feed_forward.w2.weight
        self._param_map[f"{prefix}.feed_forward.w3.weight"] = block.feed_forward.w3.weight
        self._param_map[f"{prefix}.adaLN_modulation.0.weight"] = block.adaLN_modulation[0].weight
        if block.adaLN_modulation[0].bias is not None:
            self._param_map[f"{prefix}.adaLN_modulation.0.bias"] = block.adaLN_modulation[0].bias

    def load_control_weights(self, flat: dict[str, Any], *, on_log: Any = None) -> None:
        missing = [k for k in self._param_map if k not in flat]
        if missing:
            raise RuntimeError(
                f"Z-Image controlnet weights missing {len(missing)} keys "
                f"(first: {missing[:5]!r}); wrong bundle or corrupt safetensors"
            )
        for key, param in self._param_map.items():
            param.update(flat[key])
        if on_log:
            on_log("info", f"z_image controlnet loaded {len(flat)} tensors")

    def embed_control_context(self, control_nchw, t_emb, *, x_pad_token, img_pad_mask, img_freqs):
        ctx = self.ctx
        # control_nchw: [C=33, F=1, H, W]
        pH = pW = self.config.patch_size
        pF = 1
        c, f, h, w = control_nchw.shape
        f_tok, h_tok, w_tok = f // pF, h // pH, w // pW
        tokens = f_tok * h_tok * w_tok
        img = ctx.reshape(control_nchw, (c, f_tok, pF, h_tok, pH, w_tok, pW))
        img = ctx.permute(img, (1, 3, 5, 2, 4, 6, 0))
        img = ctx.reshape(img, (tokens, pF * pH * pW * c))
        from backend.engine.common.ops.embeddings import pad_tail_with_last as _pad_tail_with_last

        pad_len = int(img_pad_mask.shape[0]) - int(img.shape[0])
        if pad_len > 0:
            img = _pad_tail_with_last(ctx, img, pad_len)
        embed = self.control_x_embedder(img)
        embed = ctx.reshape(embed, (1, embed.shape[0], embed.shape[1]))
        from backend.engine.common.ops.embeddings import apply_pad_token as _apply_pad_token

        embed = _apply_pad_token(ctx, embed[0], img_pad_mask, x_pad_token)
        embed = ctx.reshape(embed, (1, embed.shape[0], embed.shape[1]))
        return embed

    def forward_control_refiner(self, noise_stream, control_embed, img_freqs, t_emb):
        c = control_embed
        for block in self.control_noise_refiner:
            c = block.forward(c, noise_stream, None, img_freqs, t_emb)
        num_hints = int(c.shape[0]) - 1
        return [c[i] for i in range(num_hints)], c[num_hints - 1]

    def forward_control_layers(self, unified, refined_control, cap_stream, unified_freqs, t_emb):
        ctx = self.ctx
        control_unified = ctx.concat([refined_control, cap_stream], axis=1)
        c = control_unified
        for block in self.control_layers:
            c = block.forward(c, unified, None, unified_freqs, t_emb)
        num_hints = int(c.shape[0]) - 1
        return [c[i] for i in range(num_hints)]

    def hint_for_refiner_layer(self, layer_idx: int, refiner_hints: list[Any] | None) -> Any | None:
        if not refiner_hints:
            return None
        idx = self._refiner_place_to_idx.get(layer_idx)
        if idx is None or idx >= len(refiner_hints):
            return None
        return refiner_hints[idx]

    def hint_for_main_layer(self, layer_idx: int, layer_hints: list[Any] | None) -> Any | None:
        if not layer_hints:
            return None
        idx = self._layer_place_to_idx.get(layer_idx)
        if idx is None or idx >= len(layer_hints):
            return None
        return layer_hints[idx]


def apply_control_hint(x, hint, context_scale: float, ctx: RuntimeContext):
    if hint is None:
        return x
    return x + hint * float(context_scale)


def forward_layer_with_hint(layer: ZImageTransformerBlock, x, attn_mask, freqs_cis, t_emb, hint, context_scale, ctx):
    x = layer.forward(x, attn_mask, freqs_cis, t_emb)
    return apply_control_hint(x, hint, context_scale, ctx)
