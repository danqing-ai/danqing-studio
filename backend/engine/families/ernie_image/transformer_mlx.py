"""ERNIE-Image Transformer — 8B single-stream DiT (MLX).

Ported from diffusers ``ErnieImageTransformer2DModel`` / dgrauet/ernie-image-mlx.
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from backend.engine.common.model.base import TransformerBase
from backend.engine.common.ops.attention import scaled_dot_product_attention_bhsd_mx
from backend.engine.common.ops.embeddings import sinusoidal_timestep_proj
from backend.engine.families.ernie_image.weights import _squeeze_x_embedder_proj_weight
from backend.engine.config.model_configs import ErnieImageConfig
from backend.engine.runtime._base import RuntimeContext


def _axis_angles(positions: mx.array, dim: int, theta: float) -> mx.array:
    k = mx.arange(0, dim, 2, dtype=mx.float32)
    scale = k / dim
    omega = 1.0 / (theta**scale)
    return positions.astype(mx.float32)[..., None] * omega


class ErnieImageEmbedND3(nn.Module):
    def __init__(self, head_dim: int, theta: float, axes_dim: tuple[int, int, int]):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.axes_dim = axes_dim

    def __call__(self, ids: mx.array) -> mx.array:
        per_axis = [_axis_angles(ids[..., i], self.axes_dim[i], self.theta) for i in range(3)]
        emb = mx.concatenate(per_axis, axis=-1)
        emb = emb[:, :, None, :]
        emb = mx.stack([emb, emb], axis=-1)
        shape = emb.shape
        return emb.reshape(*shape[:-2], shape[-2] * shape[-1])


def _apply_rotary_emb(x: mx.array, freqs_cis: mx.array) -> mx.array:
    rot_dim = freqs_cis.shape[-1]
    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]
    cos_ = mx.cos(freqs_cis).astype(x.dtype)
    sin_ = mx.sin(freqs_cis).astype(x.dtype)
    d_half = rot_dim // 2
    x1 = x_rot[..., :d_half]
    x2 = x_rot[..., d_half:]
    x_rotated = mx.concatenate([-x2, x1], axis=-1)
    out_rot = x_rot * cos_ + x_rotated * sin_
    if x_pass.shape[-1] == 0:
        return out_rot
    return mx.concatenate([out_rot, x_pass], axis=-1)


class ErnieAttention(nn.Module):
    def __init__(self, cfg: ErnieImageConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        inner = cfg.hidden_size
        self.to_q = nn.Linear(inner, inner, bias=False)
        self.to_k = nn.Linear(inner, inner, bias=False)
        self.to_v = nn.Linear(inner, inner, bias=False)
        self.to_out_0 = nn.Linear(inner, inner, bias=False)
        if cfg.qk_norm:
            self.norm_q = nn.RMSNorm(self.head_dim, eps=cfg.eps)
            self.norm_k = nn.RMSNorm(self.head_dim, eps=cfg.eps)
        else:
            self.norm_q = None
            self.norm_k = None

    def __call__(
        self,
        x: mx.array,
        freqs_cis: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        b, n, _ = x.shape
        h, d = self.num_heads, self.head_dim
        q = self.to_q(x).reshape(b, n, h, d)
        k = self.to_k(x).reshape(b, n, h, d)
        v = self.to_v(x).reshape(b, n, h, d)
        if self.norm_q is not None:
            q = self.norm_q(q)
            k = self.norm_k(k)
        if freqs_cis is not None:
            q = _apply_rotary_emb(q, freqs_cis)
            k = _apply_rotary_emb(k, freqs_cis)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        scale = 1.0 / (d**0.5)
        out = scaled_dot_product_attention_bhsd_mx(mx, q, k, v, scale=scale, mask=attention_mask)
        out = out.transpose(0, 2, 1, 3).reshape(b, n, h * d)
        return self.to_out_0(out)


class ErnieFeedForward(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.linear_fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_fc2(self.up_proj(x) * nn.gelu(self.gate_proj(x)))


class ErnieImageSharedAdaLNBlock(nn.Module):
    def __init__(self, cfg: ErnieImageConfig):
        super().__init__()
        self.adaLN_sa_ln = nn.RMSNorm(cfg.hidden_size, eps=cfg.eps)
        self.self_attention = ErnieAttention(cfg)
        self.adaLN_mlp_ln = nn.RMSNorm(cfg.hidden_size, eps=cfg.eps)
        self.mlp = ErnieFeedForward(cfg.hidden_size, cfg.ffn_hidden_size)

    def __call__(
        self,
        x: mx.array,
        freqs_cis: mx.array,
        temb: tuple[mx.array, ...],
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = temb
        residual = x
        h = self.adaLN_sa_ln(x)
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attention(h, freqs_cis=freqs_cis, attention_mask=attention_mask)
        x = residual + gate_msa * h
        residual = x
        h = self.adaLN_mlp_ln(x)
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        return residual + gate_mlp * h


class ErnieImageAdaLNContinuous(nn.Module):
    def __init__(self, cfg: ErnieImageConfig):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.eps, affine=False)
        self.linear = nn.Linear(cfg.hidden_size, cfg.hidden_size * 2, bias=True)

    def __call__(self, x: mx.array, conditioning: mx.array) -> mx.array:
        scale, shift = mx.split(self.linear(conditioning), 2, axis=-1)
        x = self.norm(x)
        return x * (1 + scale[:, None, :]) + shift[:, None, :]


def _flatten_mlx_module_params(module: nn.Module, prefix: str, result: dict[str, Any]) -> None:
    """Flatten nested ``mlx.nn.Module.parameters()`` trees to dotted keys."""

    def _walk(node: Any, key: str) -> None:
        if isinstance(node, dict):
            for name, child in node.items():
                child_key = f"{key}.{name}" if key else name
                _walk(child, child_key)
            return
        if key:
            result[key] = node

    _walk(module.parameters(), prefix)


class ErnieImagePatchEmbedLinear(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, patch_size: int):
        super().__init__()
        if patch_size != 1:
            raise NotImplementedError(f"patch_size={patch_size} not supported")
        self.proj = nn.Linear(in_channels, hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        b, _c, h, w = x.shape
        x = x.transpose(0, 2, 3, 1).reshape(b, h * w, -1)
        return self.proj(x)


class SharedAdaLNModulation(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 6 * hidden_size, bias=True)

    def __call__(self, c: mx.array) -> mx.array:
        return self.linear(nn.silu(c))


class ErnieTimestepEmbedding(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(self, sample: mx.array) -> mx.array:
        x = self.linear_1(sample)
        x = nn.silu(x)
        return self.linear_2(x)


class _ErnieImageDiTCore(nn.Module):
    def __init__(self, cfg: ErnieImageConfig, ctx: RuntimeContext):
        super().__init__()
        self.cfg = cfg
        self.ctx = ctx
        self.x_embedder = ErnieImagePatchEmbedLinear(cfg.in_channels, cfg.hidden_size, cfg.patch_size)
        self.text_proj = (
            nn.Linear(cfg.text_in_dim, cfg.hidden_size, bias=False)
            if cfg.text_in_dim != cfg.hidden_size
            else None
        )
        self.time_embedding = ErnieTimestepEmbedding(cfg.hidden_size)
        self.pos_embed = ErnieImageEmbedND3(cfg.head_dim, cfg.rope_theta, cfg.rope_axes_dim)
        self.adaLN_modulation = SharedAdaLNModulation(cfg.hidden_size)
        self.layers = [ErnieImageSharedAdaLNBlock(cfg) for _ in range(cfg.num_layers)]
        self.final_norm = ErnieImageAdaLNContinuous(cfg)
        self.final_linear = nn.Linear(
            cfg.hidden_size,
            cfg.patch_size * cfg.patch_size * cfg.out_channels,
            bias=True,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        text_bth: mx.array,
        text_lens: mx.array,
    ) -> mx.array:
        cfg = self.cfg
        p = cfg.patch_size
        b, _c_in, h, w = hidden_states.shape
        hp, wp = h // p, w // p
        n_img = hp * wp

        img_bnh = self.x_embedder(hidden_states)
        text_bnh = self.text_proj(text_bth) if self.text_proj is not None else text_bth
        tmax = text_bnh.shape[1]
        x = mx.concatenate([img_bnh, text_bnh], axis=1)

        text_lens_f = text_lens.astype(mx.float32)
        text_lens_col = text_lens_f.reshape(b, 1, 1)
        text_lens_exp = mx.broadcast_to(text_lens_col, (b, n_img, 1))
        grid_y = mx.arange(hp, dtype=mx.float32)
        grid_x = mx.arange(wp, dtype=mx.float32)
        yy, xx = mx.meshgrid(grid_y, grid_x, indexing="ij")
        grid_yx = mx.stack([yy, xx], axis=-1).reshape(n_img, 2)[None, :, :]
        grid_yx_exp = mx.broadcast_to(grid_yx, (b, n_img, 2))
        image_ids = mx.concatenate([text_lens_exp, grid_yx_exp], axis=-1)

        if tmax > 0:
            text_pos = mx.arange(tmax, dtype=mx.float32).reshape(1, tmax, 1)
            text_pos = mx.broadcast_to(text_pos, (b, tmax, 1))
            text_zeros = mx.zeros((b, tmax, 2), dtype=mx.float32)
            text_ids = mx.concatenate([text_pos, text_zeros], axis=-1)
            ids_all = mx.concatenate([image_ids, text_ids], axis=1)
        else:
            ids_all = image_ids

        freqs_cis = self.pos_embed(ids_all)

        if tmax > 0:
            text_valid = mx.arange(tmax, dtype=mx.int32).reshape(1, tmax) < text_lens.astype(mx.int32).reshape(b, 1)
            img_ones = mx.ones((b, n_img), dtype=mx.bool_)
            attn_mask = mx.concatenate([img_ones, text_valid], axis=1)
        else:
            attn_mask = mx.ones((b, n_img), dtype=mx.bool_)
        attn_mask = attn_mask[:, None, None, :]

        t_emb = sinusoidal_timestep_proj(
            self.ctx,
            timestep.astype(mx.float32),
            cfg.hidden_size,
            flip_sin_to_cos=False,
            downscale_freq_shift=0.0,
        )
        c = self.time_embedding(t_emb.astype(x.dtype))
        six = mx.split(self.adaLN_modulation(c), 6, axis=-1)
        temb = tuple(t[:, None, :] for t in six)

        for layer in self.layers:
            x = layer(x, freqs_cis=freqs_cis, temb=temb, attention_mask=attn_mask)

        x = self.final_norm(x, c)
        patches = self.final_linear(x)[:, :n_img, :]
        out = patches.reshape(b, hp, wp, p, p, cfg.out_channels)
        out = out.transpose(0, 5, 1, 3, 2, 4)
        return out.reshape(b, cfg.out_channels, h, w)


class ErnieImageDiTMLX(TransformerBase):
    """ERNIE-Image DiT — ``[B,128,H/16,W/16]`` latents, Ministral-3 text cond."""

    def __init__(self, config: ErnieImageConfig | Any, ctx: RuntimeContext):
        self.config = config
        self.ctx = ctx
        if isinstance(config, ErnieImageConfig):
            cfg = config
        else:
            cfg = ErnieImageConfig(
                hidden_size=getattr(config, "hidden_size", 4096),
                num_heads=getattr(config, "num_heads", 32),
                num_layers=getattr(config, "num_layers", 36),
                ffn_hidden_size=getattr(config, "ffn_hidden_size", 12288),
                in_channels=getattr(config, "in_channels", 128),
                out_channels=getattr(config, "out_channels", 128),
                patch_size=getattr(config, "patch_size", 1),
                text_in_dim=getattr(config, "text_in_dim", 3072),
                rope_axes_dim=tuple(getattr(config, "rope_axes_dim", (32, 48, 48))),
                rope_theta=getattr(config, "rope_theta", 256.0),
                qk_norm=getattr(config, "qk_norm", True),
                eps=getattr(config, "eps", 1e-6),
            )
        self._cfg = cfg
        self._core = _ErnieImageDiTCore(cfg, ctx)
        self._build_param_map()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        """Map diffusers ERNIE-Image transformer keys to ``ErnieImageDiTMLX`` param map."""
        remapped: dict[str, Any] = {}
        for key, tensor in weights.items():
            new_key = key
            for prefix in ("transformer.", "model."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
            new_key = new_key.replace(".to_out.0.", ".to_out_0.")
            new_key = new_key.replace("adaLN_modulation.1.", "adaLN_modulation.linear.")
            if new_key.startswith("adaln_modulation."):
                new_key = "adaLN_modulation.linear." + new_key[len("adaln_modulation.") :]
            if new_key.startswith("time_proj."):
                continue
            if new_key == "x_embedder.proj.weight" and hasattr(tensor, "ndim") and int(tensor.ndim) == 4:
                tensor = _squeeze_x_embedder_proj_weight(tensor)
            remapped[new_key] = tensor
        return remapped

    def _build_param_map(self):
        self._param_map = {}
        core = self._core
        for attr in (
            "x_embedder",
            "time_embedding",
            "adaLN_modulation",
            "final_norm",
            "final_linear",
        ):
            _flatten_mlx_module_params(getattr(core, attr), attr, self._param_map)
        if core.text_proj is not None:
            _flatten_mlx_module_params(core.text_proj, "text_proj", self._param_map)
        for i, layer in enumerate(core.layers):
            _flatten_mlx_module_params(layer, f"layers.{i}", self._param_map)

    def load_weights(
        self,
        weights,
        strict=False,
        ctx=None,
        *,
        bundle_affine_bits=None,
        inference_mode=None,
    ):
        loaded, skipped = super().load_weights(
            weights,
            strict=strict,
            ctx=ctx,
            bundle_affine_bits=bundle_affine_bits,
            inference_mode=inference_mode,
            module_root=self._core,
        )
        self._build_param_map()
        return loaded, skipped

    def _resolve_timestep_value(
        self,
        batch_size: int,
        timestep: Any,
        sigmas: Any | None,
    ) -> Any:
        """Map denoise step index → flow timestep (×1000), matching Flux2 / diffusers."""
        ctx = self.ctx
        if sigmas is not None:
            t_idx: int | None = None
            if isinstance(timestep, int) and not isinstance(timestep, bool):
                t_idx = int(timestep)
            elif isinstance(timestep, mx.array) and timestep.ndim == 0:
                dt = str(timestep.dtype).lower()
                if "int" in dt and "float" not in dt:
                    t_idx = int(timestep.item())
            if t_idx is not None:
                return (sigmas[t_idx] * 1000.0).reshape((1,))

        timestep_val = timestep if hasattr(timestep, "shape") else ctx.array(timestep, dtype=mx.float32())
        if timestep_val.ndim == 0:
            timestep_val = ctx.full((batch_size,), float(timestep_val), dtype=mx.float32())
        timestep_scale = ctx.where(ctx.max(timestep_val) <= 1.0, 1000.0, 1.0)
        return timestep_val * timestep_scale

    def forward(
        self,
        latents: Any,
        timestep: Any,
        txt_embeds: Any = None,
        sigmas: Any = None,
        **cond: Any,
    ) -> Any:
        ctx = self.ctx
        text_lens = cond.get("text_lens")
        if txt_embeds is None:
            raise RuntimeError("ERNIE-Image requires txt_embeds")
        if text_lens is None:
            raise RuntimeError("ERNIE-Image requires text_lens in step kwargs")

        if latents.ndim == 3:
            b, seq, ch = latents.shape
            lh = cond.get("latent_h")
            lw = cond.get("latent_w")
            if lh is None or lw is None:
                side = int(seq**0.5)
                lh = lw = side
            latents = latents.reshape(b, int(lh), int(lw), ch).transpose(0, 3, 1, 2)

        b = int(latents.shape[0])
        t_arr = self._resolve_timestep_value(b, timestep, sigmas)
        if int(t_arr.shape[0]) == 1 and b > 1:
            t_arr = mx.broadcast_to(t_arr, (b,))

        if hasattr(text_lens, "astype"):
            tl = text_lens.astype(mx.int32)
        else:
            tl = mx.array(text_lens, dtype=mx.int32)

        out = self._core(
            latents.astype(mx.bfloat16),
            t_arr,
            txt_embeds.astype(mx.bfloat16),
            tl,
        )
        return out.astype(latents.dtype)
