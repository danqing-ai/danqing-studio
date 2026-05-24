"""Qwen-Image DiT（MLX）— 单文件：结构与 diffusers 对齐，权重经 ``remap_qwen_transformer_weights`` 扁平加载。"""
from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from backend.engine.common.attention import (
    build_padding_attention_bias,
    scaled_dot_product_attention_bhsd_mx,
)
from backend.engine.common._base import TransformerBase
from backend.engine.common.embeddings import apply_complex_rope_bshd, sinusoidal_timestep_proj
from backend.engine.common.norm import (
    apply_ada_layer_norm_continuous,
    apply_scale_shift,
    unpack_modulation_3way,
)
from backend.engine.config.model_configs import QwenImageConfig
from backend.engine.common.text_encoders.qwen3_mlx import MlxRMSNorm, MlxTimestepEmbeddingMLP
from backend.engine.runtime.mlx import MLXContext
from backend.engine.runtime._base import RuntimeContext

_MLX_CTX = MLXContext()


def _scalar_f64(value: Any) -> float:
    try:
        item = getattr(value, "item", None)
        if callable(item):
            return float(item())
    except Exception:
        pass
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


class AdaLayerNormContinuous(nn.Module):
    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=False)
        self.norm = nn.LayerNorm(dims=embedding_dim, eps=1e-6, affine=False)

    def __call__(self, x: mx.array, text_embeddings: mx.array) -> mx.array:
        return apply_ada_layer_norm_continuous(
            x,
            text_embeddings,
            linear=self.linear,
            norm=self.norm,
            embedding_dim=self.embedding_dim,
            silu=nn.silu,
            pre_linear_dtype=mx.bfloat16,
        )


class QwenTimeTextEmbed(nn.Module):
    def __init__(self, timestep_proj_dim: int, inner_dim: int, scale: float = 1000.0):
        super().__init__()
        self.timestep_proj_dim = timestep_proj_dim
        self.scale = scale
        self.timestep_embedder = MlxTimestepEmbeddingMLP(timestep_proj_dim, inner_dim)

    def __call__(self, timestep: mx.array, hidden_states: mx.array) -> mx.array:
        if len(timestep.shape) != 1:
            raise ValueError("timesteps must be 1-d")
        timesteps_proj = sinusoidal_timestep_proj(
            _MLX_CTX,
            timestep,
            self.timestep_proj_dim,
            flip_sin_to_cos=True,
            scale=self.scale,
        )
        return self.timestep_embedder(timesteps_proj.astype(hidden_states.dtype))


class QwenEmbedRopeMLX(nn.Module):
    """3-D RoPE tables for image + text positions (matches bundled checkpoint layout)."""

    def __init__(
        self,
        theta: int,
        axes_dim: list[int],
        scale_rope: bool = False,
        array_fn: Any | None = None,
    ):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.scale_rope = scale_rope
        self.array_fn = array_fn or mx.array

        pos_index = np.arange(4096, dtype=np.int32)
        neg_index = (np.arange(4096, dtype=np.int32)[::-1] * -1) - 1

        self.pos_freqs = np.concatenate(
            [
                self._rope_params(pos_index, self.axes_dim[0], self.theta),
                self._rope_params(pos_index, self.axes_dim[1], self.theta),
                self._rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            axis=1,
        )
        self.neg_freqs = np.concatenate(
            [
                self._rope_params(neg_index, self.axes_dim[0], self.theta),
                self._rope_params(neg_index, self.axes_dim[1], self.theta),
                self._rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            axis=1,
        )

    def _rope_params(self, index: np.ndarray, dim: int, theta: int) -> np.ndarray:
        if dim % 2 != 0:
            raise ValueError("RoPE dim must be even")
        scales = np.arange(0, dim, 2, dtype=np.float32) / dim
        omega = 1.0 / (theta**scales)
        freqs = np.outer(index.astype(np.float32), omega)
        return np.stack([np.cos(freqs), np.sin(freqs)], axis=-1)

    def _compute_video_freqs(
        self, frame: int, height: int, width: int, idx: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        seq_lens = frame * height * width
        axes_splits = [x // 2 for x in self.axes_dim]
        freqs_pos = np.split(self.pos_freqs, np.cumsum(axes_splits)[:-1], axis=1)
        freqs_neg = np.split(self.neg_freqs, np.cumsum(axes_splits)[:-1], axis=1)

        freqs_frame_raw = freqs_pos[0][idx : idx + frame]
        freqs_frame = freqs_frame_raw.reshape(frame, 1, 1, -1, 2)
        freqs_frame = np.broadcast_to(freqs_frame, (frame, height, width, freqs_frame.shape[-2], 2))

        if self.scale_rope:
            freqs_height = np.concatenate(
                [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], axis=0
            )
        else:
            freqs_height = freqs_pos[1][:height]
        freqs_height = freqs_height.reshape(1, height, 1, -1, 2)
        freqs_height = np.broadcast_to(freqs_height, (frame, height, width, freqs_height.shape[-2], 2))

        if self.scale_rope:
            freqs_width = np.concatenate(
                [freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], axis=0
            )
        else:
            freqs_width = freqs_pos[2][:width]
        freqs_width = freqs_width.reshape(1, 1, width, -1, 2)
        freqs_width = np.broadcast_to(freqs_width, (frame, height, width, freqs_width.shape[-2], 2))

        freqs = np.concatenate([freqs_frame, freqs_height, freqs_width], axis=-2)
        freqs = freqs.reshape(seq_lens, -1, 2)
        return freqs[..., 0], freqs[..., 1]

    def __call__(
        self,
        video_fhw: tuple[int, int, int] | list[tuple[int, int, int]],
        txt_seq_lens: list[int],
    ) -> tuple[tuple[mx.array, mx.array], tuple[mx.array, mx.array]]:
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_cos_list: list[np.ndarray] = []
        vid_sin_list: list[np.ndarray] = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            cos_v, sin_v = self._compute_video_freqs(frame, height, width, idx)
            vid_cos_list.append(cos_v)
            vid_sin_list.append(sin_v)
            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        vid_cos = np.concatenate(vid_cos_list, axis=0)
        vid_sin = np.concatenate(vid_sin_list, axis=0)

        max_len = max(txt_seq_lens)
        txt_cos = self.pos_freqs[max_vid_index : max_vid_index + max_len, :, 0]
        txt_sin = self.pos_freqs[max_vid_index : max_vid_index + max_len, :, 1]

        return (
            (self.array_fn(vid_cos.astype(np.float32)), self.array_fn(vid_sin.astype(np.float32))),
            (self.array_fn(txt_cos.astype(np.float32)), self.array_fn(txt_sin.astype(np.float32))),
        )


class QwenFeedForward(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp_in = nn.Linear(dim, 4 * dim, bias=True)
        self.mlp_out = nn.Linear(4 * dim, dim, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.mlp_in(hidden_states)
        hidden_states = nn.gelu_approx(hidden_states)
        return self.mlp_out(hidden_states)


class QwenAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.add_q_proj = nn.Linear(dim, dim)
        self.add_k_proj = nn.Linear(dim, dim)
        self.add_v_proj = nn.Linear(dim, dim)
        self.norm_q = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.norm_added_q = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.norm_added_k = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.attn_to_out = [nn.Linear(dim, dim)]
        self.to_add_out = nn.Linear(dim, dim)

    def __call__(
        self,
        img_modulated: mx.array,
        txt_modulated: mx.array,
        encoder_hidden_states_mask: mx.array | None,
        image_rotary_emb: tuple[mx.array, mx.array],
        block_idx: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        del block_idx
        img_query = self.to_q(img_modulated)
        img_key = self.to_k(img_modulated)
        img_value = self.to_v(img_modulated)

        txt_query = self.add_q_proj(txt_modulated)
        txt_key = self.add_k_proj(txt_modulated)
        txt_value = self.add_v_proj(txt_modulated)

        img_query = mx.reshape(img_query, (*img_query.shape[:2], self.num_heads, self.head_dim))
        img_key = mx.reshape(img_key, (*img_key.shape[:2], self.num_heads, self.head_dim))
        img_value = mx.reshape(img_value, (*img_value.shape[:2], self.num_heads, self.head_dim))

        txt_query = mx.reshape(txt_query, (*txt_query.shape[:2], self.num_heads, self.head_dim))
        txt_key = mx.reshape(txt_key, (*txt_key.shape[:2], self.num_heads, self.head_dim))
        txt_value = mx.reshape(txt_value, (*txt_value.shape[:2], self.num_heads, self.head_dim))

        img_query = self.norm_q(img_query)
        img_key = self.norm_k(img_key)
        txt_query = self.norm_added_q(txt_query)
        txt_key = self.norm_added_k(txt_key)

        (img_cos, img_sin), (txt_cos, txt_sin) = image_rotary_emb
        img_query = apply_complex_rope_bshd(mx, img_query, img_cos, img_sin)
        img_key = apply_complex_rope_bshd(mx, img_key, img_cos, img_sin)
        txt_query = apply_complex_rope_bshd(mx, txt_query, txt_cos, txt_sin)
        txt_key = apply_complex_rope_bshd(mx, txt_key, txt_cos, txt_sin)

        joint_query = mx.concatenate([txt_query, img_query], axis=1)
        joint_key = mx.concatenate([txt_key, img_key], axis=1)
        joint_value = mx.concatenate([txt_value, img_value], axis=1)

        seq_txt = txt_modulated.shape[1]
        mask = QwenAttention._convert_mask_for_qwen(
            mask=encoder_hidden_states_mask,
            joint_seq_len=joint_query.shape[1],
            txt_seq_len=seq_txt,
        )

        hidden_states = self._compute_attention_qwen(
            query=joint_query,
            key=joint_key,
            value=joint_value,
            mask=mask,
        )

        txt_attn_output = hidden_states[:, :seq_txt, :]
        img_attn_output = hidden_states[:, seq_txt:, :]
        img_attn_output = self.attn_to_out[0](img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)
        return img_attn_output, txt_attn_output

    def _compute_attention_qwen(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        query_bhsd = mx.transpose(query, (0, 2, 1, 3))
        key_bhsd = mx.transpose(key, (0, 2, 1, 3))
        value_bhsd = mx.transpose(value, (0, 2, 1, 3))
        head_dim = query.shape[-1]
        scale_value = 1.0 / (head_dim**0.5)
        hidden_states_bhsd = scaled_dot_product_attention_bhsd_mx(
            mx,
            query_bhsd,
            key_bhsd,
            value_bhsd,
            scale=scale_value,
            mask=mask,
        )
        hidden_states = mx.transpose(hidden_states_bhsd, (0, 2, 1, 3))
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        hidden_states = mx.reshape(hidden_states, (batch_size, seq_len, self.num_heads * self.head_dim))
        return hidden_states.astype(query.dtype)

    @staticmethod
    def _convert_mask_for_qwen(
        mask: mx.array | None,
        joint_seq_len: int,
        txt_seq_len: int,
    ) -> mx.array | None:
        if mask is None:
            return None
        bsz = mask.shape[0]
        img_seq_len = joint_seq_len - txt_seq_len
        ones_img = mx.ones((bsz, img_seq_len), dtype=mx.float32)
        joint_mask = mx.concatenate([mask.astype(mx.float32), ones_img], axis=1)
        if mx.all(joint_mask >= 0.999):
            return None
        return build_padding_attention_bias(
            mx,
            joint_mask,
            joint_mask.shape[1],
            mx.float32,
            valid_value=1,
            neg_value=-1e9,
        )

class QwenTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.img_mod_silu = nn.SiLU()
        self.img_mod_linear = nn.Linear(dim, 6 * dim, bias=True)
        self.img_norm1 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)
        self.attn = QwenAttention(dim=dim, num_heads=num_heads, head_dim=head_dim)
        self.img_norm2 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)
        self.img_ff = QwenFeedForward(dim=dim)

        self.txt_mod_silu = nn.SiLU()
        self.txt_mod_linear = nn.Linear(dim, 6 * dim, bias=True)
        self.txt_norm1 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)
        self.txt_norm2 = nn.LayerNorm(dims=dim, eps=1e-6, affine=False)
        self.txt_ff = QwenFeedForward(dim=dim)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array | None,
        text_embeddings: mx.array,
        image_rotary_emb: tuple[mx.array, mx.array],
        block_idx: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        img_mod_params = self.img_mod_linear(self.img_mod_silu(text_embeddings))
        txt_mod_params = self.txt_mod_linear(self.txt_mod_silu(text_embeddings))

        img_mod1, img_mod2 = mx.split(img_mod_params, 2, axis=-1)
        txt_mod1, txt_mod2 = mx.split(txt_mod_params, 2, axis=-1)

        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = QwenTransformerBlock._modulate(img_normed, img_mod1)

        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = QwenTransformerBlock._modulate(txt_normed, txt_mod1)

        img_attn_output, txt_attn_output = self.attn(
            img_modulated=img_modulated,
            txt_modulated=txt_modulated,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            block_idx=block_idx,
        )

        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = QwenTransformerBlock._modulate(img_normed2, img_mod2)
        hidden_states = hidden_states + img_gate2 * self.img_ff(img_modulated2)

        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = QwenTransformerBlock._modulate(txt_normed2, txt_mod2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * self.txt_ff(txt_modulated2)

        return encoder_hidden_states, hidden_states

    @staticmethod
    def _modulate(x: mx.array, mod_params: mx.array) -> tuple[mx.array, mx.array]:
        shift, scale, gate = unpack_modulation_3way(mod_params)
        modulated = apply_scale_shift(x, scale[:, None, :], shift[:, None, :], add_one=True)
        return modulated, gate[:, None, :]


class QwenTransformer(nn.Module):
    """Inner DiT — hyper-parameters from ``QwenImageConfig``."""

    def __init__(
        self, config: QwenImageConfig, *, patch_size: int = 2, array_fn: Any | None = None
    ) -> None:
        super().__init__()
        self.config = config
        head_dim = config.hidden_dim // config.num_heads
        self.inner_dim = config.num_heads * head_dim

        self.img_in = nn.Linear(config.in_channels, self.inner_dim)
        self.txt_norm = MlxRMSNorm(config.text_dim, eps=1e-6)
        self.txt_in = nn.Linear(config.text_dim, self.inner_dim)
        self.time_text_embed = QwenTimeTextEmbed(timestep_proj_dim=256, inner_dim=self.inner_dim)
        self.array_fn = array_fn or mx.array
        self.pos_embed = QwenEmbedRopeMLX(
            theta=10000, axes_dim=[16, 56, 56], scale_rope=True, array_fn=self.array_fn
        )
        self.transformer_blocks = [
            QwenTransformerBlock(dim=self.inner_dim, num_heads=config.num_heads, head_dim=head_dim)
            for _ in range(config.num_layers)
        ]
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim)
        self.proj_out = nn.Linear(
            self.inner_dim, patch_size * patch_size * config.out_channels
        )

    def __call__(
        self,
        t: int,
        config: Any,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        encoder_hidden_states_mask: mx.array,
        qwen_image_ids: mx.array | None = None,
        cond_image_grid: tuple[int, int, int] | None = None,
    ) -> mx.array:
        del qwen_image_ids
        hidden_states = self.img_in(hidden_states)
        batch_size = hidden_states.shape[0]
        timestep = QwenTransformer._compute_timestep(t, config, array_fn=self.array_fn)
        timestep = mx.broadcast_to(timestep, (batch_size,)).astype(hidden_states.dtype)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)
        text_embeddings = self.time_text_embed(timestep, hidden_states)
        image_rotary_embeddings = QwenTransformer._compute_rotary_embeddings(
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            pos_embed=self.pos_embed,
            config=config,
            cond_image_grid=cond_image_grid,
        )
        for idx, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                text_embeddings=text_embeddings,
                image_rotary_emb=image_rotary_embeddings,
                block_idx=idx,
            )
        hidden_states = self.norm_out(hidden_states, text_embeddings)
        return self.proj_out(hidden_states)

    @staticmethod
    def _compute_timestep(
        t: int | float, config: Any, *, array_fn: Any | None = None
    ) -> mx.array:
        if array_fn is None:
            array_fn = mx.array
        if isinstance(t, int):
            sched = config.scheduler
            sigmas = sched.sigmas
            timesteps_list = getattr(sched, "timesteps", None)
            if timesteps_list is not None and hasattr(timesteps_list, "__len__"):
                try:
                    if t < len(sigmas):
                        time_step = sigmas[t]
                        return array_fn(
                            np.full((1,), _scalar_f64(time_step)), dtype=mx.float32
                        )
                except Exception:
                    pass
                for idx, ts in enumerate(timesteps_list):
                    try:
                        ts_i = int(_scalar_f64(ts))
                    except Exception:
                        continue
                    if abs(ts_i - t) < 1:
                        time_step = sigmas[idx]
                        return array_fn(
                            np.full((1,), _scalar_f64(time_step)), dtype=mx.float32
                        )
            time_step = float(t) / 1000.0
            return array_fn(np.full((1,), time_step), dtype=mx.float32)
        time_step = float(t)
        return array_fn(np.full((1,), time_step), dtype=mx.float32)

    @staticmethod
    def _compute_rotary_embeddings(
        encoder_hidden_states_mask: mx.array,
        pos_embed: QwenEmbedRopeMLX,
        config: Any,
        cond_image_grid: tuple[int, int, int] | list[tuple[int, int, int]] | None = None,
    ) -> tuple[mx.array, mx.array]:
        latent_height = config.height // 16
        latent_width = config.width // 16

        if cond_image_grid is None:
            img_shapes = [(1, latent_height, latent_width)]
        elif isinstance(cond_image_grid, list):
            img_shapes = [(1, latent_height, latent_width)] + cond_image_grid
        else:
            img_shapes = [(1, latent_height, latent_width), cond_image_grid]

        txt_seq_lens = []
        for i in range(encoder_hidden_states_mask.shape[0]):
            txt_seq_lens.append(int(_scalar_f64(mx.sum(encoder_hidden_states_mask[i]))))
        img_rotary_emb, txt_rotary_emb = pos_embed(video_fhw=img_shapes, txt_seq_lens=txt_seq_lens)
        return img_rotary_emb, txt_rotary_emb


class _SchedulerShim:
    def __init__(self, sigmas: Any, timesteps: Any):
        self._sigmas = sigmas
        self._timesteps = timesteps

    @property
    def sigmas(self) -> Any:
        return self._sigmas

    @property
    def timesteps(self) -> Any:
        return self._timesteps


class _ConfigShim:
    def __init__(self, *, height: int, width: int, sigmas: Any, timesteps: Any | None, model_config: Any):
        self._height = int(height)
        self._width = int(width)
        self.model_config = model_config
        ts = timesteps if timesteps is not None else sigmas
        self._scheduler = _SchedulerShim(sigmas, ts)

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    @property
    def scheduler(self) -> _SchedulerShim:
        return self._scheduler


class QwenImageTransformer(TransformerBase):
    """Pipeline 入口：NCHW latent ↔ DiT 序列；权重由 Pipeline ``remap_fn`` 扁平载入 ``dit``。"""

    def __init__(self, config: QwenImageConfig, ctx: RuntimeContext):
        self.ctx = ctx
        self.config = config
        self.dit = QwenTransformer(config, array_fn=ctx.array)
        self._param_map: dict[str, Any] = {}
        self._build_param_map()

    def _build_param_map(self):
        """MLX ``Module.parameters()`` 返回嵌套 dict/list；展开为与 ``remap_qwen_transformer_weights`` 一致的扁平键。"""
        self._param_map.clear()

        def flatten(obj: Any, prefix: str) -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    p = f"{prefix}.{k}" if prefix else str(k)
                    flatten(v, p)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    p = f"{prefix}.{i}"
                    flatten(item, p)
            else:
                self._param_map[prefix] = obj

        flatten(self.dit.parameters(), "dit")

    def parameters(self):
        return list(self._param_map.items())

    def load_weights(
        self,
        weights: list[tuple[str, Any]],
        strict: bool = False,
        ctx: Any = None,
        *,
        bundle_affine_bits: int | None = None,
    ):
        load_ctx = ctx if ctx is not None else self.ctx
        loaded, skipped = super().load_weights(
            weights,
            strict=strict,
            ctx=load_ctx,
            bundle_affine_bits=bundle_affine_bits,
        )
        self._cast_param_map_dtype(mx.bfloat16)
        return loaded, skipped

    def after_load_weights(self, bundle_root: str | None = None):
        del bundle_root

    def forward(
        self,
        latents,
        timestep,
        txt_embeds=None,
        sigmas=None,
        timestep_embed_value=None,
        encoder_hidden_states_mask=None,
        scheduler_timesteps=None,
        image_height: int | None = None,
        image_width: int | None = None,
        **conditioning,
    ):
        del timestep_embed_value, conditioning
        if txt_embeds is None:
            raise RuntimeError("Qwen Image requires txt_embeds.")
        if encoder_hidden_states_mask is None:
            raise RuntimeError("Qwen Image requires encoder_hidden_states_mask.")
        if sigmas is None:
            raise RuntimeError("Qwen Image requires sigmas.")
        if image_height is None or image_width is None:
            raise RuntimeError("Qwen Image requires image_height / image_width.")

        ctx = self.ctx
        B, _C, H_lat, W_lat = latents.shape
        x = ctx.permute(latents, (0, 2, 3, 1))
        seq = ctx.reshape(x, (B, H_lat * W_lat, latents.shape[1]))
        seq_mx = seq.astype(mx.bfloat16)
        enc_mx = txt_embeds.astype(mx.bfloat16)
        mask_mx = encoder_hidden_states_mask.astype(mx.float32)

        class _MC:
            requires_sigma_shift = True

        cfg = _ConfigShim(
            height=int(image_height),
            width=int(image_width),
            sigmas=sigmas,
            timesteps=scheduler_timesteps,
            model_config=_MC(),
        )

        out_mx = self.dit(
            t=int(timestep),
            config=cfg,
            hidden_states=seq_mx,
            encoder_hidden_states=enc_mx,
            encoder_hidden_states_mask=mask_mx,
        )
        ctx.eval(out_mx)
        out_f = out_mx.astype(mx.float32)
        _, seq_len, c_out = out_f.shape
        side = int(seq_len**0.5)
        if side * side != seq_len:
            raise RuntimeError(f"Qwen DiT output seq_len={seq_len} is not square.")
        out = mx.reshape(out_f, (B, side, side, c_out))
        return mx.transpose(out, (0, 3, 1, 2))
