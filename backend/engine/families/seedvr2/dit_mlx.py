from __future__ import annotations

"""SeedVR2 MM-DiT（单文件）。``SeedVR2DiT`` 继承 ``mlx.nn.Module`` 与 ``TransformerBase``：

- 权重仍由 ``WeightApplier`` + ``WeightLoader`` 注入；``after_load_weights`` 供 LoRA 等扩展。
- 推理入口为 ``__call__(txt=, vid=, timestep=)``；``forward(latents, …)`` 未实现并显式报错。
"""

import math
from typing import Callable

import mlx.core as mx
from mlx import nn

from backend.engine.common._base import TransformerBase


# ----- rms_norm.py -----


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


# ----- swiglu_mlp.py -----

class SwiGLUMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_ratio: int = 4,
        multiple_of: int = 256,
        bias: bool = False,
    ):
        super().__init__()
        hidden_dim = int(2 * dim * expand_ratio / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.proj_in = nn.Linear(dim, hidden_dim, bias=bias)
        self.proj_in_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.proj_out = nn.Linear(hidden_dim, dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        gate = nn.silu(self.proj_in_gate(x))
        x = self.proj_in(x)
        x = gate * x
        x = self.proj_out(x)
        return x


class GELUMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_ratio: int = 4,
        bias: bool = True,
    ):
        super().__init__()
        hidden_dim = dim * expand_ratio
        self.proj_in = nn.Linear(dim, hidden_dim, bias=bias)
        self.proj_out = nn.Linear(hidden_dim, dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj_in(x)
        x = nn.gelu_approx(x)
        x = self.proj_out(x)
        return x


# ----- time_embedding.py -----

class TimeEmbedding(nn.Module):
    def __init__(
        self,
        sinusoidal_dim: int = 256,
        hidden_dim: int = 2560,
        output_dim: int = 15360,
    ):
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.proj_in = nn.Linear(sinusoidal_dim, hidden_dim)
        self.proj_hid = nn.Linear(hidden_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)

    def __call__(self, timestep: mx.array) -> mx.array:
        timestep = timestep[None] if timestep.ndim == 0 else timestep
        emb = TimeEmbedding._get_timestep_embedding(
            timesteps=timestep,
            embedding_dim=self.sinusoidal_dim,
        )
        emb = self.proj_in(emb)
        emb = nn.silu(emb)
        emb = self.proj_hid(emb)
        emb = nn.silu(emb)
        emb = self.proj_out(emb)
        return emb

    @staticmethod
    def _get_timestep_embedding(
        timesteps: mx.array,
        embedding_dim: int,
    ) -> mx.array:
        half_dim = embedding_dim // 2
        freqs = mx.exp(mx.arange(half_dim, dtype=mx.float32) * (-math.log(10000) / half_dim))
        args = timesteps[:, None].astype(mx.float32) * freqs
        return mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)


# ----- rope.py -----

class RoPEModule(nn.Module):
    def __init__(self, dim: int = 64, freqs_for: str = "lang", theta: float = 10000.0, max_freq: float = 256.0):
        super().__init__()
        self.dim = dim
        self.freqs_for = freqs_for
        self.rope_dim = 3
        self.freq_dim = dim // self.rope_dim
        if freqs_for == "pixel":
            self.freqs = mx.linspace(1.0, max_freq / 2.0, self.freq_dim // 2, dtype=mx.float32) * mx.pi
        else:
            self.freqs = 1.0 / (
                theta ** (mx.arange(0, self.freq_dim, 2, dtype=mx.float32)[: (self.freq_dim // 2)] / self.freq_dim)
            )

    def __call__(
        self,
        vid_q: mx.array,
        vid_k: mx.array,
        vid_shape: mx.array,
        txt_q: mx.array | None = None,
        txt_k: mx.array | None = None,
        txt_shape: mx.array | None = None,
    ) -> tuple[mx.array, mx.array] | tuple[mx.array, mx.array, mx.array, mx.array]:
        if txt_q is None or txt_k is None or txt_shape is None:
            return RoPEModule._apply_rope_3d(
                vid_q,
                vid_k,
                vid_shape,
                self.freqs,
                self.rope_dim,
                self.freqs_for,
            )
        return RoPEModule._apply_mm_rope_3d(
            vid_q,
            vid_k,
            vid_shape,
            txt_q,
            txt_k,
            txt_shape,
            self.freqs,
            self.rope_dim,
            self.freqs_for,
        )

    @classmethod
    def _apply_rope_3d(
        cls,
        vid_q: mx.array,
        vid_k: mx.array,
        vid_shape: mx.array,
        freqs: mx.array,
        rope_dim: int = 3,
        freqs_for: str = "lang",
    ) -> tuple[mx.array, mx.array]:
        vid_freq_list = []
        if freqs_for == "pixel":
            # For pixel-space RoPE, the position grid is normalized with linspace(-1, 1).
            # We must use exact dimensions per sample/window; slicing a larger precomputed grid
            # changes spacing and introduces spatial bias.
            for b in range(vid_shape.shape[0]):
                f, h, w = int(vid_shape[b, 0]), int(vid_shape[b, 1]), int(vid_shape[b, 2])
                vid_freq = RoPEModule._get_axial_freqs(freqs, f, h, w, freqs_for=freqs_for).reshape(-1, freqs.size * 2 * rope_dim)
                vid_freq_list.append(vid_freq)
        else:
            max_temporal = int(mx.max(vid_shape[:, 0]))
            max_height = int(mx.max(vid_shape[:, 1]))
            max_width = int(mx.max(vid_shape[:, 2]))

            clamp_temporal = min(max_temporal + 16, 1024)
            clamp_height = min(max_height + 4, 128)
            clamp_width = min(max_width + 4, 128)

            vid_freqs_full = RoPEModule._get_axial_freqs(
                freqs, clamp_temporal, clamp_height, clamp_width, freqs_for=freqs_for
            )

            for b in range(vid_shape.shape[0]):
                f, h, w = int(vid_shape[b, 0]), int(vid_shape[b, 1]), int(vid_shape[b, 2])
                vid_freq = vid_freqs_full[:f, :h, :w].reshape(-1, vid_freqs_full.shape[-1])
                vid_freq_list.append(vid_freq)

        vid_freqs = mx.concatenate(vid_freq_list, axis=0)

        vid_q = RoPEModule._apply_rotary_emb(vid_freqs[:, None, :], vid_q)
        vid_k = RoPEModule._apply_rotary_emb(vid_freqs[:, None, :], vid_k)

        return vid_q, vid_k

    @staticmethod
    def _rotate_half(x: mx.array) -> mx.array:
        x = x.reshape(*x.shape[:-1], -1, 2)
        x1, x2 = x[..., 0], x[..., 1]
        x = mx.stack([-x2, x1], axis=-1)
        return x.reshape(*x.shape[:-2], -1)

    @classmethod
    def _get_axial_freqs(cls, freqs: mx.array, *dims: int, freqs_for: str = "lang") -> mx.array:
        freq_dim_per_axis = len(freqs) * 2
        target_shape = list(dims) + [freq_dim_per_axis]
        all_freqs = []

        for ind, dim_size in enumerate(dims):
            if freqs_for == "pixel":
                pos = mx.linspace(-1.0, 1.0, dim_size, dtype=mx.float32)
            else:
                pos = mx.arange(dim_size, dtype=mx.float32)
            axis_freqs = mx.outer(pos, freqs.astype(mx.float32))
            axis_freqs = mx.repeat(axis_freqs, 2, axis=-1)

            shape = [1] * len(dims) + [freq_dim_per_axis]
            shape[ind] = dim_size
            axis_freqs = axis_freqs.reshape(*shape)
            all_freqs.append(axis_freqs)

        broadcasted = [mx.broadcast_to(f, target_shape) for f in all_freqs]
        output = mx.concatenate(broadcasted, axis=-1)
        return output

    @staticmethod
    def _apply_rotary_emb(freqs: mx.array, t: mx.array) -> mx.array:
        rot_dim = freqs.shape[-1]
        t_middle = t[..., :rot_dim]
        t_right = t[..., rot_dim:]

        t_dtype = t_middle.dtype
        freqs_f = freqs.astype(mx.float32)
        t_middle_f = t_middle.astype(mx.float32)
        cos_freqs = mx.cos(freqs_f)
        sin_freqs = mx.sin(freqs_f)
        t_transformed = (t_middle_f * cos_freqs) + (RoPEModule._rotate_half(t_middle_f) * sin_freqs)
        t_transformed = t_transformed.astype(t_dtype)

        if t_right.shape[-1] > 0:
            return mx.concatenate([t_transformed, t_right], axis=-1)
        return t_transformed

    @classmethod
    def _apply_mm_rope_3d(
        cls,
        vid_q: mx.array,
        vid_k: mx.array,
        vid_shape: mx.array,
        txt_q: mx.array,
        txt_k: mx.array,
        txt_shape: mx.array,
        freqs: mx.array,
        rope_dim: int = 3,
        freqs_for: str = "lang",
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        vid_freq_list = []
        txt_freq_list = []
        if freqs_for == "pixel":
            # Same rationale as _apply_rope_3d: compute exact grids in pixel mode.
            for b in range(vid_shape.shape[0]):
                f, h, w = int(vid_shape[b, 0]), int(vid_shape[b, 1]), int(vid_shape[b, 2])
                txt_len = int(txt_shape[b, 0])

                full = RoPEModule._get_axial_freqs(freqs, txt_len + f, h, w, freqs_for=freqs_for)
                vid_freq = full[txt_len : txt_len + f, :h, :w].reshape(-1, full.shape[-1])
                txt_freq = RoPEModule._get_axial_freqs(freqs, txt_len, freqs_for=freqs_for)
                txt_freq = mx.tile(txt_freq, (1, rope_dim))

                vid_freq_list.append(vid_freq)
                txt_freq_list.append(txt_freq)
        else:
            max_temporal = int(mx.max(vid_shape[:, 0] + txt_shape[:, 0]))
            max_height = int(mx.max(vid_shape[:, 1]))
            max_width = int(mx.max(vid_shape[:, 2]))
            max_txt_len = int(mx.max(txt_shape[:, 0]))

            clamp_temporal = min(max_temporal + 16, 1024)
            clamp_height = min(max_height + 4, 128)
            clamp_width = min(max_width + 4, 128)

            vid_freqs_full = RoPEModule._get_axial_freqs(
                freqs, clamp_temporal, clamp_height, clamp_width, freqs_for=freqs_for
            )
            txt_freqs_1d = RoPEModule._get_axial_freqs(freqs, min(max_txt_len + 16, 1024), freqs_for=freqs_for)

            for b in range(vid_shape.shape[0]):
                f, h, w = int(vid_shape[b, 0]), int(vid_shape[b, 1]), int(vid_shape[b, 2])
                txt_len = int(txt_shape[b, 0])

                vid_freq = vid_freqs_full[txt_len : txt_len + f, :h, :w].reshape(-1, vid_freqs_full.shape[-1])
                txt_freq = mx.tile(txt_freqs_1d[:txt_len], (1, rope_dim))

                vid_freq_list.append(vid_freq)
                txt_freq_list.append(txt_freq)

        vid_freqs = mx.concatenate(vid_freq_list, axis=0)
        txt_freqs = mx.concatenate(txt_freq_list, axis=0)

        vid_q = RoPEModule._apply_rotary_emb(vid_freqs[:, None, :], vid_q)
        vid_k = RoPEModule._apply_rotary_emb(vid_freqs[:, None, :], vid_k)

        txt_q = RoPEModule._apply_rotary_emb(txt_freqs[:, None, :], txt_q)
        txt_k = RoPEModule._apply_rotary_emb(txt_freqs[:, None, :], txt_k)

        return vid_q, vid_k, txt_q, txt_k


# ----- window.py -----

class WindowPartitioner:
    def __init__(
        self,
        shape: mx.array,
        window_size: tuple[int, int, int],
        shift: bool = False,
    ):
        self.forward_idx, self.reverse_idx, self.window_shapes, self.window_counts = (
            WindowPartitioner._create_window_indices(
                shape, lambda size: WindowPartitioner._make_windows(size, window_size, shift)
            )
        )

    def partition(self, tensor: mx.array) -> mx.array:
        return tensor[self.forward_idx]

    def reverse(self, tensor: mx.array) -> mx.array:
        return tensor[self.reverse_idx]

    @staticmethod
    def _make_windows(
        size: tuple[int, int, int],
        num_windows: tuple[int, int, int],
        shift: bool = False,
    ) -> list[tuple[slice, slice, slice]]:
        t, h, w = size
        resized_nt, resized_nh, resized_nw = num_windows

        scale = math.sqrt((45 * 80) / (h * w))
        resized_h, resized_w = round(h * scale), round(w * scale)

        wh = math.ceil(resized_h / resized_nh)
        ww = math.ceil(resized_w / resized_nw)
        wt = math.ceil(min(t, 30) / resized_nt)

        if shift:
            st = 0.5 if wt < t else 0
            sh = 0.5 if wh < h else 0
            sw = 0.5 if ww < w else 0
            nt = math.ceil((t - st) / wt) + 1 if st > 0 else 1
            nh = math.ceil((h - sh) / wh) + 1 if sh > 0 else 1
            nw = math.ceil((w - sw) / ww) + 1 if sw > 0 else 1
        else:
            st = sh = sw = 0
            nt = math.ceil(t / wt)
            nh = math.ceil(h / wh)
            nw = math.ceil(w / ww)

        windows = []
        for iw in range(nw):
            w_start = max(int((iw - sw) * ww), 0)
            w_end = min(int((iw - sw + 1) * ww), w)
            if w_end <= w_start:
                continue
            for ih in range(nh):
                h_start = max(int((ih - sh) * wh), 0)
                h_end = min(int((ih - sh + 1) * wh), h)
                if h_end <= h_start:
                    continue
                for it in range(nt):
                    t_start = max(int((it - st) * wt), 0)
                    t_end = min(int((it - st + 1) * wt), t)
                    if t_end <= t_start:
                        continue
                    windows.append((slice(t_start, t_end), slice(h_start, h_end), slice(w_start, w_end)))

        return windows

    @staticmethod
    def _flatten_list(tensors: list[mx.array]) -> tuple[mx.array, mx.array]:
        assert len(tensors) > 0
        shapes = mx.array([x.shape[:-1] for x in tensors], dtype=mx.int32)
        result = mx.concatenate([x.reshape(-1, x.shape[-1]) for x in tensors], axis=0)
        return result, shapes

    @staticmethod
    def _unflatten_list(
        tensor: mx.array,
        shapes: mx.array,
    ) -> list[mx.array]:
        lengths = mx.prod(shapes, axis=1).tolist()
        indices = mx.cumsum(mx.array(lengths[:-1])).tolist()
        pieces = mx.split(tensor, indices)
        return [p.reshape(*s.tolist(), -1) for p, s in zip(pieces, shapes)]

    @classmethod
    def _window_partition(
        cls,
        tensor: mx.array,
        shape: mx.array,
        window_fn: Callable,
    ) -> tuple[mx.array, mx.array, list[int]]:
        unflattened = WindowPartitioner._unflatten_list(tensor, shape)

        windowed = []
        window_counts = []
        for x in unflattened:
            t, h, w = x.shape[:-1]
            slices = window_fn((t, h, w))
            window_counts.append(len(slices))
            for st, sh, sw in slices:
                window = x[st, sh, sw]
                windowed.append(window)

        result, result_shape = WindowPartitioner._flatten_list(windowed)
        return result, result_shape, window_counts

    @classmethod
    def _create_window_indices(
        cls,
        shape: mx.array,
        window_fn: Callable,
    ) -> tuple[mx.array, mx.array, mx.array, list[int]]:
        total_len = int(mx.sum(mx.prod(shape, axis=1)))
        idx = mx.arange(total_len).reshape(-1, 1)
        windowed_idx, window_shapes, window_counts = WindowPartitioner._window_partition(idx, shape, window_fn)
        target_idx = windowed_idx.reshape(-1)
        reverse_idx = mx.argsort(target_idx)
        return target_idx, reverse_idx, window_shapes, window_counts


# ----- ada_modulation.py -----

class AdaModulation(nn.Module):
    def __init__(self, dim: int, shared_weights: bool = False, is_last_layer: bool = False):
        super().__init__()
        self.shared_weights = shared_weights
        self.is_last_layer = is_last_layer

        if shared_weights:
            self.params_all = AdaModulation._init_params(dim)
        else:
            self.params_vid = AdaModulation._init_params(dim)
            if not is_last_layer:
                self.params_txt = AdaModulation._init_params(dim)

    @staticmethod
    def _init_params(dim: int) -> dict[str, mx.array]:
        return {
            "attn_shift": mx.zeros((dim,)),
            "attn_scale": mx.ones((dim,)),
            "attn_gate": mx.zeros((dim,)),
            "mlp_shift": mx.zeros((dim,)),
            "mlp_scale": mx.ones((dim,)),
            "mlp_gate": mx.zeros((dim,)),
        }

    def modulate_vid(
        self,
        hidden: mx.array,
        emb: mx.array,
        layer: str,
        mode: str,
    ) -> mx.array:
        params = self.params_all if self.shared_weights else self.params_vid
        return AdaModulation._apply_modulation(hidden, emb, params, layer, mode)

    def modulate_txt(
        self,
        hidden: mx.array,
        emb: mx.array,
        layer: str,
        mode: str,
    ) -> mx.array:
        if self.is_last_layer:
            return hidden
        params = self.params_all if self.shared_weights else self.params_txt
        return AdaModulation._apply_modulation(hidden, emb, params, layer, mode)

    @staticmethod
    def _apply_modulation(
        hidden: mx.array,
        emb: mx.array,
        params: dict[str, mx.array],
        layer: str,
        mode: str,
    ) -> mx.array:
        layer_idx = 0 if layer == "attn" else 1
        mod = emb[:, :, layer_idx, :]

        if mode == "in":
            shift = mod[..., 0][:, None] + params[f"{layer}_shift"]
            scale = mod[..., 1][:, None] + params[f"{layer}_scale"]
            return hidden * scale + shift
        else:
            gate = mod[..., 2][:, None] + params[f"{layer}_gate"]
            return hidden * gate


# ----- patch_in.py -----

class PatchIn(nn.Module):
    def __init__(
        self,
        in_channels: int = 33,
        patch_size: tuple = (1, 2, 2),
        dim: int = 2560,
    ):
        super().__init__()
        self.patch_size = patch_size
        t, h, w = patch_size
        self.proj = nn.Linear(in_channels * t * h * w, dim)

    def __call__(self, vid: mx.array) -> tuple[mx.array, mx.array]:
        t, h, w = self.patch_size
        B, C, T, H, W = vid.shape

        T_patches = T // t
        H_patches = H // h
        W_patches = W // w

        vid = vid.reshape(B, C, T_patches, t, H_patches, h, W_patches, w)
        vid = vid.transpose(0, 2, 4, 6, 3, 5, 7, 1)
        vid = vid.reshape(B, T_patches, H_patches, W_patches, t * h * w * C)

        vid = self.proj(vid)
        vid = vid.reshape(B, -1, vid.shape[-1])
        vid_shape = mx.broadcast_to(mx.array([T_patches, H_patches, W_patches], dtype=mx.int32), (B, 3))

        return vid, vid_shape


# ----- patch_out.py -----

class PatchOut(nn.Module):
    def __init__(
        self,
        out_channels: int = 16,
        patch_size: tuple = (1, 2, 2),
        dim: int = 2560,
    ):
        super().__init__()
        self.patch_size = patch_size
        t, h, w = patch_size
        self.proj = nn.Linear(dim, out_channels * t * h * w)

    def __call__(self, vid: mx.array, vid_shape: mx.array) -> tuple[mx.array, mx.array]:
        t, h, w = self.patch_size
        vid = self.proj(vid)

        B = vid.shape[0]
        T_patches = int(vid_shape[0, 0])
        H_patches = int(vid_shape[0, 1])
        W_patches = int(vid_shape[0, 2])
        C = vid.shape[-1] // (t * h * w)

        vid = vid.reshape(B, T_patches, H_patches, W_patches, t, h, w, C)
        vid = vid.transpose(0, 7, 1, 4, 2, 5, 3, 6)
        vid = vid.reshape(B, C, T_patches * t, H_patches * h, W_patches * w)

        return vid, vid_shape


# ----- attention.py -----

class MMAttention(nn.Module):
    def __init__(
        self,
        vid_dim: int,
        txt_dim: int,
        heads: int = 20,
        head_dim: int = 128,
        qk_bias: bool = False,
        qk_norm_eps: float = 1e-5,
        rope_dim: int = 128,
        rope_freqs_for: str = "lang",
        rope_on_text: bool = True,
        shared_weights: bool = False,
        window: tuple[int, int, int] = (4, 3, 3),
        shift: bool = False,
    ):
        super().__init__()
        self.shared_weights = shared_weights
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.window = window
        self.shift = shift
        self.rope_on_text = rope_on_text

        inner_dim = heads * head_dim

        self.proj_qkv_vid = nn.Linear(vid_dim, 3 * inner_dim, bias=qk_bias)
        self.proj_out_vid = nn.Linear(inner_dim, vid_dim, bias=True)
        self.norm_q_vid = RMSNorm(head_dim, eps=qk_norm_eps)
        self.norm_k_vid = RMSNorm(head_dim, eps=qk_norm_eps)

        if shared_weights:
            self.proj_qkv_txt = self.proj_qkv_vid
            self.proj_out_txt = self.proj_out_vid
            self.norm_q_txt = self.norm_q_vid
            self.norm_k_txt = self.norm_k_vid
        else:
            self.proj_qkv_txt = nn.Linear(txt_dim, 3 * inner_dim, bias=qk_bias)
            self.proj_out_txt = nn.Linear(inner_dim, txt_dim, bias=True)
            self.norm_q_txt = RMSNorm(head_dim, eps=qk_norm_eps)
            self.norm_k_txt = RMSNorm(head_dim, eps=qk_norm_eps)

        self.rope = RoPEModule(dim=rope_dim, freqs_for=rope_freqs_for)

    def __call__(self, vid, txt, vid_shape, txt_shape):
        B, L, Bt, Lt = vid.shape[0], vid.shape[1], txt.shape[0], txt.shape[1]

        # 1. Project to QKV and Partition
        qkv_vid = self.proj_qkv_vid(vid.reshape(-1, vid.shape[-1])).reshape(-1, 3, self.heads, self.head_dim)
        qkv_txt = self.proj_qkv_txt(txt.reshape(-1, txt.shape[-1])).reshape(-1, 3, self.heads, self.head_dim)

        partitioner = WindowPartitioner(vid_shape, self.window, self.shift)
        qkv_vid = partitioner.partition(qkv_vid)

        # 2. Normalize and repeat text
        q_vid, k_vid, v_vid = self.norm_q_vid(qkv_vid[:, 0]), self.norm_k_vid(qkv_vid[:, 1]), qkv_vid[:, 2]
        q_txt, k_txt, v_txt = self.norm_q_txt(qkv_txt[:, 0]), self.norm_k_txt(qkv_txt[:, 1]), qkv_txt[:, 2]

        counts, txt_len = partitioner.window_counts, txt_shape[:, 0]
        qkv_t_rep = self._repeat_text_for_windows(mx.stack([q_txt, k_txt, v_txt], axis=1), txt_len, counts)
        q_txt_rep, k_txt_rep, v_txt_rep = qkv_t_rep[:, 0], qkv_t_rep[:, 1], qkv_t_rep[:, 2]

        # 3. Apply RoPE
        if self.rope_on_text:
            q_vid, k_vid, q_txt_rep, k_txt_rep = self.rope(
                vid_q=q_vid,
                vid_k=k_vid,
                vid_shape=partitioner.window_shapes,
                txt_q=q_txt_rep,
                txt_k=k_txt_rep,
                txt_shape=mx.repeat(txt_shape, mx.array(counts), axis=0),
            )
        else:
            q_vid, k_vid = self.rope(
                vid_q=q_vid,
                vid_k=k_vid,
                vid_shape=partitioner.window_shapes,
            )

        # 4. Attention
        vid_lens = mx.prod(partitioner.window_shapes, axis=1)
        qkv = self._concat_with_text(
            mx.stack([q_vid, k_vid, v_vid], axis=1),
            mx.stack([q_txt_rep, k_txt_rep, v_txt_rep], axis=1),
            vid_lens,
            txt_len,
            counts,
        )

        win_lens = vid_lens + txt_len[mx.repeat(mx.arange(len(counts)), mx.array(counts))]
        windows = mx.split(qkv, mx.cumsum(win_lens[:-1]).tolist())

        out = []
        for w in windows:
            q, k, v = [x[None].transpose(0, 2, 1, 3) for x in [w[:, 0], w[:, 1], w[:, 2]]]
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
            out.append(o.transpose(0, 2, 1, 3).squeeze(0))

        # 5. Coalesce and Project Out
        out = mx.concatenate(out, axis=0).reshape(-1, self.heads * self.head_dim)
        vid_out, txt_out = self._unconcat_and_coalesce(out, vid_lens, txt_len, counts)

        return (
            self.proj_out_vid(partitioner.reverse(vid_out)).reshape(B, L, -1),
            self.proj_out_txt(txt_out).reshape(Bt, Lt, -1),
        )

    @staticmethod
    def _repeat_text_for_windows(txt, txt_len, counts):
        B, L = len(counts), int(txt_len[0])
        txt = txt.reshape(B, L, *txt.shape[1:])
        return mx.repeat(txt, mx.array(counts), axis=0).reshape(-1, *txt.shape[2:])

    @staticmethod
    def _concat_with_text(vid, txt, vid_lens, txt_len, counts):
        v_parts = mx.split(vid, mx.cumsum(vid_lens[:-1]).tolist())
        t_parts = mx.split(txt, mx.arange(int(txt_len[0]), txt.shape[0], int(txt_len[0])).tolist())
        parts = [p for pair in zip(v_parts, t_parts) for p in pair]
        return mx.concatenate(parts, axis=0)

    @staticmethod
    def _unconcat_and_coalesce(combined, vid_lens, txt_len, counts):
        win_to_batch = mx.repeat(mx.arange(len(txt_len)), mx.array(counts))
        lens = mx.stack([vid_lens, txt_len[win_to_batch]], axis=1).reshape(-1)
        parts = mx.split(combined, mx.cumsum(lens[:-1]).tolist())

        vid_out = mx.concatenate(parts[0::2], axis=0)
        t_parts = parts[1::2]

        final_txt, offset = [], 0
        for count in counts:
            final_txt.append(mx.stack(t_parts[offset : offset + count]).mean(axis=0))
            offset += count
        return vid_out, mx.concatenate(final_txt, axis=0)


# ----- mm_swiglu.py -----

class MMSwiGLU(nn.Module):
    def __init__(
        self,
        vid_dim: int,
        txt_dim: int,
        expand_ratio: int = 4,
        shared_weights: bool = False,
        is_last_layer: bool = False,
        mlp_type: str = "swiglu",
    ):
        super().__init__()
        self.shared_weights = shared_weights
        self.is_last_layer = is_last_layer
        self.mlp_type = mlp_type

        mlp_cls = SwiGLUMLP
        mlp_kwargs = {"expand_ratio": expand_ratio}
        if mlp_type == "normal":
            mlp_cls = GELUMLP
            mlp_kwargs["bias"] = True

        if shared_weights:
            self.all = mlp_cls(dim=vid_dim, **mlp_kwargs)
        else:
            self.vid = mlp_cls(dim=vid_dim, **mlp_kwargs)
            if not is_last_layer:
                self.txt = mlp_cls(dim=txt_dim, **mlp_kwargs)

    def __call__(self, vid: mx.array, txt: mx.array) -> tuple[mx.array, mx.array]:
        if self.shared_weights:
            vid_out = self.all(vid)
            txt_out = self.all(txt)
        else:
            vid_out = self.vid(vid)
            txt_out = self.txt(txt) if not self.is_last_layer else txt

        return vid_out, txt_out


# ----- transformer_block.py -----

class TransformerBlock(nn.Module):
    def __init__(
        self,
        vid_dim: int = 2560,
        txt_dim: int = 2560,
        heads: int = 20,
        head_dim: int = 128,
        expand_ratio: int = 4,
        mlp_type: str = "swiglu",
        norm_eps: float = 1e-5,
        qk_bias: bool = False,
        rope_dim: int = 128,
        rope_freqs_for: str = "lang",
        rope_on_text: bool = True,
        shared_weights: bool = False,
        is_last_layer: bool = False,
        window: tuple[int, int, int] = (4, 3, 3),
        shift: bool = False,
    ):
        super().__init__()
        self.shared_weights = shared_weights
        self.is_last_layer = is_last_layer
        self.vid_dim = vid_dim
        self.txt_dim = txt_dim
        self.norm_eps = norm_eps

        self.attn = MMAttention(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            heads=heads,
            head_dim=head_dim,
            qk_bias=qk_bias,
            qk_norm_eps=norm_eps,
            rope_dim=rope_dim,
            rope_freqs_for=rope_freqs_for,
            rope_on_text=rope_on_text,
            shared_weights=shared_weights,
            window=window,
            shift=shift,
        )

        self.mlp = MMSwiGLU(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            expand_ratio=expand_ratio,
            shared_weights=shared_weights,
            is_last_layer=is_last_layer,
            mlp_type=mlp_type,
        )

        self.ada = AdaModulation(
            dim=vid_dim,
            shared_weights=shared_weights,
            is_last_layer=is_last_layer,
        )

    def __call__(
        self,
        vid: mx.array,
        txt: mx.array,
        emb: mx.array,
        vid_shape: mx.array,
        txt_shape: mx.array,
    ) -> tuple[mx.array, mx.array]:
        vid_attn = TransformerBlock._rms_norm(vid, self.norm_eps)
        txt_attn = TransformerBlock._rms_norm(txt, self.norm_eps)

        vid_attn = self.ada.modulate_vid(vid_attn, emb, layer="attn", mode="in")
        txt_attn = self.ada.modulate_txt(txt_attn, emb, layer="attn", mode="in")

        vid_attn, txt_attn = self.attn(vid_attn, txt_attn, vid_shape, txt_shape)

        vid_attn = self.ada.modulate_vid(vid_attn, emb, layer="attn", mode="out")
        txt_attn = self.ada.modulate_txt(txt_attn, emb, layer="attn", mode="out")

        vid = vid + vid_attn
        if not self.is_last_layer:
            txt = txt + txt_attn

        vid_mlp = TransformerBlock._rms_norm(vid, self.norm_eps)
        if self.is_last_layer:
            txt_mlp = txt
        else:
            txt_mlp = TransformerBlock._rms_norm(txt, self.norm_eps)

        vid_mlp = self.ada.modulate_vid(vid_mlp, emb, layer="mlp", mode="in")
        txt_mlp = self.ada.modulate_txt(txt_mlp, emb, layer="mlp", mode="in")

        vid_mlp, txt_mlp = self.mlp(vid_mlp, txt_mlp)

        vid_mlp = self.ada.modulate_vid(vid_mlp, emb, layer="mlp", mode="out")
        txt_mlp = self.ada.modulate_txt(txt_mlp, emb, layer="mlp", mode="out")

        vid = vid + vid_mlp
        if not self.is_last_layer:
            txt = txt + txt_mlp

        return vid, txt

    @staticmethod
    def _rms_norm(x: mx.array, eps: float = 1e-5) -> mx.array:
        return mx.fast.rms_norm(x, mx.ones(x.shape[-1]), eps)


# ----- transformer.py -----

class SeedVR2DiT(nn.Module, TransformerBase):
    def __init__(
        self,
        vid_in_channels: int = 33,
        vid_out_channels: int = 16,
        vid_dim: int = 2560,
        txt_in_dim: int = 5120,
        txt_dim: int | None = None,
        emb_dim: int | None = None,
        heads: int = 20,
        head_dim: int = 128,
        expand_ratio: int = 4,
        mlp_type: str = "swiglu",
        rope_on_text: bool = True,
        rope_freqs_for: str = "lang",
        use_output_ada: bool = True,
        last_layer_vid_only: bool = True,
        norm_eps: float = 1e-5,
        patch_size: tuple = (1, 2, 2),
        num_layers: int = 32,
        mm_layers: int = 10,
        rope_dim: int = 128,
        window: tuple[int, int, int] = (4, 3, 3),
    ):
        super().__init__()

        txt_dim = txt_dim if txt_dim is not None else vid_dim
        emb_dim = emb_dim if emb_dim is not None else 6 * vid_dim

        self.vid_dim = vid_dim
        self.txt_dim = txt_dim
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.mm_layers = mm_layers
        self.use_output_ada = use_output_ada
        self.last_layer_vid_only = last_layer_vid_only

        self.vid_in = PatchIn(
            in_channels=vid_in_channels,
            patch_size=patch_size,
            dim=vid_dim,
        )

        self.txt_in = nn.Linear(txt_in_dim, txt_dim)

        self.emb_in = TimeEmbedding(
            sinusoidal_dim=256,
            hidden_dim=max(vid_dim, txt_dim),
            output_dim=emb_dim,
        )

        self.blocks = []
        for i in range(num_layers):
            shared_weights = i >= mm_layers
            is_last_layer = self.last_layer_vid_only and i == num_layers - 1
            shift = i % 2 == 1

            self.blocks.append(
                TransformerBlock(
                    vid_dim=vid_dim,
                    txt_dim=txt_dim,
                    heads=heads,
                    head_dim=head_dim,
                    expand_ratio=expand_ratio,
                    mlp_type=mlp_type,
                    norm_eps=norm_eps,
                    qk_bias=False,
                    rope_dim=rope_dim,
                    rope_freqs_for=rope_freqs_for,
                    rope_on_text=rope_on_text,
                    shared_weights=shared_weights,
                    is_last_layer=is_last_layer,
                    window=window,
                    shift=shift,
                )
            )

        if use_output_ada:
            self.vid_out_norm = RMSNorm(vid_dim, eps=norm_eps)
            self.out_shift = mx.zeros((vid_dim,))
            self.out_scale = mx.ones((vid_dim,))

        self.vid_out = PatchOut(
            out_channels=vid_out_channels,
            patch_size=patch_size,
            dim=vid_dim,
        )

    def forward(self, *args, **kwargs):
        """MM-DiT 不走 ``ImagePipeline`` 的 ``latents`` 签名；热路径为 ``__call__(txt=, vid=, timestep=)``。"""
        raise RuntimeError(
            "SeedVR2DiT is an MM-DiT for SeedVR2 upscale: invoke as "
            "`dit(txt=pos_embeds, vid=concat_latents, timestep=t)` from "
            "`SeedVR2UpscalePipeline` (not `forward(latents, t, txt_embeds=...)`)."
        )

    def __call__(
        self,
        vid: mx.array,
        txt: mx.array,
        timestep: mx.array,
    ) -> mx.array:
        txt = self.txt_in(txt)
        txt_shape = mx.full((txt.shape[0], 1), txt.shape[1], dtype=mx.int32)
        vid, vid_shape = self.vid_in(vid)
        emb = self.emb_in(timestep)
        emb = emb.reshape(-1, self.vid_dim, 2, 3)

        for block in self.blocks:
            vid, txt = block(
                vid=vid,
                txt=txt,
                emb=emb,
                vid_shape=vid_shape,
                txt_shape=txt_shape,
            )

        if self.use_output_ada:
            vid = self.vid_out_norm(vid)
            vid = self._apply_out_ada(vid, emb=emb)
        vid, vid_shape = self.vid_out(vid, vid_shape)
        return vid

    def _apply_out_ada(self, hidden: mx.array, emb: mx.array) -> mx.array:
        shift_a = emb[:, :, 0, 0][:, None, :]
        scale_a = emb[:, :, 0, 1][:, None, :]
        return hidden * (scale_a + self.out_scale) + (shift_a + self.out_shift)
