"""
位置编码与嵌入 — RoPE 2D/3D、Timestep Embedding、Patch Embedding。

所有模型经由 RuntimeContext 创建。
"""
from __future__ import annotations

import math
from typing import Any


def _ops_float32_dtype(ops: Any) -> Any:
    """``RuntimeContext.float32()`` or ``mlx.core.float32`` dtype."""
    dtype = getattr(ops, "float32", None)
    if dtype is None:
        raise RuntimeError("ops must expose float32 dtype or float32() callable")
    return dtype() if callable(dtype) else dtype


def build_position_ids_2d(ops: Any, batch_size: int, seq_len: int, *, dtype: Any) -> Any:
    """Build position ids ``[B, S]`` from ``arange(S)``."""
    pos = ops.arange(int(seq_len), dtype=dtype)[None, :]
    return ops.broadcast_to(pos, (int(batch_size), int(seq_len)))


def build_position_ids_3d_axes(ops: Any, batch_size: int, seq_len: int, *, dtype: Any) -> Any:
    """Build stacked position ids ``[3, B, S]`` for multimodal rotary paths."""
    pos2d = build_position_ids_2d(ops, batch_size, seq_len, dtype=dtype)
    return ops.broadcast_to(ops.expand_dims(pos2d, axis=0), (3, int(batch_size), int(seq_len)))


def pad_ragged_2d_sequences(
    ops: Any,
    sequences: list[Any],
    *,
    target_len: int | None = None,
    dtype: Any | None = None,
    pad_value: float = 0.0,
) -> Any:
    """Pad list of ``[L, D]`` tensors into ``[B, T, D]``."""
    if not sequences:
        raise RuntimeError("pad_ragged_2d_sequences requires non-empty sequences")
    max_len = max(int(s.shape[0]) for s in sequences)
    t = int(target_len) if target_len is not None else max_len
    padded: list[Any] = []
    for s in sequences:
        cur = int(s.shape[0])
        dim = int(s.shape[1])
        use = s[:t] if cur > t else s
        if cur < t:
            pad = ops.full((t - cur, dim), float(pad_value), dtype=s.dtype)
            use = ops.concat([use, pad], axis=0)
        padded.append(use)
    out = ops.stack(padded, axis=0)
    return out.astype(dtype) if dtype is not None else out


def pad_ragged_1d_sequences(
    ops: Any,
    sequences: list[Any],
    *,
    target_len: int | None = None,
    dtype: Any | None = None,
    pad_value: float = 0.0,
) -> Any:
    """Pad list of ``[L]`` tensors into ``[B, T]``."""
    if not sequences:
        raise RuntimeError("pad_ragged_1d_sequences requires non-empty sequences")
    max_len = max(int(s.shape[0]) for s in sequences)
    t = int(target_len) if target_len is not None else max_len
    padded: list[Any] = []
    for s in sequences:
        cur = int(s.shape[0])
        use = s[:t] if cur > t else s
        if cur < t:
            pad = ops.full((t - cur,), float(pad_value), dtype=s.dtype)
            use = ops.concat([use, pad], axis=0)
        padded.append(use)
    out = ops.stack(padded, axis=0)
    return out.astype(dtype) if dtype is not None else out


def pad_len_to_multiple(length: int, multiple: int = 32) -> int:
    """Return trailing pad length to align ``length`` to ``multiple``."""
    return (-int(length)) % int(multiple)


def build_tail_pad_mask(ops: Any, valid_len: int, pad_len: int, *, dtype: Any | None = None) -> Any:
    """Build boolean mask with ``False`` valid prefix and ``True`` padded tail."""
    mask_dtype = dtype if dtype is not None else _ops_float32_dtype(ops)
    valid = ops.zeros((int(valid_len),), dtype=mask_dtype) > 0
    if int(pad_len) <= 0:
        return valid
    tail = ops.ones((int(pad_len),), dtype=mask_dtype) > 0
    return ops.concat([valid, tail], axis=0)


def pad_tail_with_last(ops: Any, values: Any, pad_len: int) -> Any:
    """Pad tail by repeating the last token/value row ``pad_len`` times."""
    if int(pad_len) <= 0:
        return values
    return ops.concat([values, ops.repeat(values[-1:], int(pad_len), axis=0)], axis=0)


def apply_pad_token(ops: Any, embeds: Any, pad_mask: Any, pad_token: Any) -> Any:
    """Apply ``pad_token`` to positions selected by 1D ``pad_mask``."""
    return ops.where(ops.reshape(pad_mask, (-1, 1)), pad_token, embeds)


def _ops_transpose(ops: Any, x: Any, axes: tuple[int, ...]) -> Any:
    permute = getattr(ops, "permute", None)
    if callable(permute):
        return permute(x, axes)
    return ops.transpose(x, axes)


def apply_complex_rope_bshd(ops: Any, x: Any, cos_vals: Any, sin_vals: Any) -> Any:
    """Apply complex rotary embedding on ``[B, S, H, D]`` with separate cos/sin."""
    seq_len = int(x.shape[1])
    cos_vals = cos_vals[:seq_len]
    sin_vals = sin_vals[:seq_len]
    x_float = x.astype(_ops_float32_dtype(ops))
    x_pairs = ops.reshape(x_float, (*x.shape[:-1], -1, 2))
    x_real = x_pairs[..., 0]
    x_imag = x_pairs[..., 1]
    freqs_cos = cos_vals[None, :, None, :]
    freqs_sin = sin_vals[None, :, None, :]
    if freqs_cos.shape[-1] != x_real.shape[-1]:
        freqs_cos = freqs_cos[..., : x_real.shape[-1]]
        freqs_sin = freqs_sin[..., : x_real.shape[-1]]
    out_real = x_real * freqs_cos - x_imag * freqs_sin
    out_imag = x_real * freqs_sin + x_imag * freqs_cos
    out_pairs = ops.stack([out_real, out_imag], axis=-1)
    return ops.reshape(out_pairs, (*x.shape[:-1], -1)).astype(x.dtype)


def apply_complex_rope_bhsd(ops: Any, x: Any, cos_vals: Any, sin_vals: Any) -> Any:
    """Apply complex rotary embedding on ``[B, H, S, D]`` (Flux2 joint attention layout)."""
    x_bshd = _ops_transpose(ops, x, (0, 2, 1, 3))
    y_bshd = apply_complex_rope_bshd(ops, x_bshd, cos_vals, sin_vals)
    return _ops_transpose(ops, y_bshd, (0, 2, 1, 3))


def apply_complex_rope_from_cis_bshd(ops: Any, x: Any, freqs_cis: Any) -> Any:
    """Apply complex rotary embedding on ``[B, S, H, D]`` with ``[..., 2]`` cis table."""
    b, s, h, d = x.shape
    half = d // 2
    x_pairs = ops.reshape(x, (b, s, h, half, 2))
    freqs = ops.reshape(freqs_cis, (1, s, 1, half, 2))
    x_real = x_pairs[..., 0]
    x_imag = x_pairs[..., 1]
    c_real = freqs[..., 0]
    c_imag = freqs[..., 1]
    out_real = x_real * c_real - x_imag * c_imag
    out_imag = x_real * c_imag + x_imag * c_real
    out = ops.stack([out_real, out_imag], axis=-1)
    return ops.reshape(out, (b, s, h, d))


class TimestepEmbedding:
    """正弦时间步嵌入 → MLP 投影。

    用于: Flux1 / Flux2 / LTX / Wan 等几乎所有扩散模型。
    """

    def __init__(self, dim: int, ctx: Any, frequency_embedding_size: int = 256):
        self.ctx = ctx
        nn = ctx
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, timesteps):
        """timesteps: [B] 或 [B, 1]"""
        ctx = self.ctx
        half = self.frequency_embedding_size // 2
        freqs = ctx.exp(
            -ctx.log(ctx.full((half,), 10000.0))
            * ctx.arange(half, dtype=timesteps.dtype) / half
        )
        args = ctx.reshape(timesteps, (-1, 1)) * ctx.reshape(freqs, (1, -1))
        embedding = ctx.concat([ctx.cos(args), ctx.sin(args)], axis=-1)
        return self.mlp(embedding)

    def __call__(self, timesteps):
        return self.forward(timesteps)


def sinusoidal_embedding_1d(ctx: Any, dim: int, position: Any, *, base: float = 10000.0) -> Any:
    """Standard 1D sinusoidal embedding used by diffusion timestep paths."""
    if dim % 2 != 0:
        raise ValueError("dim must be even")
    half = dim // 2
    pos = position.astype(ctx.float32())
    freqs = ctx.power(
        ctx.array(float(base), dtype=ctx.float32()),
        -ctx.arange(half, dtype=ctx.float32()) / half,
    )
    sinusoid = ctx.outer(pos, freqs)
    return ctx.concat([ctx.cos(sinusoid), ctx.sin(sinusoid)], axis=-1)


def sinusoidal_timestep_proj(
    ctx: Any,
    timesteps: Any,
    embedding_dim: int,
    *,
    sin_first: bool = True,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 0.0,
    scale: float = 1.0,
    max_period: float = 10000.0,
) -> Any:
    """diffusers-style sinusoidal timestep projection (Flux1 / Hunyuan)."""
    if not hasattr(timesteps, "astype"):
        timesteps = ctx.array([float(timesteps)], dtype=ctx.float32())
    else:
        timesteps = timesteps.astype(ctx.float32())
    timesteps = ctx.reshape(timesteps, (-1,))
    half_dim = embedding_dim // 2
    denom = max(half_dim - downscale_freq_shift, 1e-8)
    exp_arg = -math.log(max_period) * ctx.arange(half_dim, dtype=ctx.float32()) / denom
    emb_freq = ctx.exp(exp_arg)
    emb = scale * timesteps[:, None] * emb_freq[None, :]
    if sin_first:
        emb = ctx.concat([ctx.sin(emb), ctx.cos(emb)], axis=-1)
    else:
        emb = ctx.concat([ctx.cos(emb), ctx.sin(emb)], axis=-1)
    if flip_sin_to_cos:
        emb = ctx.concat([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
    if embedding_dim % 2 == 1:
        z = ctx.zeros((emb.shape[0], 1), dtype=emb.dtype)
        emb = ctx.concat([emb, z], axis=-1)
    return emb


class TimestepEmbeddingMLP:
    """diffusers ``TimestepEmbedding``: linear → SiLU → linear (explicit layers for remap)."""

    def __init__(self, ctx: Any, in_channels: int, time_embed_dim: int):
        nn = ctx
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)

    def __call__(self, x: Any) -> Any:
        return self.linear_2(self.act(self.linear_1(x)))


class LTXTimestepEmbeddingMLP:
    """LTX sinusoidal timestep MLP — flat ``mlp_in`` / ``mlp_out`` keys (dual-platform ctx)."""

    def __init__(self, dim: int, ctx: Any, frequency_embedding_size: int = 256):
        nn = ctx
        self.ctx = ctx
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp_in = nn.Linear(frequency_embedding_size, dim, bias=True)
        self.mlp_out = nn.Linear(dim, dim, bias=True)

    def __call__(self, timesteps: Any) -> Any:
        ctx = self.ctx
        embedding = sinusoidal_timestep_proj(
            ctx,
            timesteps,
            self.frequency_embedding_size,
            flip_sin_to_cos=True,
        )
        embedding = self.mlp_in(embedding)
        embedding = ctx.silu(embedding)
        return self.mlp_out(embedding)


class RoPE2D:
    """2D 旋转位置编码 (图像模型)。

    用于: Flux1 / Flux2 / Qwen / FIBO / Z-Image。
    """

    def __init__(self, dim: int, ctx: Any, base: float = 10000.0):
        self.ctx = ctx
        self.dim = dim
        self.base = base

    def forward(self, height: int, width: int) -> tuple[Any, Any]:
        """返回 (cos, sin) 用于 Q/K 旋转。"""
        ctx = self.ctx
        half_dim = self.dim // 2
        freqs = ctx.exp(
            -ctx.log(ctx.full((half_dim // 2,), self.base))
            * ctx.arange(half_dim // 2, dtype=ctx.float32()) / (half_dim // 2)
        )

        # 2D grid
        h_pos = ctx.arange(height, dtype=ctx.float32())
        w_pos = ctx.arange(width, dtype=ctx.float32())
        h_grid, w_grid = ctx.meshgrid(h_pos, w_pos)
        h_grid_flat = ctx.reshape(h_grid, (-1, 1))
        w_grid_flat = ctx.reshape(w_grid, (-1, 1))

        h_freqs = ctx.reshape(h_grid_flat * freqs, (1, 1, height * width, half_dim // 2))
        w_freqs = ctx.reshape(w_grid_flat * freqs, (1, 1, height * width, half_dim // 2))

        freqs_concat = ctx.concat([h_freqs, w_freqs], axis=-1)
        cos = ctx.cos(freqs_concat)
        sin = ctx.sin(freqs_concat)
        return cos, sin

    def __call__(self, height: int, width: int) -> tuple[Any, Any]:
        return self.forward(height, width)


class RoPE3D:
    """3D 旋转位置编码 (视频模型)。

    用于: LTX / Wan。沿 T, H, W 三轴编码。
    """

    def __init__(self, dim: int, ctx: Any, base: float = 10000.0,
                 temporal_dim: int | None = None):
        self.ctx = ctx
        self.dim = dim
        self.base = base
        self.temporal_dim = temporal_dim or dim // 3

    def forward(self, num_frames: int, height: int, width: int) -> tuple[Any, Any]:
        """返回 (cos, sin) 用于 3D Q/K 旋转。"""
        ctx = self.ctx
        spatial_dim = (self.dim - self.temporal_dim) // 2

        # 时间频率
        t_freqs = ctx.exp(
            -ctx.log(ctx.full((self.temporal_dim // 2,), self.base))
            * ctx.arange(self.temporal_dim // 2, dtype=ctx.float32()) / (self.temporal_dim // 2)
        )
        # 空间频率
        s_freqs = ctx.exp(
            -ctx.log(ctx.full((spatial_dim // 2,), self.base))
            * ctx.arange(spatial_dim // 2, dtype=ctx.float32()) / (spatial_dim // 2)
        )

        t_pos = ctx.arange(num_frames, dtype=ctx.float32())
        h_pos = ctx.arange(height, dtype=ctx.float32())
        w_pos = ctx.arange(width, dtype=ctx.float32())

        t_grid, h_grid, w_grid = ctx.meshgrid(t_pos, h_pos, w_pos)
        total = num_frames * height * width

        t_flat = ctx.reshape(t_grid, (total, 1))
        h_flat = ctx.reshape(h_grid, (total, 1))
        w_flat = ctx.reshape(w_grid, (total, 1))

        t_freq = ctx.reshape(t_flat * t_freqs, (1, 1, total, self.temporal_dim // 2))
        h_freq = ctx.reshape(h_flat * s_freqs, (1, 1, total, spatial_dim // 2))
        w_freq = ctx.reshape(w_flat * s_freqs, (1, 1, total, spatial_dim // 2))

        freqs = ctx.concat([t_freq, h_freq, w_freq], axis=-1)
        cos = ctx.cos(freqs)
        sin = ctx.sin(freqs)
        return cos, sin

    def __call__(self, num_frames: int, height: int, width: int) -> tuple[Any, Any]:
        return self.forward(num_frames, height, width)


def _factorized_rope_dims(half_d: int) -> tuple[int, int, int]:
    d_t = half_d - 2 * (half_d // 3)
    d_h = half_d // 3
    d_w = half_d // 3
    return d_t, d_h, d_w


def factorized_rope_params(
    ops: Any,
    max_seq_len: int,
    dim: int,
    *,
    theta: float = 10000.0,
) -> Any:
    """Precompute factorized RoPE frequencies as ``[L, dim//2, 2]``."""
    if dim % 2 != 0:
        raise ValueError("rope dim must be even")
    import numpy as np

    freqs = (
        np.arange(max_seq_len, dtype=np.float64)[:, None]
        * (1.0 / np.power(theta, np.arange(0, dim, 2, dtype=np.float64) / dim)[None, :])
    )
    return ops.array(np.stack([np.cos(freqs), np.sin(freqs)], axis=-1).astype(np.float32))


def factorized_rope_concat_params(
    ops: Any,
    max_seq_len: int,
    dims: list[int],
    *,
    theta: float = 10000.0,
) -> Any:
    """Concatenate per-axis factorized RoPE params into one ``[L, sum(d//2), 2]`` table."""
    if not dims:
        raise RuntimeError("factorized_rope_concat_params requires non-empty dims")
    parts = [factorized_rope_params(ops, max_seq_len, int(d), theta=theta) for d in dims]
    return ops.concat(parts, axis=1)


def bernini_source_id_cos_sin(
    ops: Any,
    head_dim: int,
    source_id: float,
    *,
    theta: float = 10000.0,
    dtype: Any | None = None,
) -> tuple[Any, Any]:
    """SA-3D RoPE source-id phase as ``(cos, sin)`` with shape ``[1, head_dim//2]``."""
    if head_dim % 2 != 0:
        raise ValueError("bernini_source_id_cos_sin requires even head_dim")
    import numpy as np

    half = head_dim // 2
    pos = float(source_id)
    inv_freq = 1.0 / np.power(theta, np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    freqs = pos * inv_freq
    cos = ops.array(np.cos(freqs).astype(np.float32))
    sin = ops.array(np.sin(freqs).astype(np.float32))
    if dtype is not None:
        cos = cos.astype(dtype)
        sin = sin.astype(dtype)
    return cos, sin


def bernini_modulate_rope_cos_sin(
    ops: Any,
    cos_spatial: Any,
    sin_spatial: Any,
    source_id: float,
    *,
    theta: float = 10000.0,
) -> tuple[Any, Any]:
    """Multiply spatial RoPE by per-source-id phase (Bernini SA-3D RoPE)."""
    head_dim = int(cos_spatial.shape[-1]) * 2
    cos_id, sin_id = bernini_source_id_cos_sin(
        ops, head_dim, source_id, theta=theta, dtype=cos_spatial.dtype,
    )
  # Broadcast source-id across sequence positions.
    cos_id = ops.broadcast_to(cos_id.reshape(1, 1, -1), cos_spatial.shape)
    sin_id = ops.broadcast_to(sin_id.reshape(1, 1, -1), sin_spatial.shape)
    out_cos = cos_spatial * cos_id - sin_spatial * sin_id
    out_sin = sin_spatial * cos_id + cos_spatial * sin_id
    return out_cos, out_sin


def factorized_rope_precompute_cos_sin(
    ops: Any,
    grid_sizes: list[tuple[int, int, int]],
    freqs: Any,
    *,
    dtype: Any,
) -> tuple[Any, Any]:
    """Precompute ``(cos, sin)`` for a fixed 3D grid (all batch items identical)."""
    if freqs.dtype != dtype:
        freqs = freqs.astype(dtype)

    f, h, w = grid_sizes[0]
    seq_len = int(f * h * w)
    half_d = int(freqs.shape[1])
    d_t, d_h, d_w = _factorized_rope_dims(half_d)

    freqs_t = freqs[:, :d_t]
    freqs_h = freqs[:, d_t : d_t + d_h]
    freqs_w = freqs[:, d_t + d_h : d_t + d_h + d_w]

    ft = ops.broadcast_to(freqs_t[:f].reshape(f, 1, 1, d_t, 2), (f, h, w, d_t, 2))
    fh = ops.broadcast_to(freqs_h[:h].reshape(1, h, 1, d_h, 2), (f, h, w, d_h, 2))
    fw = ops.broadcast_to(freqs_w[:w].reshape(1, 1, w, d_w, 2), (f, h, w, d_w, 2))

    freqs_i = ops.concatenate([ft, fh, fw], axis=3).reshape(seq_len, 1, half_d, 2)
    return freqs_i[..., 0], freqs_i[..., 1]


def factorized_rope_apply(
    ops: Any,
    x: Any,
    grid_sizes: list[tuple[int, int, int]],
    freqs: Any,
    *,
    precomputed_cos_sin: tuple[Any, Any] | None = None,
) -> Any:
    """Apply factorized 3-way RoPE on ``[B, L, num_heads, head_dim]``."""
    b, s, _n, d = x.shape
    half_d = d // 2

    if precomputed_cos_sin is not None:
        cos_f, sin_f = precomputed_cos_sin
        rope_len = int(cos_f.shape[0])
        if rope_len != s:
            f0, h0, w0 = grid_sizes[0]
            grid_len = int(f0 * h0 * w0)
            raise RuntimeError(
                f"factorized_rope_apply: precomputed RoPE length {rope_len} != "
                f"token sequence {s} (single-grid length {grid_len}, grid={grid_sizes[0]})"
            )
        seq_len = rope_len
        all_same = all(grid_sizes[i] == grid_sizes[0] for i in range(1, b)) if b > 1 else True

        if all_same:
            x_seq = x.reshape(b, seq_len, -1, half_d, 2)
            x_real = x_seq[..., 0]
            x_imag = x_seq[..., 1]
            out_real = x_real * cos_f - x_imag * sin_f
            out_imag = x_real * sin_f + x_imag * cos_f
            return ops.stack([out_real, out_imag], axis=-1).reshape(b, seq_len, -1, d)

        outputs = []
        for i in range(b):
            x_i = x[i, :seq_len].reshape(seq_len, -1, half_d, 2)
            x_real = x_i[..., 0]
            x_imag = x_i[..., 1]
            out_real = x_real * cos_f - x_imag * sin_f
            out_imag = x_real * sin_f + x_imag * cos_f
            outputs.append(
                ops.stack([out_real, out_imag], axis=-1).reshape(seq_len, -1, d)
            )
        return ops.stack(outputs)

    if freqs.dtype != x.dtype:
        freqs = freqs.astype(x.dtype)

    d_t, d_h, d_w = _factorized_rope_dims(half_d)
    freqs_t = freqs[:, :d_t]
    freqs_h = freqs[:, d_t : d_t + d_h]
    freqs_w = freqs[:, d_t + d_h : d_t + d_h + d_w]

    outputs = []
    for i in range(b):
        f, h, w = grid_sizes[i]
        seq_len = int(f * h * w)
        x_i = x[i, :seq_len].reshape(seq_len, -1, half_d, 2)

        ft = ops.broadcast_to(freqs_t[:f].reshape(f, 1, 1, d_t, 2), (f, h, w, d_t, 2))
        fh = ops.broadcast_to(freqs_h[:h].reshape(1, h, 1, d_h, 2), (f, h, w, d_h, 2))
        fw = ops.broadcast_to(freqs_w[:w].reshape(1, 1, w, d_w, 2), (f, h, w, d_w, 2))
        freqs_i = ops.concatenate([ft, fh, fw], axis=3).reshape(seq_len, 1, half_d, 2)
        cos_f = freqs_i[..., 0]
        sin_f = freqs_i[..., 1]

        x_real = x_i[..., 0]
        x_imag = x_i[..., 1]
        out_real = x_real * cos_f - x_imag * sin_f
        out_imag = x_real * sin_f + x_imag * cos_f
        x_rotated = ops.stack([out_real, out_imag], axis=-1).reshape(seq_len, -1, d)
        if seq_len < s:
            x_rotated = ops.concatenate([x_rotated, x[i, seq_len:]], axis=0)
        outputs.append(x_rotated)

    return ops.stack(outputs)


class PatchEmbed2D:
    """2D Patch Embedding (图像模型)。

    用于: Flux1 / Flux2 / Qwen / FIBO / Z-Image VAE latent → 展平为 token 序列。
    """

    def __init__(self, in_channels: int, dim: int, patch_size: int = 1, ctx: Any = None):
        self.ctx = ctx
        nn = ctx
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size,
                              stride=patch_size, bias=True)

    def forward(self, x) -> Any:
        ctx = self.ctx
        # Pipeline latents are NCHW [B,C,H,W]; MLX Conv2d expects NHWC [B,H,W,C].
        x = ctx.permute(x, (0, 2, 3, 1))
        x = self.proj(x)
        B, H, W, C = x.shape
        x = ctx.reshape(x, (B, H * W, C))
        return x

    def __call__(self, x):
        return self.forward(x)


class PatchEmbed3D:
    """3D Patch Embedding (视频模型)。

    用于: LTX / Wan VAE latent → 展平为时空 token 序列。
    """

    def __init__(self, in_channels: int, dim: int,
                 patch_size: tuple = (1, 2, 2), ctx: Any = None):
        self.ctx = ctx
        nn = ctx
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, dim, kernel_size=patch_size,
                              stride=patch_size, bias=True)

    def forward(self, x) -> Any:
        ctx = self.ctx
        # Pipeline latents are NCTHW [B,C,T,H,W]; MLX Conv3d expects NDHWC [B,T,H,W,C].
        x = ctx.permute(x, (0, 2, 3, 4, 1))  # [B, C, T, H, W] → [B, T, H, W, C]
        x = self.proj(x)  # [B, T, H, W, C] → [B, T/pt, H/ph, W/pw, dim]
        B, T, H, W, C = x.shape
        total_tokens = T * H * W
        x = ctx.reshape(x, (B, total_tokens, C))
        return x

    def __call__(self, x):
        return self.forward(x)
