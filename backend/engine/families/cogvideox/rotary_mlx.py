"""CogVideoX 3D RoPE — MLX port of diffusers ``get_3d_rotary_pos_embed`` (``patch_size_t is None`` path).

Used when ``use_rotary_positional_embeddings`` is true (e.g. zai-org / current HF CogVideoX-5b).
"""
from __future__ import annotations

from typing import Any

from backend.engine.runtime._base import RuntimeContext


def get_resize_crop_region_for_grid(
    src: tuple[int, int],
    tgt_width: int,
    tgt_height: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Match ``diffusers.pipelines.cogvideo.pipeline_cogvideox.get_resize_crop_region_for_grid``."""
    tw = int(tgt_width)
    th = int(tgt_height)
    h, w = int(src[0]), int(src[1])
    r = h / w if w else 1.0
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))
    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))
    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


def _get_1d_rotary_pos_embed(
    ctx: RuntimeContext,
    dim: int,
    pos: Any,
    theta: float = 10000.0,
) -> tuple[Any, Any]:
    """``use_real=True``, ``repeat_interleave_real=True`` branch of diffusers ``get_1d_rotary_pos_embed``."""
    if dim % 2 != 0:
        raise ValueError(f"RoPE dim must be even, got {dim}")
    fp32 = ctx.float32()
    idx = ctx.arange(0, dim, 2, dtype=fp32)
    inv = ctx.power(ctx.array(theta, dtype=fp32), ctx.mul(idx, -1.0 / float(dim)))
    p = ctx.cast(pos, fp32)
    p = ctx.reshape(p, (-1,))
    freqs = ctx.mul(p[:, None], inv[None, :])
    c = ctx.cos(freqs)
    s = ctx.sin(freqs)
    c2 = ctx.reshape(ctx.stack([c, c], axis=-1), (c.shape[0], -1))
    s2 = ctx.reshape(ctx.stack([s, s], axis=-1), (s.shape[0], -1))
    return c2, s2


def _get_3d_rotary_pos_embed_linspace(
    ctx: RuntimeContext,
    embed_dim: int,
    crops_coords: tuple[tuple[int, int], tuple[int, int]],
    grid_size: tuple[int, int],
    temporal_size: int,
    theta: int = 10000,
) -> tuple[Any, Any]:
    """CogVideoX 1.0 path: ``grid_type == \"linspace\"`` (``patch_size_t is None``)."""
    start, stop = crops_coords
    grid_size_h, grid_size_w = int(grid_size[0]), int(grid_size[1])
    ts = int(temporal_size)
    if grid_size_h < 1 or grid_size_w < 1 or ts < 1:
        raise RuntimeError(
            f"CogVideoX RoPE: invalid grid temporal_size={ts} grid=({grid_size_h},{grid_size_w})"
        )

    fp32 = ctx.float32()
    grid_h = ctx.linspace(
        float(start[0]),
        float(stop[0]) * (grid_size_h - 1) / float(grid_size_h),
        grid_size_h,
        dtype=fp32,
    )
    grid_w = ctx.linspace(
        float(start[1]),
        float(stop[1]) * (grid_size_w - 1) / float(grid_size_w),
        grid_size_w,
        dtype=fp32,
    )
    grid_t = ctx.linspace(
        0.0,
        float(ts * (ts - 1) / ts) if ts else 0.0,
        ts,
        dtype=fp32,
    )

    dim_t = embed_dim // 4
    dim_h = embed_dim // 8 * 3
    dim_w = embed_dim // 8 * 3
    if dim_t + dim_h + dim_w != embed_dim:
        raise RuntimeError(
            f"CogVideoX RoPE: embed_dim {embed_dim} cannot be split as diffusers (t+h+w)."
        )

    t_cos, t_sin = _get_1d_rotary_pos_embed(ctx, dim_t, grid_t, float(theta))
    h_cos, h_sin = _get_1d_rotary_pos_embed(ctx, dim_h, grid_h, float(theta))
    w_cos, w_sin = _get_1d_rotary_pos_embed(ctx, dim_w, grid_w, float(theta))

    tc = ctx.reshape(t_cos, (ts, 1, 1, dim_t))
    tsin = ctx.reshape(t_sin, (ts, 1, 1, dim_t))
    hc = ctx.reshape(h_cos, (1, grid_size_h, 1, dim_h))
    hsin = ctx.reshape(h_sin, (1, grid_size_h, 1, dim_h))
    wc = ctx.reshape(w_cos, (1, 1, grid_size_w, dim_w))
    wsin = ctx.reshape(w_sin, (1, 1, grid_size_w, dim_w))

    cos = ctx.concat(
        [
            ctx.broadcast_to(tc, (ts, grid_size_h, grid_size_w, dim_t)),
            ctx.broadcast_to(hc, (ts, grid_size_h, grid_size_w, dim_h)),
            ctx.broadcast_to(wc, (ts, grid_size_h, grid_size_w, dim_w)),
        ],
        axis=-1,
    )
    sin = ctx.concat(
        [
            ctx.broadcast_to(tsin, (ts, grid_size_h, grid_size_w, dim_t)),
            ctx.broadcast_to(hsin, (ts, grid_size_h, grid_size_w, dim_h)),
            ctx.broadcast_to(wsin, (ts, grid_size_h, grid_size_w, dim_w)),
        ],
        axis=-1,
    )
    cos = ctx.reshape(cos, (ts * grid_size_h * grid_size_w, embed_dim))
    sin = ctx.reshape(sin, (ts * grid_size_h * grid_size_w, embed_dim))
    return cos, sin


def prepare_cogvideox_image_rotary_emb(
    ctx: RuntimeContext,
    cfg: Any,
    pixel_height: int,
    pixel_width: int,
    latent_num_frames: int,
    vae_spatial_scale: int,
) -> tuple[Any, Any]:
    """Build ``(cos, sin)`` for ``CogVideoXTransformer3D.forward(..., image_rotary_emb=...)``.

    Matches diffusers ``CogVideoXPipeline._prepare_rotary_positional_embeddings`` for ``patch_size_t is None``.
    """
    if getattr(ctx, "backend", None) != "mlx":
        raise RuntimeError("CogVideoX RoPE is implemented for MLX RuntimeContext only.")

    if getattr(cfg, "patch_size_t", None) is not None:
        raise RuntimeError(
            "CogVideoX RoPE: ``patch_size_t`` / CogVideoX 1.5 layout is not implemented in DanQing yet."
        )

    p = int(getattr(cfg, "patch_size", 2))
    vae_sf = int(vae_spatial_scale)
    grid_h = int(pixel_height) // (vae_sf * p)
    grid_w = int(pixel_width) // (vae_sf * p)
    base_w = int(getattr(cfg, "sample_width", 90)) // p
    base_h = int(getattr(cfg, "sample_height", 60)) // p
    embed_dim = int(getattr(cfg, "attention_head_dim", 64))

    crops = get_resize_crop_region_for_grid((grid_h, grid_w), base_w, base_h)
    cos, sin = _get_3d_rotary_pos_embed_linspace(
        ctx,
        embed_dim,
        crops_coords=crops,
        grid_size=(grid_h, grid_w),
        temporal_size=int(latent_num_frames),
        theta=10000,
    )
    if int(cos.shape[-1]) != embed_dim:
        raise RuntimeError(f"CogVideoX RoPE: expected last dim {embed_dim}, got {cos.shape}")
    seq = int(cos.shape[0])
    cos_b = ctx.reshape(cos, (1, 1, seq, embed_dim))
    sin_b = ctx.reshape(sin, (1, 1, seq, embed_dim))
    return cos_b, sin_b
