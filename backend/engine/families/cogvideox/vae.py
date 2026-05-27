"""CogVideoX 3D causal VAE — 解码入口（潜变量 → PIL 帧，与 ``VideoPipeline`` 约定一致）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from backend.engine.common.vae.video_frames import ncthw_pixels_to_pil_frames

from .vae_mlx import decode_latents_ncthw


def decode_cogvideox_latents_to_pil_frames(
    ctx: Any,
    latents_bcthw: Any,
    bundle_root: Path | None,
    on_stage: Callable[[float], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    frame_batch_size: int = 2,
) -> list:
    """将 ``[B,C,T,H,W]`` 潜变量解码为 RGB ``PIL.Image`` 帧列表。"""
    if bundle_root is None:
        raise RuntimeError("CogVideoX VAE decode requires a local model bundle path.")

    pixels_ncthw = decode_latents_ncthw(
        ctx, latents_bcthw, bundle_root,
        on_stage=on_stage, on_log=on_log, frame_batch_size=frame_batch_size,
    )
    return ncthw_pixels_to_pil_frames(ctx, pixels_ncthw, model_label="CogVideoX")
