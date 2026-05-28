"""SeedVR2 视频修复：3D VAE + MM-DiT 时空 latent 推理（与逐帧图像超分路径分离）。

``ffmpeg`` 仅负责解码/封装；核心前向与 ``SeedVR2UpscalePipeline.generate_image`` 同源但输入为 ``(B,3,T,H,W)``。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np
from PIL import Image

from backend.engine.common.mlx_runtime_fallback import seeded_random_normal
from backend.engine.common.scale_factor import ScaleFactor
from backend.engine.common.vae.mlx_tiling import VAEUtil
from backend.engine.families.seedvr2.preprocess_mlx import (
    SeedVR2LatentCreator,
    SeedVR2PositiveEmbeddings,
    SeedVR2Util,
)

from .job_mlx import (
    SeedVR2UpscalePipeline,
    SeedVR2UpscaleRuntime,
    _UpscaleDenoiseCtx,
    _resolve_array_fn,
    _resolve_eval_fn,
    _resolve_seeded_randn_fn,
)


def _pad_stack_rgb_frames(
    frame_paths: list[Path],
    *,
    resolution: int | ScaleFactor,
    softness: float,
    array_fn: Callable[..., Any] = mx.array,
) -> tuple[mx.array, int, int, int, int]:
    """将若干帧对齐为同一空间尺寸后堆成 ``(1,3,T,H,W)``（[-1,1]）。

    返回 ``(volume, max_h, max_w, true_h0, true_w0)``：``true_*`` 为首帧内容尺寸，用于输出裁剪。
    """
    planes: list[mx.array] = []
    max_h = 0
    max_w = 0
    true_h0 = true_w0 = 0
    for i, p in enumerate(frame_paths):
        t4, th, tw = SeedVR2Util.preprocess_image(image_path=p, resolution=resolution, softness=softness)
        if i == 0:
            true_h0, true_w0 = int(th), int(tw)
        # (1,3,h,w)
        planes.append(t4[0])
        max_h = max(max_h, int(t4.shape[-2]))
        max_w = max(max_w, int(t4.shape[-1]))
    padded: list[mx.array] = []
    for x in planes:
        h, w = int(x.shape[-2]), int(x.shape[-1])
        canvas = mx.zeros((3, max_h, max_w), dtype=x.dtype)
        canvas = mx.slice_update(
            canvas,
            x,
            start_indices=array_fn([0, 0], dtype=mx.int32),
            axes=(1, 2),
        )
        padded.append(canvas)
    vol = mx.stack(padded, axis=0)
    vol = mx.transpose(vol, (1, 0, 2, 3))
    vol = vol[None, ...]
    return vol, max_h, max_w, true_h0, true_w0


def _decode_latents_to_frame_tensors(
    decoded: mx.array,
    *,
    true_h: int,
    true_w: int,
) -> list[mx.array]:
    """``decoded`` 为 ``(1,3,T,H,W)`` 或 ``(1,3,1,H,W)``，裁剪到内容尺寸后返回每帧 ``(3,h,w)``。"""
    if decoded.ndim == 4:
        decoded = decoded[:, :, None, :, :]
    _, _, t, _, _ = decoded.shape
    out: list[mx.array] = []
    for ti in range(int(t)):
        fr = decoded[:, :, ti : ti + 1, :, :]
        fr = fr[:, :, 0, :true_h, :true_w]
        out.append(fr[0])
    return out


def restore_video_chunk_spatiotemporal(
    *,
    pipeline: SeedVR2UpscalePipeline,
    frame_paths: list[Path],
    resolution: int | ScaleFactor,
    softness: float,
    seed: int,
    bundle_path: Path | None,
) -> list[mx.array]:
    """对一段连续帧做 SeedVR2 视频修复，返回每帧 RGB ``(3,h,w)`` 张量列表（[-1,1]）。"""
    if not frame_paths:
        return []
    eval_fn = _resolve_eval_fn(pipeline.dit)
    array_fn = _resolve_array_fn(pipeline.dit)
    seeded_randn_fn = _resolve_seeded_randn_fn(pipeline.dit)

    processed, max_h, max_w, true_h, true_w = _pad_stack_rgb_frames(
        frame_paths,
        resolution=resolution,
        softness=softness,
        array_fn=array_fn,
    )

    runtime = SeedVR2UpscaleRuntime.from_aligned_hw(
        model_config=pipeline.model_config,
        height=max_h,
        width=max_w,
        num_inference_steps=1,
        guidance=1.0,
        image_path=None,
        scheduler_key="seedvr2_euler",
    )

    initial_latent = VAEUtil.encode(vae=pipeline.vae, image=processed, tiling_config=pipeline.tiling_config)
    eval_fn(initial_latent)

    static_condition = SeedVR2LatentCreator.create_condition(encoded_latent=initial_latent)
    t_lat = int(initial_latent.shape[2])
    h_lat = int(initial_latent.shape[-2])
    w_lat = int(initial_latent.shape[-1])
    latents = seeded_random_normal(
        seeded_randn_fn,
        (1, 16, t_lat, h_lat, w_lat),
        int(seed) & 0x7FFFFFFF,
    )
    txt_pos = SeedVR2PositiveEmbeddings.load(bundle_path=bundle_path)

    ctx = _UpscaleDenoiseCtx()
    ctx.before_loop(latents)

    for t in runtime.time_steps:
        model_input = mx.concatenate([latents, static_condition], axis=1)
        noise = pipeline.dit(
            txt=txt_pos,
            vid=model_input,
            timestep=runtime.scheduler.timesteps[t],
        )
        latents = runtime.scheduler.step(noise=noise, timestep=t, latents=latents)
        ctx.in_loop(t, latents)
        eval_fn(latents)

    ctx.after_loop(latents)

    decoded = VAEUtil.decode(vae=pipeline.vae, latent=latents, tiling_config=pipeline.tiling_config)
    eval_fn(decoded)
    if decoded.ndim == 4:
        decoded = decoded[:, :, None, :, :]
    _, _, t_dec, _, _ = decoded.shape
    decoded = decoded[:, :, :, :true_h, :true_w]
    style = processed[:, :, :, :true_h, :true_w]
    corrected_slices: list[mx.array] = []
    for ti in range(int(t_dec)):
        d4 = decoded[:, :, ti, :, :]
        s4 = style[:, :, ti, :, :]
        corrected_slices.append(SeedVR2Util.apply_color_correction(d4, s4))
    decoded = mx.stack(corrected_slices, axis=2)

    return _decode_latents_to_frame_tensors(decoded, true_h=true_h, true_w=true_w)


def run_seedvr2_spatiotemporal_video(
    *,
    pipeline: SeedVR2UpscalePipeline,
    frames_dir: Path,
    n_frames: int,
    resolution: ScaleFactor,
    softness: float,
    seed_base: int,
    frames_out_dir: Path,
    png_pattern_name: str,
    chunk_frames: int,
    on_log: Callable[[str, str], None] | None,
    on_progress: Callable[[float, int, int], None] | None,
    is_cancelled: Callable[[], bool] | None,
) -> None:
    """读取 ``frames_dir/frame_%06d.png``，按 ``chunk_frames`` 做视频修复并写出 ``frames_out_dir/{pattern}%06d.png``。"""
    if chunk_frames < 4:
        raise RuntimeError("SeedVR2 video restoration requires chunk_frames >= 4")

    paths = [frames_dir / f"frame_{i:06d}.png" for i in range(1, n_frames + 1)]
    for p in paths:
        if not p.is_file():
            raise RuntimeError(f"Missing decoded frame: {p}")

    frames_out_dir.mkdir(parents=True, exist_ok=True)
    out_idx = 0
    bundle = pipeline._bundle_path

    for start in range(0, n_frames, chunk_frames):
        if is_cancelled and is_cancelled():
            return
        end = min(start + chunk_frames, n_frames)
        logical_paths = paths[start:end]
        infer_paths = list(logical_paths)
        while len(infer_paths) < 4:
            infer_paths.append(infer_paths[-1])
        sd = (int(seed_base) + start * 1009) & 0x7FFFFFFF
        t0 = time.perf_counter()
        frames_rgb = restore_video_chunk_spatiotemporal(
            pipeline=pipeline,
            frame_paths=[Path(x) for x in infer_paths],
            resolution=resolution,
            softness=softness,
            seed=sd,
            bundle_path=bundle,
        )
        frames_rgb = frames_rgb[: len(logical_paths)]
        if len(frames_rgb) != len(logical_paths):
            raise RuntimeError(
                f"SeedVR2 video restoration chunk produced {len(frames_rgb)} frame(s) for "
                f"{len(logical_paths)} input frame(s); VAE temporal length mismatch."
            )
        if on_log:
            on_log(
                "info",
                f"seedvr2_video_restoration chunk start={start} end={end} "
                f"frames={len(frames_rgb)} elapsed_s={time.perf_counter() - t0:.2f}",
            )
        for fr in frames_rgb:
            out_idx += 1
            arr = np.array(fr, dtype=np.float32)
            arr = (arr + 1.0) * 0.5
            arr = np.clip(arr, 0.0, 1.0)
            arr = np.transpose(arr, (1, 2, 0))
            rgb8 = (arr * 255.0).round().astype(np.uint8)
            im = Image.fromarray(rgb8, mode="RGB")
            im.save(str(frames_out_dir / f"{png_pattern_name}_{out_idx:06d}.png"))
        if on_progress:
            on_progress(end / max(n_frames, 1), end, n_frames)
