#!/usr/bin/env python3
"""Parity checks: DanQing CogVideoX vs diffusers (scheduler / RoPE / DiT one-step).

Usage (from repo root, MLX venv):
  python scripts/compare_cogvideox_diffusers.py
  python scripts/compare_cogvideox_diffusers.py --dit-forward   # loads DiT (~18GB); slow
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BUNDLE = REPO / "models" / "Video" / "cogvideox-5b-fp16"
HEIGHT, WIDTH, NUM_FRAMES = 480, 720, 49
STEPS = 50
SEED = 42


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def _ok(msg: str) -> None:
    print(f"OK: {msg}")


def compare_scheduler() -> None:
    from diffusers.schedulers.scheduling_dpm_cogvideox import CogVideoXDPMScheduler as RefSched

    from backend.engine.config.model_configs import cogvideox_scheduler_kwargs_from_bundle
    from backend.engine.common.schedulers import CogVideoXDPMScheduler
    from backend.engine.runtime.mlx import MLXContext

    if not BUNDLE.is_dir():
        _fail(f"bundle missing: {BUNDLE}")

    kw = cogvideox_scheduler_kwargs_from_bundle(BUNDLE)
    ref = RefSched.from_pretrained(str(BUNDLE / "scheduler"))
    ref_cfg = ref.config

    checks = {
        "prediction_type": (kw["prediction_type"], ref_cfg.prediction_type),
        "timestep_spacing": (kw["timestep_spacing"], ref_cfg.timestep_spacing),
        "snr_shift_scale": (kw["snr_shift_scale"], float(ref_cfg.snr_shift_scale)),
        "rescale_betas_zero_snr": (kw["rescale_betas_zero_snr"], bool(ref_cfg.rescale_betas_zero_snr)),
    }
    for name, (ours, theirs) in checks.items():
        if ours != theirs:
            _fail(f"scheduler {name}: ours={ours!r} ref={theirs!r}")
    _ok(f"scheduler config matches diffusers ({checks})")

    ctx = MLXContext()
    ours = CogVideoXDPMScheduler(ctx=ctx, **kw)
    ours.set_timesteps(STEPS)
    ref.set_timesteps(STEPS)
    ref_ts = ref.timesteps.cpu().numpy().astype(int).tolist()
    if ref_ts != ours._timesteps_list:
        _fail(f"timesteps mismatch at 50 steps:\n  ref={ref_ts[:5]}...{ref_ts[-3:]}\n  ours={ours._timesteps_list[:5]}...{ours._timesteps_list[-3:]}")
    _ok(f"50-step timesteps match (first={ref_ts[0]}, last={ref_ts[-1]})")


def compare_rope() -> None:
    import torch
    from diffusers.models.embeddings import get_3d_rotary_pos_embed
    from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid

    import json
    from backend.engine.config.model_configs import CogVideoXConfig, merge_cogvideox_transformer_config_from_bundle
    from backend.engine.families.cogvideox.rotary_mlx import prepare_cogvideox_image_rotary_emb
    from backend.engine.runtime.mlx import MLXContext

    cfg = CogVideoXConfig()
    merge_cogvideox_transformer_config_from_bundle(cfg, BUNDLE)
    ctx = MLXContext()

    latent_frames = (NUM_FRAMES - 1) // cfg.temporal_compression_ratio + 1
    vae_sf = 8
    p = cfg.patch_size
    grid_h = HEIGHT // (vae_sf * p)
    grid_w = WIDTH // (vae_sf * p)
    base_w = cfg.sample_width // p
    base_h = cfg.sample_height // p
    crops = get_resize_crop_region_for_grid((grid_h, grid_w), base_w, base_h)

    ref_cos, ref_sin = get_3d_rotary_pos_embed(
        embed_dim=cfg.attention_head_dim,
        crops_coords=crops,
        grid_size=(grid_h, grid_w),
        temporal_size=latent_frames,
        device=torch.device("cpu"),
    )
    ours_cos, ours_sin = prepare_cogvideox_image_rotary_emb(
        ctx, cfg, HEIGHT, WIDTH, latent_frames, vae_sf,
    )
    oc = np.asarray(ours_cos)[0, 0]
    os_ = np.asarray(ours_sin)[0, 0]
    rc = ref_cos.cpu().numpy()
    rs = ref_sin.cpu().numpy()
    if oc.shape != rc.shape:
        _fail(f"RoPE shape mismatch cos {oc.shape} vs {rc.shape}")
    cos_err = float(np.max(np.abs(oc - rc)))
    sin_err = float(np.max(np.abs(os_ - rs)))
    if cos_err > 1e-4 or sin_err > 1e-4:
        _fail(f"RoPE max abs err too large: cos={cos_err:.2e} sin={sin_err:.2e}")
    _ok(f"RoPE cos/sin match (max err cos={cos_err:.2e} sin={sin_err:.2e}, seq={rc.shape[0]})")


def compare_scheduler_step() -> None:
    """One DPM++ step with fixed tensors — catches v_prediction / mult formula drift."""
    import torch
    from diffusers.schedulers.scheduling_dpm_cogvideox import CogVideoXDPMScheduler as RefSched

    from backend.engine.config.model_configs import cogvideox_scheduler_kwargs_from_bundle
    from backend.engine.common.schedulers import CogVideoXDPMScheduler
    from backend.engine.runtime.mlx import MLXContext

    kw = cogvideox_scheduler_kwargs_from_bundle(BUNDLE)
    ref = RefSched.from_pretrained(str(BUNDLE / "scheduler"))
    ref.set_timesteps(STEPS)
    ctx = MLXContext()
    ours = CogVideoXDPMScheduler(ctx=ctx, **kw)
    ours.set_timesteps(STEPS)

    shape = (1, 16, 13, 60, 90)
    torch.manual_seed(SEED)
    latents = torch.randn(shape, dtype=torch.float32)
    noise_pred = torch.randn(shape, dtype=torch.float32)
    t = int(ref.timesteps[10].item())
    t_back = int(ref.timesteps[9].item()) if 9 >= 0 else None

    with torch.no_grad():
        ref_next, ref_x0 = ref.step(
            noise_pred, None, t, t_back, latents, return_dict=False,
        )
    ours_next, ours_x0 = ours.step(
        ctx.array(noise_pred.numpy()),
        t,
        ctx.array(latents.numpy()),
        old_pred_original_sample=None,
        timestep_back=t_back,
    )
    ctx.eval(ours_next, ours_x0)
    n_err = float(np.max(np.abs(ref_next.numpy() - np.asarray(ours_next))))
    x0_err = float(np.max(np.abs(ref_x0.numpy() - np.asarray(ours_x0))))
    if x0_err > 1e-4:
        _fail(f"scheduler v_prediction x0 mismatch: err={x0_err:.2e}")
    _ok(
        f"DPM++ step @ t={t}: pred_original_sample matches diffusers (err={x0_err:.2e}); "
        f"prev_sample differs by noise draw unless variance_noise is fixed (err={n_err:.2e})"
    )


def compare_dit_one_step() -> None:
    import gc

    import torch
    from diffusers.models import CogVideoXTransformer3DModel
    from diffusers.models.embeddings import get_3d_rotary_pos_embed
    from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid

    from backend.engine.config.model_configs import CogVideoXConfig, merge_cogvideox_transformer_config_from_bundle
    from backend.engine.families.cogvideox.transformer_mlx import CogVideoXTransformer3D
    from backend.engine.families.cogvideox.weights import remap_cogvideox_weights
    from backend.engine.families.cogvideox.rotary_mlx import prepare_cogvideox_image_rotary_emb
    from backend.engine.pipelines.video_bundle_layout import resolve_video_transformer_weight_sources
    from backend.engine.runtime.mlx import MLXContext

    cfg = CogVideoXConfig()
    merge_cogvideox_transformer_config_from_bundle(cfg, BUNDLE)
    latent_frames = (NUM_FRAMES - 1) // cfg.temporal_compression_ratio + 1
    lh, lw = HEIGHT // 8, WIDTH // 8

    torch.manual_seed(SEED)
    latents_ref = torch.randn(1, latent_frames, cfg.in_channels, lh, lw, dtype=torch.float32)
    timestep = torch.tensor([500], dtype=torch.float32)
    txt = torch.randn(1, cfg.max_text_seq_length, cfg.text_dim, dtype=torch.float32)

    p = cfg.patch_size
    grid_h = HEIGHT // (8 * p)
    grid_w = WIDTH // (8 * p)
    crops_coords = get_resize_crop_region_for_grid(
        (grid_h, grid_w), cfg.sample_width // p, cfg.sample_height // p,
    )
    rope = get_3d_rotary_pos_embed(
        embed_dim=cfg.attention_head_dim,
        crops_coords=crops_coords,
        grid_size=(grid_h, grid_w),
        temporal_size=latent_frames,
    )

    ref_model = CogVideoXTransformer3DModel.from_pretrained(
        str(BUNDLE / "transformer"), torch_dtype=torch.float32,
    )
    ref_model.eval()
    with torch.no_grad():
        ref_out = ref_model(
            hidden_states=latents_ref,
            encoder_hidden_states=txt,
            timestep=timestep,
            image_rotary_emb=rope,
            return_dict=False,
        )[0].cpu().numpy()

    latents_np = latents_ref.numpy()
    txt_np = txt.numpy()
    del ref_model, latents_ref, txt, rope
    gc.collect()

    ctx = MLXContext()
    ours = CogVideoXTransformer3D(cfg, ctx, num_frames=latent_frames)
    _, shard_paths = resolve_video_transformer_weight_sources(BUNDLE, "cogvideox", "cogvideox-5b")
    w: dict = {}
    for sf in shard_paths:
        w.update(ctx.load_weights(str(sf)))
    w = remap_cogvideox_weights(w)
    ours.load_weights(list(w.items()), strict=False)
    ctx.eval(*[p for _, p in ours.parameters()])

    latents_ours = np.transpose(latents_np, (0, 2, 1, 3, 4))
    txt_mx = ctx.array(txt_np)
    t_mx = ctx.array(timestep.numpy())
    cos_sin = prepare_cogvideox_image_rotary_emb(ctx, cfg, HEIGHT, WIDTH, latent_frames, 8)
    ours_out = np.asarray(ours(latents_ours, t_mx, txt_embeds=txt_mx, image_rotary_emb=cos_sin))
    ours_out_btc = np.transpose(ours_out, (0, 2, 1, 3, 4))

    err = float(np.max(np.abs(ref_out - ours_out_btc)))
    rel = err / max(float(np.max(np.abs(ref_out))), 1e-8)
    if err > 0.05 and rel > 0.01:
        _fail(f"DiT one-step max abs err={err:.4f} rel={rel:.4f} (threshold 0.05 abs or 1% rel)")
    _ok(f"DiT one-step forward close (max abs err={err:.4f}, rel={rel:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="DanQing CogVideoX vs diffusers parity")
    parser.add_argument("--dit-forward", action="store_true", help="Run DiT one-step (loads full transformer)")
    args = parser.parse_args()

    print(f"bundle={BUNDLE}")
    compare_scheduler()
    compare_scheduler_step()
    compare_rope()
    if args.dit_forward:
        compare_dit_one_step()
    else:
        print("SKIP: DiT one-step (pass --dit-forward to load ~18GB transformer)")
    print("All requested checks passed.")


if __name__ == "__main__":
    main()
