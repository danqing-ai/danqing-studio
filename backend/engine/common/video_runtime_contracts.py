"""Video-family runtime semantics — config-driven (no ``family ==`` in helpers)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def video_encoder_type(config: Any) -> str:
    enc = getattr(config, "encoder_type", None)
    if enc:
        return str(enc)
    fallback = getattr(config, "default_encoder_type", None)
    if fallback:
        return str(fallback)
    return "t5"


def video_t5_max_seq_len(config: Any) -> int:
    if hasattr(config, "max_text_seq_length"):
        return int(getattr(config, "max_text_seq_length", 226))
    if hasattr(config, "text_len"):
        return int(getattr(config, "text_len", 512))
    return 512


def _merge_hunyuan_bundle(config: Any, bundle_root: Path | None) -> None:
    from backend.engine.config.model_configs import (
        HunyuanVideoConfig,
        merge_hunyuan_transformer_config_from_bundle,
    )

    if isinstance(config, HunyuanVideoConfig):
        merge_hunyuan_transformer_config_from_bundle(config, bundle_root)


def _merge_wan_bundle(config: Any, bundle_root: Path | None) -> None:
    from backend.engine.config.model_configs import WanConfig, merge_wan_bundle_config

    if isinstance(config, WanConfig):
        merge_wan_bundle_config(config, bundle_root)


_BUNDLE_CONFIG_MERGE: dict[str, Callable[[Any, Path | None], None]] = {
    "hunyuan": _merge_hunyuan_bundle,
    "wan": _merge_wan_bundle,
}


def merge_video_bundle_config(config: Any, bundle_root: Path | None) -> None:
    merger = str(getattr(config, "bundle_config_merger", "") or "")
    if not merger:
        return
    fn = _BUNDLE_CONFIG_MERGE.get(merger)
    if fn is None:
        raise RuntimeError(f"Unknown video bundle_config_merger: {merger!r}")
    fn(config, bundle_root)


def video_scheduler_ctor_kwargs(
    config: Any,
    scheduler_name: str,
    bundle_root: Path | None,
) -> dict[str, Any]:
    extras = str(getattr(config, "scheduler_bundle_extras", "") or "")
    if not extras or scheduler_name != extras:
        return {}
    if extras == "wan_flow_unipc":
        ctor_kwargs: dict[str, Any] = {"num_train_timesteps": 1000}
        if bundle_root is not None:
            sched_cfg = bundle_root / "scheduler" / "scheduler_config.json"
            if sched_cfg.is_file():
                try:
                    data = json.loads(sched_cfg.read_text(encoding="utf-8"))
                    if "flow_shift" in data:
                        ctor_kwargs["shift"] = float(data["flow_shift"])
                except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
                    raise RuntimeError(
                        f"Wan: cannot read scheduler config {sched_cfg}: {e}"
                    ) from e
        return ctor_kwargs
    raise RuntimeError(f"Unknown video scheduler_bundle_extras: {extras!r}")


def video_cfg_negative_prompt(config: Any, negative_prompt: str | None) -> str:
    style = str(getattr(config, "cfg_negative_prompt_style", "default") or "default")
    if style == "wan":
        from backend.engine.families.wan.conditioning import WAN_SAMPLE_NEG_PROMPT

        return (negative_prompt or "").strip() or WAN_SAMPLE_NEG_PROMPT
    return negative_prompt.strip() if negative_prompt else " "


def video_resolve_shift_value(
    config: Any,
    *,
    request_shift: Any | None,
    registry_shift: Any | None,
    scheduler_default_shift: Any | None,
    on_log: Callable | None = None,
) -> float | None:
    if bool(getattr(config, "uses_wan_shift", False)):
        return resolve_wan_shift_value(
            request_shift=request_shift,
            registry_shift=registry_shift,
            scheduler_default_shift=scheduler_default_shift,
            on_log=on_log,
        )
    if request_shift is not None:
        return float(request_shift)
    if registry_shift is not None:
        return float(registry_shift)
    return None


def resolve_wan_shift_value(
    *,
    request_shift: Any | None,
    registry_shift: Any | None,
    scheduler_default_shift: Any | None,
    on_log: Callable | None = None,
) -> float | None:
    req = float(request_shift) if request_shift is not None else None
    reg = float(registry_shift) if registry_shift is not None else None
    sch = float(scheduler_default_shift) if scheduler_default_shift is not None else None
    eps = 1e-6

    if req is not None:
        return req
    if reg is not None:
        return reg
    if sch is not None and abs(sch - 1.0) > eps:
        return sch
    if on_log is not None:
        on_log("warning", "Wan shift unresolved; using scheduler default")
    return sch


def video_rotary_model_kwargs(
    config: Any,
    ctx: Any,
    pixel_h: int,
    pixel_w: int,
    latents: Any,
) -> dict[str, Any]:
    return {}


def video_infer_log_extras(config: Any, scheduler: Any, extra_cond: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    if bool(getattr(config, "uses_wan_shift", False)) and extra_cond.get("wan_expand_timesteps") is not None:
        parts.append(
            f"wan_expand_timesteps={bool(extra_cond.get('wan_expand_timesteps', False))}"
        )
    if bool(getattr(config, "uses_prediction_type", False)) and hasattr(scheduler, "_prediction_type"):
        parts.append(f"sched_prediction={scheduler._prediction_type}")
        parts.append(f"sched_spacing={getattr(scheduler, '_timestep_spacing', '?')}")
    return parts


def video_validate_generate_geometry(config: Any, w: int, h: int, num_frames: int) -> None:
    if w <= 0 or h <= 0:
        raise RuntimeError(f"invalid video size {w}x{h}")
    vae_sf = int(getattr(config, "vae_scale", 8) or 8)
    mode = str(getattr(config, "geometry_check", "generic") or "generic")
    tvs = int(getattr(config, "temporal_vae_scale", 0) or 0)

    if mode == "wan" and tvs > 0:
        if (num_frames - 1) % tvs != 0:
            raise RuntimeError(
                f"Wan requires (num_frames - 1) % {tvs} == 0; got {num_frames} "
                f"(valid examples: 81, 49 for temporal VAE scale {tvs})"
            )
        _, ph, pw = getattr(config, "patch_size", (1, 2, 2))
        step_h = vae_sf * int(ph)
        step_w = vae_sf * int(pw)
        if h % step_h != 0 or w % step_w != 0:
            raise RuntimeError(
                f"Wan requires height % {step_h} == 0 and width % {step_w} == 0 "
                f"(vae_scale={vae_sf}, patch_size={getattr(config, 'patch_size', (1, 2, 2))}); "
                f"got {w}x{h}. Example valid size: 480x704 for 480x720 intent."
            )
    if w % vae_sf != 0 or h % vae_sf != 0:
        raise RuntimeError(
            f"video width and height must be divisible by vae_scale={vae_sf}; got {w}x{h}"
        )
    if tvs > 0 and mode != "wan" and (num_frames - 1) % tvs != 0:
        raise RuntimeError(
            f"video requires (num_frames - 1) % {tvs} == 0; got {num_frames}"
        )


def video_snap_pixel_dims_if_needed(
    config: Any,
    w: int,
    h: int,
    *,
    on_log: Callable | None = None,
) -> tuple[int, int]:
    if not bool(getattr(config, "snap_pixel_dims", False)):
        return w, h
    from backend.engine.families.wan.conditioning import snap_wan_pixel_dims

    vae_scale = int(getattr(config, "vae_scale", 16) or 16)
    patch_size = tuple(getattr(config, "patch_size", (1, 2, 2)))
    ow, oh = w, h
    w, h = snap_wan_pixel_dims(w, h, vae_scale=vae_scale, patch_size=patch_size)
    if (w, h) != (ow, oh) and on_log is not None:
        on_log("info", f"Wan adjusted output size {ow}x{oh} -> {w}x{h} for patch/VAE alignment")
    return w, h


def wan_t5_bundle_paths(bundle_root: Path) -> tuple[str, str]:
    from backend.engine.families.wan.text_encoder_mlx import resolve_wan_umt5_pth

    assets = resolve_wan_umt5_pth(bundle_root)
    if assets is None:
        raise RuntimeError(
            f"Wan T5 assets not found under {bundle_root}. Expected "
            f"``models_t5_umt5-xxl-enc-bf16.pth`` (or ``models_t5*.pth``) and "
            f"``google/umt5-xxl`` tokenizer."
        )
    pth_path, tok_dir = assets
    return str(pth_path), str(tok_dir)


def _apply_hunyuan_i2v(ctx: Any, latents: Any, vae_latent: Any, extra_cond: dict[str, Any]) -> None:
    B, C, T, H, W = latents.shape
    cond_lat = ctx.zeros((B, C, T, H, W), dtype=latents.dtype)
    cond_lat = ctx.concat([vae_latent[:, :, :1, :, :], cond_lat[:, :, 1:, :, :]], axis=2)
    mask = ctx.zeros((B, 1, T, H, W), dtype=latents.dtype)
    mask_slice = ctx.ones((B, 1, 1, H, W), dtype=latents.dtype)
    mask = ctx.concat([mask_slice, mask[:, :, 1:, :, :]], axis=2)
    extra_cond["cond_latents"] = cond_lat
    extra_cond["mask_concat"] = mask
    extra_cond["i2v_mode"] = True


def _apply_wan_i2v(ctx: Any, latents: Any, vae_latent: Any, extra_cond: dict[str, Any]) -> None:
    from backend.engine.families.wan.conditioning import expand_wan_cond_latent, masks_like

    z = ctx.squeeze(vae_latent, 0)
    noise_cthw = ctx.squeeze(latents, 0)
    z = expand_wan_cond_latent(ctx, z, int(noise_cthw.shape[1]))
    _, mask2 = masks_like(ctx, [noise_cthw], zero=True)
    extra_cond["wan_cond_latent"] = z
    extra_cond["wan_i2v_mask"] = mask2[0]
    extra_cond["wan_i2v"] = True


def _apply_ltx_i2v(ctx: Any, latents: Any, vae_latent: Any, extra_cond: dict[str, Any]) -> Any:
    """LTX I2V: pass encoded latent to transformer hooks via extra_cond.

    Unlike concat/hunyuan/wan, LTX uses ``LTXLatentState`` + ``denoise_mask``
    managed by ``LTXTransformer.before_denoise`` / ``reblend_i2v_latents``.
    """
    extra_cond["ltx_i2v_cond_latent"] = vae_latent
    return latents


def video_apply_i2v_conditioning(
    config: Any,
    ctx: Any,
    latents: Any,
    vae_latent: Any,
    extra_cond: dict[str, Any],
) -> Any:
    style = str(getattr(config, "video_i2v_style", "concat") or "concat")
    if style == "hunyuan":
        _apply_hunyuan_i2v(ctx, latents, vae_latent, extra_cond)
        return latents
    if style == "wan":
        _apply_wan_i2v(ctx, latents, vae_latent, extra_cond)
        return latents
    if style == "ltx":
        return _apply_ltx_i2v(ctx, latents, vae_latent, extra_cond)
    if style == "concat":
        return ctx.concat([vae_latent[:, :, :1, :, :], latents[:, :, 1:, :, :]], axis=2)
    raise RuntimeError(f"Unknown video_i2v_style: {style!r}")


def video_i2v_encode_failure_message(config: Any) -> str:
    style = str(getattr(config, "video_i2v_style", "concat") or "concat")
    if style == "wan":
        return (
            "Wan image-to-video (animate) failed to VAE-encode the source image. "
            "Ensure the model bundle includes Wan2.2_VAE.pth (or vae/*.safetensors) "
            "and text_encoder assets under the bundle root."
        )
    if style == "ltx":
        return (
            "LTX image-to-video (animate) failed to VAE-encode the source image. "
            "Ensure the model bundle includes LTX Video VAE encoder weights "
            "(vae/ or vae_decoder.safetensors) under the bundle root."
        )
    return (
        "Image-to-video (animate) requires encoding the first RGB frame into "
        "video latents. DanQing does not yet implement the Lightricks "
        "`AutoencoderKLLTXVideo`-class encoder in MLX; first-frame conditioning "
        "cannot be applied. Use text-to-video (`create`) or add an MLX encoder "
        "port for your bundle VAE."
    )


def video_prepare_i2v_source_image(config: Any, src_img: Any, w: int, h: int) -> Any:
    if str(getattr(config, "video_i2v_style", "concat") or "concat") == "wan":
        from backend.engine.families.wan.conditioning import prepare_wan_reference_image

        return prepare_wan_reference_image(src_img, w, h)
    return src_img.resize((w, h))
