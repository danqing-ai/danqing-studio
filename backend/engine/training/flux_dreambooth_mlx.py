"""Flux.1 DreamBooth LoRA training on MLX (DanQing bundle + engine integration)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from PIL import Image

from backend.core.contracts import ExecutionContext, LoraTrainingRequest, ProgressEvent, LogEvent
from backend.engine._transformer_registry import get_transformer_class
from backend.engine.common.codecs.vae import VAEEncoder, infer_latent_channels, prepare_vae_encoder_weight_items
from backend.engine.config.model_configs import get_config_class
from backend.engine.contracts import local_bundle_root
from backend.engine.families.flux1.flux1_dual_mlx import Flux1TextEncoder
from backend.engine.families.flux1.weights import remap_flux1_lora_keys
from backend.engine.pipelines.image_model_load import load_image_transformer
from backend.engine.training.dataset_store import load_training_pairs, resize_rgb_image
from backend.engine.training.lora_layers import (
    add_grad_trees,
    apply_lora_to_flux1_dit,
    collect_lora_safetensors,
    prepare_dit_for_lora_training,
    scale_grad_tree,
)
from backend.engine.training.presets import merge_training_request_config, resolve_preset
from backend.engine.training.user_lora_registry import register_user_lora


def _log(ctx: ExecutionContext, level: str, message: str) -> None:
    ctx.on_log(LogEvent(level=level, message=message))  # type: ignore[arg-type]


def _progress(
    ctx: ExecutionContext,
    *,
    step: int,
    total: int,
    message: str = "",
    loss: float | None = None,
    phase: str = "training",
    progress: float | None = None,
) -> None:
    meta = f" loss={loss:.4f}" if loss is not None else ""
    frac = progress if progress is not None else min(1.0, step / max(total, 1))
    ctx.on_progress(
        ProgressEvent(
            progress=frac,
            step=step if phase == "training" else None,
            total=total if phase == "training" else None,
            message=(message or f"Training iteration {step}/{total}") + meta,
            phase=phase,
        )
    )


def _load_vae_encoder(ctx: Any, bundle_root: Path) -> VAEEncoder:
    from backend.engine.common.codecs.vae import load_vae_weight_dict, read_vae_dir_config

    vae_dir = bundle_root / "vae"
    vae_cfg, _, _ = read_vae_dir_config(vae_dir)
    vae_weights = load_vae_weight_dict(ctx, vae_dir)
    if not vae_weights:
        raise RuntimeError(f"VAE encode: no weights under {vae_dir}")
    scaling_factor = float(vae_cfg.get("scaling_factor", 1.0))
    shift_factor = float(vae_cfg.get("shift_factor", 0.0))
    latent_c = infer_latent_channels(vae_cfg, vae_weights)
    enc = VAEEncoder(
        latent_channels=latent_c,
        ctx=ctx,
        scaling_factor=scaling_factor,
        shift_factor=shift_factor,
    )
    enc.load_weights(prepare_vae_encoder_weight_items(vae_weights), strict=False)
    enc.cast_floating_params(ctx.bfloat16())
    return enc


def _encode_dataset(
    *,
    ctx: Any,
    pairs: list[tuple[Path, str]],
    vae: VAEEncoder,
    text_encoder: Flux1TextEncoder,
    resolution: tuple[int, int],
    num_augmentations: int,
    exec_ctx: ExecutionContext,
) -> tuple[list[Any], list[Any], list[Any]]:
    latents: list[Any] = []
    t5_feats: list[Any] = []
    clip_feats: list[Any] = []
    _log(exec_ctx, "info", f"Encoding {len(pairs)} images × {num_augmentations} augmentations …")
    import mlx.core as mx

    for img_path, prompt in pairs:
        for _ in range(num_augmentations):
            arr = resize_rgb_image(img_path, resolution)
            # NCHW [0,1] → [-1,1]
            nchw = mx.array(arr.transpose(2, 0, 1)[None].astype("float32"))
            n11 = nchw * 2.0 - 1.0
            z = vae.encode(n11)
            if getattr(z, "ndim", 0) == 5:
                z = z[:, :, 0, :, :]
            latents.append(z.astype(ctx.bfloat16()))
            t5, pooled = text_encoder.encode([prompt])
            mx.eval(z, t5, pooled)
            t5_feats.append(t5)
            clip_feats.append(pooled)
    return latents, t5_feats, clip_feats


def _training_loss(
    model: Any,
    x0: mx.array,
    t5: mx.array,
    clip_pooled: mx.array,
    guidance: mx.array,
    ctx: Any,
) -> mx.array:
    b = x0.shape[0]
    t = mx.random.uniform(shape=(b,), dtype=ctx.float32())
    eps = mx.random.normal(x0.shape, dtype=ctx.bfloat16())
    sigma = mx.reshape(t, (b, 1, 1, 1)).astype(ctx.bfloat16())
    x_t = (1.0 - sigma) * x0 + sigma * eps
    x_t = mx.stop_gradient(x_t)
    pred = model(
        x_t,
        timestep=0,
        txt_embeds=t5,
        pooled_embeds=clip_pooled,
        sigmas=t,
        guidance_scale=float(guidance[0]),
    )
    return mx.mean(mx.square(pred + x0 - eps))


def _save_adapter(path: Path, model: Any, rank: int, meta: dict[str, Any]) -> None:
    weights = collect_lora_safetensors(model, rank=rank)
    # Drop scalar helper before safetensors
    weights.pop("lora_rank", None)
    mx.save_safetensors(str(path), weights)
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _validate_saved_lora(path: Path, ctx: Any) -> None:
    from backend.engine.common.bundle.weights import load_safetensors

    flat = load_safetensors(str(path))
    remapped = remap_flux1_lora_keys(flat)
    if not remapped:
        raise RuntimeError(
            f"Saved LoRA {path} has no remappable (lora_down, lora_up) pairs for Flux.1"
        )


def _generate_progress_image(
    *,
    model: Any,
    vae_dec: Any,
    text_encoder: Flux1TextEncoder,
    prompt: str,
    resolution: tuple[int, int],
    guidance: float,
    ctx: Any,
    steps: int = 20,
) -> np.ndarray:
    """Quick sampling for training progress preview."""
    from backend.engine.common.ops.schedulers import FlowMatchEulerScheduler

    w, h = resolution
    lh, lw = h // 8, w // 8
    latents = mx.random.normal((1, 16, lh, lw), dtype=ctx.bfloat16())
    t5, pooled = text_encoder.encode([prompt])
    mx.eval(latents, t5, pooled)
    sched = FlowMatchEulerScheduler(num_train_timesteps=1000, shift=1.0, ctx=ctx)
    sched.set_timesteps(steps, mu=1.0)
    for i, t in enumerate(sched._timesteps):
        t_val = float(np.asarray(t).reshape(-1)[0]) if hasattr(t, "shape") else float(t)
        sigmas = mx.array([t_val], dtype=ctx.float32())
        pred = model(
            latents,
            timestep=i,
            txt_embeds=t5,
            pooled_embeds=pooled,
            sigmas=sigmas,
            guidance_scale=guidance,
            timestep_embed_value=t_val * 1000.0,
        )
        latents = sched.step(pred, t, latents)
        mx.eval(latents)
    # Decode with VAE decoder stub — use simple linear map for preview if full decode heavy
    from backend.engine.common.codecs.vae.decoder import VAEDecoder

    dec = vae_dec
    img = dec.forward(latents)
    mx.eval(img)
    arr = np.asarray(img[0].transpose(1, 2, 0), dtype=np.float32)
    arr = np.clip((arr + 1.0) * 0.5, 0, 1)
    return (arr * 255).astype(np.uint8)


def run_flux_dreambooth_training(
    request: LoraTrainingRequest,
    exec_ctx: ExecutionContext,
    *,
    registry: Any,
    project_root: Path,
    runtime: Any,
    path_resolver: Any,
) -> dict[str, Any]:
    if getattr(runtime, "backend", None) != "mlx":
        raise RuntimeError("LoRA training requires MLX runtime (Apple Silicon)")

    from backend.utils.path_utils import get_memory_gb

    mem_gb = get_memory_gb()
    if mem_gb > 0 and mem_gb < 48:
        raise RuntimeError(
            f"Flux.1 LoRA training requires ~50GB unified memory (detected {mem_gb:.0f}GB). "
            "Reduce resolution/lora_blocks or wait for QLoRA support."
        )

    base_model_id, version_key = request.base_model.split(":", 1) if ":" in request.base_model else (request.base_model, "")
    entry = registry.require(base_model_id)
    if str(getattr(entry, "family", "")) != "flux1":
        raise RuntimeError(
            f"Training MVP supports flux1-dev only (model {base_model_id!r} is family={entry.family!r})"
        )

    preset = resolve_preset(request.preset, base_model=request.base_model)
    cfg = merge_training_request_config(request, preset)
    iterations = int(cfg.get("iterations", 600))
    batch_size = int(cfg.get("batch_size", 1))
    lora_rank = int(cfg.get("lora_rank", 8))
    lora_blocks = int(cfg.get("lora_blocks") if cfg.get("lora_blocks") is not None else -1)
    learning_rate = float(cfg.get("learning_rate") or 1e-4)
    grad_accumulate = int(cfg.get("grad_accumulate") or 4)
    warmup_steps = int(cfg.get("warmup_steps") or 100)
    resolution_list = cfg.get("resolution") or [512, 512]
    resolution = (int(resolution_list[0]), int(resolution_list[1]))
    num_augmentations = int(cfg.get("num_augmentations") or 5)
    progress_prompt = (request.progress_prompt or "").strip()
    if not progress_prompt:
        raise RuntimeError("progress_prompt is required for LoRA training")
    progress_every = int(cfg.get("progress_every") or 300)
    progress_steps = int(cfg.get("progress_steps") or 20)
    checkpoint_every = int(cfg.get("checkpoint_every") or 300)
    guidance = float(cfg.get("guidance") or 4.0)

    ctx = runtime
    bundle_root = local_bundle_root(project_root, entry, version_key or None)
    if bundle_root is None or not bundle_root.is_dir():
        raise RuntimeError(f"Base model {base_model_id!r} is not installed")

    work_dir = Path(exec_ctx.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = work_dir / "adapters"
    adapter_dir.mkdir(exist_ok=True)

    pairs = load_training_pairs(project_root, request.dataset_id)
    if len(pairs) < 3:
        raise RuntimeError("Dataset must contain at least 3 images")

    config = get_config_class("flux1")()
    _log(exec_ctx, "info", "Loading VAE encoder and text encoders …")
    vae_enc = _load_vae_encoder(ctx, bundle_root)
    text_encoder = Flux1TextEncoder(ctx, bundle_root)

    latents, t5_feats, clip_feats = _encode_dataset(
        ctx=ctx,
        pairs=pairs,
        vae=vae_enc,
        text_encoder=text_encoder,
        resolution=resolution,
        num_augmentations=num_augmentations,
        exec_ctx=exec_ctx,
    )
    del vae_enc
    text_encoder.release_weights()
    ctx.clear_cache()

    _log(exec_ctx, "info", "Loading Flux.1 DiT …")
    model = load_image_transformer(
        ctx=ctx,
        family="flux1",
        config=config,
        entry=entry,
        version_key=version_key or None,
        project_root=project_root,
        model_cache=None,
        allow_cache=False,
    )
    if model is None:
        raise RuntimeError("Failed to load Flux.1 transformer from bundle")

    model, train_module = prepare_dit_for_lora_training(
        model,
        apply_lora_to_flux1_dit,
        rank=lora_rank,
        lora_blocks=lora_blocks,
    )

    warmup = optim.linear_schedule(0, learning_rate, warmup_steps)
    cosine = optim.cosine_decay(learning_rate, max(1, iterations // grad_accumulate))
    lr_schedule = optim.join_schedules([warmup, cosine], [warmup_steps])
    optimizer = optim.Adam(learning_rate=lr_schedule)

    xs = mx.concatenate(latents)
    t5_all = mx.concatenate(t5_feats)
    clip_all = mx.concatenate(clip_feats)
    mx.eval(xs, t5_all, clip_all)
    n_samples = len(latents)
    guidance_val = guidance

    loss_history: list[dict[str, float]] = []
    (work_dir / "loss_history.json").write_text("[]", encoding="utf-8")

    loss_and_grad = nn.value_and_grad(
        train_module,
        lambda x0, t5, clip_p: _training_loss(
            model,
            x0,
            t5,
            clip_p,
            mx.full((x0.shape[0],), guidance_val, dtype=ctx.bfloat16()),
            ctx,
        ),
    )

    _log(exec_ctx, "info", f"Training {iterations} iterations (rank={lora_rank}, batch={batch_size}) …")
    accum_grads: dict | None = None
    losses: list[float] = []
    tic = time.time()

    for i in range(iterations):
        exec_ctx.cancel_token.raise_if_cancelled()
        idx = int(mx.random.randint(0, n_samples, (1,)).item())
        x0 = xs[idx : idx + 1]
        cap_idx = idx // num_augmentations
        t5 = t5_all[cap_idx : cap_idx + 1]
        clip_p = clip_all[cap_idx : cap_idx + 1]
        loss, grads = loss_and_grad(x0, t5, clip_p)
        if accum_grads is None:
            accum_grads = grads
        else:
            accum_grads = add_grad_trees(accum_grads, grads)
        if (i + 1) % grad_accumulate == 0:
            scaled = scale_grad_tree(accum_grads, grad_accumulate)
            optimizer.update(train_module, scaled)
            accum_grads = None
        mx.eval(loss, train_module.parameters(), optimizer.state)
        losses.append(float(loss.item()))

        if (i + 1) % 10 == 0:
            avg = sum(losses) / len(losses)
            peak = mx.metal.get_peak_memory() / 1024**3
            _log(
                exec_ctx,
                "info",
                f"Iter {i + 1}/{iterations} loss={avg:.4f} peak_mem={peak:.1f}GB it/s={10 / max(time.time() - tic, 1e-6):.2f}",
            )
            loss_history.append({"step": i + 1, "loss": avg})
            (work_dir / "loss_history.json").write_text(
                json.dumps(loss_history), encoding="utf-8"
            )
            losses = []
            tic = time.time()

        _progress(exec_ctx, step=i + 1, total=iterations, loss=float(loss.item()))

        if (i + 1) % progress_every == 0:
            _log(exec_ctx, "info", f"Generating progress preview at step {i + 1} …")
            try:
                from backend.engine.common.codecs.vae import load_vae_weight_dict, read_vae_dir_config
                from backend.engine.common.codecs.vae.decoder import create_loaded_vae_decoder

                vae_dir = bundle_root / "vae"
                vae_cfg, _, _ = read_vae_dir_config(vae_dir)
                vae_weights = load_vae_weight_dict(ctx, vae_dir)
                dec, _, _, _ = create_loaded_vae_decoder(
                    ctx,
                    xs[0:1],
                    vae_weights,
                    float(vae_cfg.get("scaling_factor", 1.0)),
                    float(vae_cfg.get("shift_factor", 0.0)),
                )
                te = Flux1TextEncoder(ctx, bundle_root)
                preview = _generate_progress_image(
                    model=model,
                    vae_dec=dec,
                    text_encoder=te,
                    prompt=progress_prompt,
                    resolution=resolution,
                    guidance=guidance,
                    ctx=ctx,
                    steps=progress_steps,
                )
                out_png = work_dir / f"{i + 1:07d}_progress.png"
                Image.fromarray(preview).save(out_png)
                te.release_weights()
                ctx.clear_cache()
            except Exception as e:
                _log(exec_ctx, "warning", f"Progress preview failed: {e}")

        if (i + 1) % checkpoint_every == 0:
            ckpt = adapter_dir / f"{i + 1:07d}_adapters.safetensors"
            meta = {"iteration": i + 1, "lora_rank": lora_rank, "base_model": base_model_id}
            _save_adapter(ckpt, train_module, lora_rank, meta)

    final_path = adapter_dir / "final_adapters.safetensors"
    meta = {
        "iteration": iterations,
        "lora_rank": lora_rank,
        "base_model": base_model_id,
        "progress_prompt": progress_prompt,
    }
    _save_adapter(final_path, train_module, lora_rank, meta)
    _validate_saved_lora(final_path, ctx)

    output_name = (request.output_name or f"{base_model_id}-{request.dataset_id}").strip()
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in output_name)[:64]
    loras_dir = path_resolver.get_loras_dir()
    dest_dir = loras_dir / slug
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / "adapter.safetensors"
    import shutil

    shutil.copy2(final_path, dest_file)
    lora_config = {
        "lora_rank": lora_rank,
        "base_model": base_model_id,
        "alpha": lora_rank,
        "trigger_word": "",
    }
    (dest_dir / "lora_config.json").write_text(json.dumps(lora_config, indent=2), encoding="utf-8")

    user_lora_id = ""
    if request.auto_register:
        entry_row = register_user_lora(
            path_resolver.get_workspace_config_dir(),
            name=output_name,
            base_model=base_model_id,
            local_path=f"models/Lora/{slug}",
            lora_rank=lora_rank,
            task_id=exec_ctx.task_id,
        )
        user_lora_id = entry_row["id"]

    _log(exec_ctx, "success", f"Training complete → {dest_file}")
    return {
        "adapter_path": str(dest_file),
        "user_lora_id": user_lora_id,
        "output_name": slug,
        "loss_history": loss_history,
    }
