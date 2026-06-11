"""Z-Image Base DreamBooth LoRA training on MLX."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from PIL import Image

from backend.core.contracts import ExecutionContext, LoraTrainingRequest, LogEvent, ProgressEvent
from backend.engine._transformer_registry import _instantiate_image_text_encoder, get_text_encoder
from backend.engine.common.codecs.vae import VAEEncoder, infer_latent_channels, prepare_vae_encoder_weight_items
from backend.engine.config.model_configs import get_config_class
from backend.engine.contracts import local_bundle_root
from backend.engine.families.z_image.weights import remap_zimage_lora_keys
from backend.engine.pipelines.image_model_load import load_image_transformer
from backend.engine.training.crop import prepare_training_rgb_image, resolve_training_resolution
from backend.engine.training.dataset_store import load_training_pairs_unified
from backend.engine.training.flux_dreambooth_mlx import _load_vae_encoder, _log, _progress, _save_adapter
from backend.engine.training.lora_layers import (
    apply_lora_to_zimage_dit,
    add_grad_trees,
    prepare_dit_for_lora_training,
    scale_grad_tree,
)
from backend.engine.training.presets import merge_training_request_config, resolve_preset, train_min_memory_gb
from backend.engine.training.user_lora_registry import register_user_lora

_Z_IMAGE_TRAINABLE_ID = "z-image"
_Z_IMAGE_BLOCKED_IDS = frozenset({"z-image-turbo"})


def _load_zimage_text_encoder(ctx: Any, bundle_root: Path, config: Any) -> Any:
    enc_cls = get_text_encoder("z_image")
    return _instantiate_image_text_encoder(
        ctx,
        enc_cls,
        encoder_type="z_image",
        bundle_root=bundle_root,
        config=config,
        enc_kwargs={
            "max_seq_len": getattr(config, "max_seq_len", 512),
            "hidden_state_layers": getattr(config, "text_encoder_out_layers", None),
            "enable_thinking": getattr(config, "enable_thinking", True),
        },
    )


def _encode_dataset(
    *,
    ctx: Any,
    pairs: list[tuple[Path, str]],
    vae: VAEEncoder,
    text_encoder: Any,
    base_model_id: str,
    train_cfg: dict[str, Any],
    preset: str | None,
    num_augmentations: int,
    exec_ctx: ExecutionContext,
) -> tuple[list[Any], list[Any]]:
    latents: list[Any] = []
    cap_feats: list[Any] = []
    total_samples = len(pairs) * num_augmentations
    _log(exec_ctx, "info", f"Encoding {len(pairs)} images × {num_augmentations} augmentations …")
    _progress(
        exec_ctx,
        step=0,
        total=1,
        message=f"Encoding 0/{total_samples} samples …",
        phase="encoding",
        progress=0.02,
    )
    done = 0
    for img_path, prompt in pairs:
        cap = text_encoder.encode([prompt])
        mx.eval(cap)
        for aug_i in range(num_augmentations):
            arr, _ = prepare_training_rgb_image(
                img_path,
                base_model_id,
                train_cfg,
                preset=preset,
                augmentation_index=aug_i,
            )
            nchw = mx.array(arr.transpose(2, 0, 1)[None].astype("float32"))
            n11 = nchw * 2.0 - 1.0
            z = vae.encode(n11)
            if getattr(z, "ndim", 0) == 5:
                z = z[:, :, 0, :, :]
            latents.append(z.astype(ctx.bfloat16()))
            cap_feats.append(cap)
            done += 1
            if done == total_samples or done % max(1, total_samples // 8) == 0:
                frac = 0.02 + 0.08 * (done / max(total_samples, 1))
                _progress(
                    exec_ctx,
                    step=0,
                    total=1,
                    message=f"Encoding {done}/{total_samples} samples …",
                    phase="encoding",
                    progress=frac,
                )
    return latents, cap_feats


def _training_loss(model: Any, x0: mx.array, cap: mx.array, ctx: Any) -> mx.array:
    b = x0.shape[0]
    t = mx.random.uniform(shape=(b,), dtype=ctx.float32())
    eps = mx.random.normal(x0.shape, dtype=ctx.bfloat16())
    sigma = mx.reshape(t, (b, 1, 1, 1)).astype(ctx.bfloat16())
    x_t = (1.0 - sigma) * x0 + sigma * eps
    x_t = mx.stop_gradient(x_t)
    pred = model(x_t, timestep=0, txt_embeds=cap, sigmas=t)
    return mx.mean(mx.square(pred + x0 - eps))


def _validate_saved_lora(path: Path) -> None:
    from backend.engine.common.bundle.weights import load_safetensors
    from backend.engine.config.model_configs import ZImageConfig
    from backend.engine.families.z_image.transformer import ZImageTransformer
    from backend.engine.runtime.mlx import MLXContext
    from backend.engine.training.lora_layers import (
        enumerate_zimage_lora_module_paths,
        repair_indexed_lora_weights,
    )

    flat = load_safetensors(str(path))
    weights = dict(flat)
    if any(key.startswith("lora_") and ".lora_A." in key for key in weights):
        config_path = path.parent / "lora_config.json"
        lora_blocks = -1
        if config_path.is_file():
            lora_blocks = int(json.loads(config_path.read_text()).get("lora_blocks") or -1)
        probe = ZImageTransformer(ZImageConfig(), MLXContext())
        paths = enumerate_zimage_lora_module_paths(probe, lora_blocks=lora_blocks)
        weights = repair_indexed_lora_weights(weights, module_paths=paths)
    remapped = remap_zimage_lora_keys(weights)
    if not remapped:
        raise RuntimeError(
            f"Saved LoRA {path} has no remappable (lora_A, lora_B) pairs for Z-Image"
        )
    infer = ZImageTransformer(ZImageConfig(), MLXContext())
    matched = sum(1 for tgt in remapped if f"{tgt}.weight" in infer._param_map)
    if matched == 0:
        raise RuntimeError(
            f"Saved LoRA {path}: remapped {len(remapped)} groups but none match Z-Image DiT weights "
            "(re-export with a current Studio build or retrain)."
        )
    if matched < len(remapped):
        raise RuntimeError(
            f"Saved LoRA {path}: only {matched}/{len(remapped)} groups match Z-Image DiT weights"
        )


def _generate_progress_image(
    *,
    model: Any,
    vae_dec: Any,
    text_encoder: Any,
    prompt: str,
    resolution: tuple[int, int],
    ctx: Any,
    steps: int = 20,
) -> np.ndarray:
    from backend.engine.common.ops.schedulers import FlowMatchEulerScheduler

    w, h = resolution
    lh, lw = h // 8, w // 8
    latents = mx.random.normal((1, 16, lh, lw), dtype=ctx.bfloat16())
    cap = text_encoder.encode([prompt])
    mx.eval(latents, cap)
    image_seq_len = lh * lw
    sched = FlowMatchEulerScheduler(num_train_timesteps=1000, shift=1.0, ctx=ctx)
    sched.set_timesteps(steps, mu=sched._compute_empirical_mu(image_seq_len, steps))
    for i, t in enumerate(sched._timesteps):
        t_val = float(np.asarray(t).reshape(-1)[0]) if hasattr(t, "shape") else float(t)
        sigmas = mx.array([t_val], dtype=ctx.float32())
        pred = model(latents, timestep=i, txt_embeds=cap, sigmas=sigmas)
        latents = sched.step(pred, t, latents)
        mx.eval(latents)
    img = vae_dec.forward(latents)
    mx.eval(img)
    arr = np.asarray(img[0].transpose(1, 2, 0), dtype=np.float32)
    arr = np.clip((arr + 1.0) * 0.5, 0, 1)
    return (arr * 255).astype(np.uint8)


def run_z_image_dreambooth_training(
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

    base_model_id, version_key = (
        request.base_model.split(":", 1) if ":" in request.base_model else (request.base_model, "")
    )
    if base_model_id in _Z_IMAGE_BLOCKED_IDS:
        raise RuntimeError(
            "LoRA training supports Z-Image Base (z-image) only; "
            "z-image-turbo is distilled and not trainable."
        )

    mem_gb = get_memory_gb()
    min_mem = train_min_memory_gb(base_model_id)
    if mem_gb > 0 and mem_gb < min_mem - 2:
        raise RuntimeError(
            f"Z-Image LoRA training requires ~{min_mem:.0f}GB unified memory "
            f"(detected {mem_gb:.0f}GB). Reduce resolution/lora_blocks or wait for QLoRA support."
        )

    entry = registry.require(base_model_id)
    if str(getattr(entry, "family", "")) != "z_image":
        raise RuntimeError(
            f"Z-Image training runner expects family z_image (model {base_model_id!r} is {entry.family!r})"
        )
    if base_model_id != _Z_IMAGE_TRAINABLE_ID:
        raise RuntimeError(
            f"LoRA training supports Z-Image Base ({_Z_IMAGE_TRAINABLE_ID!r}) only "
            f"(got {base_model_id!r})"
        )

    preset = resolve_preset(request.preset, base_model=request.base_model)
    cfg = merge_training_request_config(request, preset)
    iterations = int(cfg.get("iterations", 800))
    lora_rank = int(cfg.get("lora_rank", 16))
    lora_blocks = int(cfg.get("lora_blocks") if cfg.get("lora_blocks") is not None else -1)
    learning_rate = float(cfg.get("learning_rate") or 1e-4)
    grad_accumulate = int(cfg.get("grad_accumulate") or 4)
    warmup_steps = int(cfg.get("warmup_steps") or 100)
    resolution = resolve_training_resolution(base_model_id, cfg, preset=request.preset)
    num_augmentations = int(cfg.get("num_augmentations") or 5)
    progress_prompt = (request.progress_prompt or "").strip()
    if not progress_prompt:
        raise RuntimeError("progress_prompt is required for LoRA training")
    progress_every = int(cfg.get("progress_every") or 400)
    progress_steps = int(cfg.get("progress_steps") or 20)
    checkpoint_every = int(cfg.get("checkpoint_every") or 400)

    ctx = runtime
    bundle_root = local_bundle_root(project_root, entry, version_key or None)
    if bundle_root is None or not bundle_root.is_dir():
        raise RuntimeError(f"Base model {base_model_id!r} is not installed")

    work_dir = Path(exec_ctx.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = work_dir / "adapters"
    adapter_dir.mkdir(exist_ok=True)

    pairs, training_caption = load_training_pairs_unified(
        project_root,
        request.dataset_id,
        progress_prompt=progress_prompt,
    )
    if len(pairs) < 3:
        raise RuntimeError("Dataset must contain at least 3 images")
    _log(exec_ctx, "info", f"DreamBooth caption: {training_caption!r}")

    config = get_config_class("z_image")()
    _log(exec_ctx, "info", "Loading VAE encoder and Z-Image text encoder …")
    _progress(
        exec_ctx,
        step=0,
        total=1,
        message="Loading VAE and text encoder …",
        phase="loading_model",
        progress=0.01,
    )
    vae_enc = _load_vae_encoder(ctx, bundle_root)
    text_encoder = _load_zimage_text_encoder(ctx, bundle_root, config)

    _log(
        exec_ctx,
        "info",
        f"Training crop {resolution[0]}×{resolution[1]} (portrait-biased cover, Z-Image VAE grid ÷8) …",
    )
    latents, cap_feats = _encode_dataset(
        ctx=ctx,
        pairs=pairs,
        vae=vae_enc,
        text_encoder=text_encoder,
        base_model_id=base_model_id,
        train_cfg=cfg,
        preset=request.preset,
        num_augmentations=num_augmentations,
        exec_ctx=exec_ctx,
    )
    del vae_enc
    text_encoder.release_weights()
    ctx.clear_cache()

    _log(exec_ctx, "info", "Loading Z-Image DiT …")
    _progress(
        exec_ctx,
        step=0,
        total=1,
        message="Loading Z-Image DiT …",
        phase="loading_model",
        progress=0.10,
    )
    model = load_image_transformer(
        ctx=ctx,
        family="z_image",
        config=config,
        entry=entry,
        version_key=version_key or None,
        project_root=project_root,
        model_cache=None,
        allow_cache=False,
    )
    if model is None:
        raise RuntimeError("Failed to load Z-Image transformer from bundle")

    model, train_module = prepare_dit_for_lora_training(
        model,
        apply_lora_to_zimage_dit,
        rank=lora_rank,
        lora_blocks=lora_blocks,
    )

    warmup = optim.linear_schedule(0, learning_rate, warmup_steps)
    cosine = optim.cosine_decay(learning_rate, max(1, iterations // grad_accumulate))
    lr_schedule = optim.join_schedules([warmup, cosine], [warmup_steps])
    optimizer = optim.Adam(learning_rate=lr_schedule)

    xs = mx.concatenate(latents)
    if cap_feats:
        mx.eval(xs, *cap_feats)
    else:
        mx.eval(xs)
    n_samples = len(latents)

    loss_history: list[dict[str, float]] = []
    (work_dir / "loss_history.json").write_text("[]", encoding="utf-8")

    loss_and_grad = nn.value_and_grad(
        train_module,
        lambda x0, cap: _training_loss(model, x0, cap, ctx),
    )

    _log(exec_ctx, "info", f"Training {iterations} iterations (rank={lora_rank}) …")
    _progress(
        exec_ctx,
        step=0,
        total=iterations,
        message=f"Training 0/{iterations} …",
        phase="training",
        progress=0.10,
    )
    accum_grads: dict | None = None
    losses: list[float] = []
    tic = time.time()

    for i in range(iterations):
        exec_ctx.cancel_token.raise_if_cancelled()
        idx = int(mx.random.randint(0, n_samples, (1,)).item())
        x0 = xs[idx : idx + 1]
        cap = cap_feats[idx]
        loss, grads = loss_and_grad(x0, cap)
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

        if (i + 1) == 1 or (i + 1) % 10 == 0:
            avg = sum(losses) / len(losses)
            peak = mx.metal.get_peak_memory() / 1024**3
            _log(
                exec_ctx,
                "info",
                f"Iter {i + 1}/{iterations} loss={avg:.4f} peak_mem={peak:.1f}GB "
                f"it/s={10 / max(time.time() - tic, 1e-6):.2f}",
            )
            loss_history.append({"step": i + 1, "loss": avg})
            (work_dir / "loss_history.json").write_text(
                json.dumps(loss_history), encoding="utf-8"
            )
            losses = []
            tic = time.time()

        _progress(
            exec_ctx,
            step=i + 1,
            total=iterations,
            loss=float(loss.item()),
            progress=0.10 + 0.90 * ((i + 1) / max(iterations, 1)),
        )

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
                te = _load_zimage_text_encoder(ctx, bundle_root, config)
                preview = _generate_progress_image(
                    model=model,
                    vae_dec=dec,
                    text_encoder=te,
                    prompt=progress_prompt,
                    resolution=resolution,
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
    _validate_saved_lora(final_path)

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
        "lora_blocks": lora_blocks,
        "base_model": base_model_id,
        "alpha": lora_rank,
        "trigger_word": "",
        "training_caption": training_caption,
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
        "training_caption": training_caption,
    }
