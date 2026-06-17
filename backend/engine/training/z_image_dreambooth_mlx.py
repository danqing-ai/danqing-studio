"""Z-Image Base DreamBooth LoRA training on MLX."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
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
    list_zimage_lora_blocks,
    prepare_dit_for_lora_training,
)
from backend.engine.training.dit_training_loss import (
    combine_instance_prior_loss,
    flow_match_mse,
    make_prior_latent,
    sample_noisy_latent,
)
from backend.engine.training.latent_cache import LatentCache
from backend.engine.training.lora_train_loop import run_dit_lora_train_loop
from backend.engine.training.lora_train_runtime import (
    assert_training_memory,
    parse_lora_train_runtime_config,
    split_train_val_indices,
)
from backend.engine.training.presets import merge_training_request_config, resolve_preset
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


def _encode_dataset_to_cache(
    *,
    cache: LatentCache,
    ctx: Any,
    pairs: list[tuple[Path, str]],
    vae: VAEEncoder,
    text_encoder: Any,
    base_model_id: str,
    train_cfg: dict[str, Any],
    preset: str | None,
    num_augmentations: int,
    dataset_id: str,
    resolution: tuple[int, int],
    exec_ctx: ExecutionContext,
    class_prompt: str | None,
) -> int:
    total_samples = len(pairs) * num_augmentations
    cache.begin(
        dataset_id=dataset_id,
        n_pairs=len(pairs),
        num_augmentations=num_augmentations,
        resolution=resolution,
        family="z_image",
        tensor_keys=["latent", "cap"],
    )
    _log(exec_ctx, "info", f"Encoding {len(pairs)} images × {num_augmentations} augmentations …")
    _progress(
        exec_ctx,
        step=0,
        total=1,
        message=f"Encoding 0/{total_samples} samples …",
        phase="encoding",
        progress=0.02,
    )
    sample_idx = 0
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
            z = z.astype(ctx.bfloat16())
            mx.eval(z)
            cache.write_sample(sample_idx, {"latent": z[0], "cap": cap})
            sample_idx += 1
            if sample_idx == total_samples or sample_idx % max(1, total_samples // 8) == 0:
                frac = 0.02 + 0.08 * (sample_idx / max(total_samples, 1))
                _progress(
                    exec_ctx,
                    step=0,
                    total=1,
                    message=f"Encoding {sample_idx}/{total_samples} samples …",
                    phase="encoding",
                    progress=frac,
                )
    if class_prompt:
        prior_cap = text_encoder.encode([class_prompt])
        mx.eval(prior_cap)
        cache.write_prior({"cap": prior_cap})
    return cache.finalize()


def _training_loss(
    model: Any,
    x0: mx.array,
    cap: mx.array,
    ctx: Any,
    *,
    min_snr_gamma: float = 0.0,
    prior_cap: mx.array | None = None,
    prior_loss_weight: float = 0.0,
) -> mx.array:
    x_t, eps, t = sample_noisy_latent(x0, ctx)
    pred = model(x_t, timestep=0, txt_embeds=cap, sigmas=t)
    b = x0.shape[0]
    sigma = mx.reshape(t, (b,) + (1,) * (x0.ndim - 1)).astype(ctx.bfloat16())
    inst = flow_match_mse(pred, x0, eps, sigma=sigma, min_snr_gamma=min_snr_gamma)
    if prior_cap is None or prior_loss_weight <= 0:
        return inst
    x0p = make_prior_latent(x0, ctx)
    x_tp, epsp, tp = sample_noisy_latent(x0p, ctx)
    predp = model(x_tp, timestep=0, txt_embeds=prior_cap, sigmas=tp)
    sigmap = mx.reshape(tp, (b,) + (1,) * (x0.ndim - 1)).astype(ctx.bfloat16())
    prior = flow_match_mse(predp, x0p, epsp, sigma=sigmap, min_snr_gamma=min_snr_gamma)
    return combine_instance_prior_loss(inst, prior, prior_loss_weight=prior_loss_weight)


def _validate_saved_lora(path: Path) -> None:
    from backend.engine.common.bundle.weights import load_safetensors
    from backend.engine.config.model_configs import ZImageConfig
    from backend.engine.families.z_image.transformer import ZImageTransformer
    from backend.engine.runtime.mlx import MLXContext
    from backend.engine.training.lora_layers import (
        enumerate_zimage_lora_module_paths,
        repair_indexed_lora_weights,
    )
    from backend.engine.training.lora_train_export import is_dense_delta_adapter

    flat = load_safetensors(str(path))
    weights = dict(flat)
    if is_dense_delta_adapter(weights):
        return
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
    # Z-Image Base uses static scheduler_shift (registry default 6.0), not Flux empirical μ.
    sched = FlowMatchEulerScheduler(num_train_timesteps=1000, shift=6.0, ctx=ctx)
    sched.set_timesteps(
        steps,
        image_seq_len=image_seq_len,
        scheduler_shift=6.0,
        use_empirical_mu=False,
    )
    sigma_schedule = sched._sigmas
    for i, _t in enumerate(sched._timesteps):
        pred = model(latents, timestep=i, txt_embeds=cap, sigmas=sigma_schedule)
        latents = sched.step(pred, i, latents)
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
    train_runtime = parse_lora_train_runtime_config(cfg, defaults=preset)
    mem_gb = get_memory_gb()
    assert_training_memory(base_model_id, mem_gb, qlora_bits=train_runtime.qlora_bits)

    resolution = resolve_training_resolution(base_model_id, cfg, preset=request.preset)
    progress_prompt = (request.progress_prompt or "").strip()
    if not progress_prompt:
        raise RuntimeError("progress_prompt is required for LoRA training")

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
    latent_cache = LatentCache(work_dir)
    class_prompt = train_runtime.class_prompt
    if train_runtime.prior_loss_weight > 0 and not class_prompt:
        class_prompt = "a photo"

    def _run_encode() -> int:
        return _encode_dataset_to_cache(
            cache=latent_cache,
            ctx=ctx,
            pairs=pairs,
            vae=vae_enc,
            text_encoder=text_encoder,
            base_model_id=base_model_id,
            train_cfg=cfg,
            preset=request.preset,
            num_augmentations=train_runtime.num_augmentations,
            dataset_id=request.dataset_id,
            resolution=resolution,
            exec_ctx=exec_ctx,
            class_prompt=class_prompt if train_runtime.prior_loss_weight > 0 else None,
        )

    if latent_cache.is_valid(
        dataset_id=request.dataset_id,
        n_pairs=len(pairs),
        num_augmentations=train_runtime.num_augmentations,
        resolution=resolution,
        family="z_image",
        n_samples=len(pairs) * train_runtime.num_augmentations,
    ):
        _log(exec_ctx, "info", "Reusing cached latents from work_dir/latent_cache …")
        n_samples = len(pairs) * train_runtime.num_augmentations
    else:
        n_samples = _run_encode()
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
        list_lora_blocks_fn=list_zimage_lora_blocks,
        rank=train_runtime.lora_rank,
        lora_blocks=train_runtime.lora_blocks,
        lora_scale=train_runtime.lora_scale,
        lora_dropout=train_runtime.lora_dropout,
        lora_module_keys=train_runtime.lora_module_keys,
        qlora_bits=train_runtime.qlora_bits,
        grad_checkpoint=train_runtime.grad_checkpoint,
        train_type=train_runtime.train_type,
    )

    prior_cap: Any | None = None
    if train_runtime.prior_loss_weight > 0:
        try:
            prior_cap = latent_cache.load_prior()["cap"]
        except RuntimeError:
            _log(exec_ctx, "warning", "Prior preservation requested but prior cache missing; disabled")
            prior_cap = None

    train_pairs, val_pairs = split_train_val_indices(len(pairs), val_split=train_runtime.val_split)
    train_indices = [
        pi * train_runtime.num_augmentations + aug
        for pi in train_pairs
        for aug in range(train_runtime.num_augmentations)
    ]
    val_indices = [
        pi * train_runtime.num_augmentations + aug
        for pi in val_pairs
        for aug in range(train_runtime.num_augmentations)
    ]

    _log(exec_ctx, "info", f"Loading {n_samples} cached latents into memory …")
    zimage_samples = latent_cache.materialize_z_image(n_samples)
    mx.eval(*[t for sample in zimage_samples for t in sample])

    def sample_batch(indices: list[int]) -> tuple[Any, ...]:
        return zimage_samples[indices[0]]

    def loss_fn(x0: mx.array, cap: mx.array) -> mx.array:
        return _training_loss(
            train_module,
            x0,
            cap,
            ctx,
            min_snr_gamma=train_runtime.min_snr_gamma,
            prior_cap=prior_cap,
            prior_loss_weight=train_runtime.prior_loss_weight if prior_cap is not None else 0.0,
        )

    def preview_at(step: int) -> None:
        from backend.engine.common.codecs.vae import load_vae_weight_dict, read_vae_dir_config
        from backend.engine.common.codecs.vae.decoder import create_loaded_vae_decoder

        vae_dir = bundle_root / "vae"
        vae_cfg, _, _ = read_vae_dir_config(vae_dir)
        vae_weights = load_vae_weight_dict(ctx, vae_dir)
        preview_latent, _ = zimage_samples[0]
        dec, _, _, _ = create_loaded_vae_decoder(
            ctx,
            preview_latent,
            vae_weights,
            float(vae_cfg.get("scaling_factor", 1.0)),
            float(vae_cfg.get("shift_factor", 0.0)),
        )
        te = _load_zimage_text_encoder(ctx, bundle_root, config)
        preview = _generate_progress_image(
            model=train_module,
            vae_dec=dec,
            text_encoder=te,
            prompt=progress_prompt,
            resolution=resolution,
            ctx=ctx,
            steps=train_runtime.progress_steps,
        )
        Image.fromarray(preview).save(work_dir / f"{step:07d}_progress.png")
        te.release_weights()
        ctx.clear_cache()

    loss_history, best_path = run_dit_lora_train_loop(
        exec_ctx=exec_ctx,
        model=model,
        train_module=train_module,
        runtime=train_runtime,
        work_dir=work_dir,
        adapter_dir=adapter_dir,
        base_model_id=base_model_id,
        n_samples=n_samples,
        sample_batch=sample_batch,
        train_indices=train_indices,
        val_indices=val_indices,
        loss_fn=loss_fn,
        on_progress_preview=preview_at,
        mlx_ctx=ctx,
    )

    final_path = adapter_dir / "final_adapters.safetensors"
    export_meta = {
        "iteration": train_runtime.iterations,
        "lora_rank": train_runtime.lora_rank,
        "base_model": base_model_id,
        "progress_prompt": progress_prompt,
        "qlora_bits": train_runtime.qlora_bits,
        "train_type": train_runtime.train_type,
    }
    if best_path is not None and best_path.is_file():
        final_path.write_bytes(best_path.read_bytes())
        best_meta = adapter_dir / "best_adapters.json"
        if best_meta.is_file():
            export_meta = json.loads(best_meta.read_text(encoding="utf-8"))
    else:
        _save_adapter(final_path, train_module, train_runtime.lora_rank, export_meta)

    from backend.engine.training.lora_train_export import export_registered_adapter

    registered_path = export_registered_adapter(
        adapter_dir=adapter_dir,
        train_module=train_module,
        train_runtime=train_runtime,
        base_model_id=base_model_id,
        final_path=final_path,
        meta=export_meta,
        save_adapter=lambda path: _save_adapter(
            path, train_module, train_runtime.lora_rank, export_meta
        ),
    )
    _validate_saved_lora(registered_path)

    output_name = (request.output_name or f"{base_model_id}-{request.dataset_id}").strip()
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in output_name)[:64]
    loras_dir = path_resolver.get_loras_dir()
    dest_dir = loras_dir / slug
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / "adapter.safetensors"
    import shutil

    shutil.copy2(registered_path, dest_file)
    lora_config = {
        "lora_rank": train_runtime.lora_rank,
        "lora_blocks": train_runtime.lora_blocks,
        "base_model": base_model_id,
        "alpha": train_runtime.lora_scale,
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
            lora_rank=train_runtime.lora_rank,
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
