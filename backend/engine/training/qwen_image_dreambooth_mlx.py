"""Qwen-Image DreamBooth LoRA training on MLX."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image

from backend.core.contracts import ExecutionContext, LoraTrainingRequest
from backend.engine._transformer_registry import _instantiate_image_text_encoder, get_text_encoder
from backend.engine.config.model_configs import get_config_class
from backend.engine.contracts import local_bundle_root
from backend.engine.families.qwen.weights import remap_qwen_lora_keys
from backend.engine.pipelines.image_model_load import load_image_transformer
from backend.engine.training.crop import prepare_training_rgb_image, resolve_training_resolution
from backend.engine.training.dataset_store import load_training_pairs_unified
from backend.engine.training.flux_dreambooth_mlx import _log, _progress, _save_adapter
from backend.engine.training.lora_layers import (
    apply_lora_to_qwen_dit,
    enumerate_qwen_lora_module_paths,
    list_qwen_lora_blocks,
    prepare_dit_for_lora_training,
    repair_indexed_lora_weights,
)
from backend.engine.training.lora_train_loop import run_dit_lora_train_loop
from backend.engine.training.lora_train_runtime import (
    assert_training_memory,
    parse_lora_train_runtime_config,
    split_train_val_indices,
)
from backend.engine.training.presets import merge_training_request_config, resolve_preset
from backend.engine.training.user_lora_registry import register_user_lora

_QWEN_IMAGE_TRAINABLE_ID = "qwen-image"


def _load_qwen_text_encoder(
    ctx: Any,
    bundle_root: Path,
    config: Any,
    *,
    entry: Any,
    version_key: str | None,
) -> Any:
    enc_cls = get_text_encoder("qwen_image")
    return _instantiate_image_text_encoder(
        ctx,
        enc_cls,
        encoder_type="qwen_image",
        bundle_root=bundle_root,
        config=config,
        enc_kwargs={},
        registry_entry=entry,
        registry_version_key=version_key,
    )


def _encode_qwen_vae(
    ctx: Any,
    bundle_root: Path,
    project_root: Path,
    image_n11: mx.array,
    *,
    height_px: int,
    width_px: int,
) -> mx.array:
    from backend.engine.families.qwen.vae import QwenVAE, apply_qwen_vae_weights_from_bundle
    from backend.engine.vae_codec_registry import qwen_pack_latents_nchw

    vae = QwenVAE()
    apply_qwen_vae_weights_from_bundle(vae, bundle_root, project_root=project_root)
    enc_out = vae.encode(image_n11)
    if getattr(enc_out, "ndim", 0) == 5 and int(enc_out.shape[2]) == 1:
        enc_out = enc_out[:, :, 0, :, :]
    packed = qwen_pack_latents_nchw(ctx, enc_out, height_px, width_px)
    mx.eval(packed)
    return packed.astype(ctx.bfloat16())


def _encode_dataset(
    *,
    ctx: Any,
    pairs: list[tuple[Path, str]],
    bundle_root: Path,
    project_root: Path,
    text_encoder: Any,
    base_model_id: str,
    train_cfg: dict[str, Any],
    preset: str | None,
    resolution: tuple[int, int],
    num_augmentations: int,
    exec_ctx: ExecutionContext,
) -> tuple[list[Any], list[Any], list[Any]]:
    latents: list[Any] = []
    txt_feats: list[Any] = []
    txt_masks: list[Any] = []
    w, h = resolution[0], resolution[1]
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
        txt, mask = text_encoder.encode([prompt])
        mx.eval(txt, mask)
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
            z = _encode_qwen_vae(
                ctx,
                bundle_root,
                project_root,
                n11,
                height_px=h,
                width_px=w,
            )
            latents.append(z)
            txt_feats.append(txt)
            txt_masks.append(mask)
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
    return latents, txt_feats, txt_masks


def _training_loss(
    model: Any,
    x0: mx.array,
    txt: mx.array,
    mask: mx.array,
    *,
    image_height: int,
    image_width: int,
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
        txt_embeds=txt,
        sigmas=t,
        encoder_hidden_states_mask=mask,
        image_height=image_height,
        image_width=image_width,
    )
    return mx.mean(mx.square(pred + x0 - eps))


def _strip_dit_lora_paths(train_module: Any) -> None:
    paths = getattr(train_module, "_lora_paths", None)
    if not paths:
        return
    train_module._lora_paths = [
        path[4:] if path.startswith("dit.") else path for path in paths
    ]


def _validate_saved_lora(path: Path, *, lora_blocks: int) -> None:
    from backend.engine.common.bundle.weights import load_safetensors
    from backend.engine.config.model_configs import QwenImageConfig
    from backend.engine.families.qwen.transformer import QwenImageTransformer
    from backend.engine.runtime.mlx import MLXContext

    flat = load_safetensors(str(path))
    weights = dict(flat)
    if any(key.startswith("lora_") and ".lora_A." in key for key in weights):
        paths = enumerate_qwen_lora_module_paths(
            QwenImageTransformer(QwenImageConfig(), MLXContext()),
            lora_blocks=lora_blocks,
        )
        weights = repair_indexed_lora_weights(weights, module_paths=paths)
    remapped = remap_qwen_lora_keys(weights)
    if not remapped:
        raise RuntimeError(
            f"Saved LoRA {path} has no remappable (lora_A, lora_B) pairs for Qwen-Image"
        )
    probe = QwenImageTransformer(QwenImageConfig(), MLXContext())
    matched = sum(1 for tgt in remapped if f"dit.{tgt}.weight" in probe._param_map)
    if matched == 0:
        raise RuntimeError(
            f"Saved LoRA {path}: remapped {len(remapped)} groups but none match Qwen-Image DiT weights "
            "(re-export with a current Studio build or retrain)."
        )
    if matched < len(remapped):
        raise RuntimeError(
            f"Saved LoRA {path}: only {matched}/{len(remapped)} groups match Qwen-Image DiT weights"
        )


def _generate_progress_image(
    *,
    model: Any,
    bundle_root: Path,
    project_root: Path,
    text_encoder: Any,
    prompt: str,
    resolution: tuple[int, int],
    ctx: Any,
    steps: int = 20,
) -> np.ndarray:
    from backend.engine.common.ops.schedulers import FlowMatchEulerScheduler
    from backend.engine.families.qwen.vae import QwenVAE, apply_qwen_vae_weights_from_bundle
    from backend.engine.vae_codec_registry import qwen_unpack_latents_nchw

    w, h = resolution
    lh, lw = h // 16, w // 16
    latents = mx.random.normal((1, 64, lh, lw), dtype=ctx.bfloat16())
    txt, mask = text_encoder.encode([prompt])
    mx.eval(latents, txt, mask)
    image_seq_len = lh * lw
    sched = FlowMatchEulerScheduler(num_train_timesteps=1000, shift=1.0, ctx=ctx)
    sched.set_timesteps(steps, mu=sched._compute_empirical_mu(image_seq_len, steps))
    for i, t in enumerate(sched._timesteps):
        t_val = float(np.asarray(t).reshape(-1)[0]) if hasattr(t, "shape") else float(t)
        sigmas = mx.array([t_val], dtype=ctx.float32())
        pred = model(
            latents,
            timestep=i,
            txt_embeds=txt,
            sigmas=sigmas,
            encoder_hidden_states_mask=mask,
            image_height=h,
            image_width=w,
            scheduler_timesteps=sched._timesteps,
        )
        latents = sched.step(pred, t, latents)
        mx.eval(latents)
    z = qwen_unpack_latents_nchw(ctx, latents)
    vae = QwenVAE()
    apply_qwen_vae_weights_from_bundle(vae, bundle_root, project_root=project_root)
    img = vae.decode(z)
    if getattr(img, "ndim", 0) == 5 and int(img.shape[2]) == 1:
        img = img[:, :, 0, :, :]
    mx.eval(img)
    arr = np.asarray(img[0].transpose(1, 2, 0), dtype=np.float32)
    arr = np.clip((arr + 1.0) * 0.5, 0, 1)
    return (arr * 255).astype(np.uint8)


def run_qwen_image_dreambooth_training(
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
    if base_model_id != _QWEN_IMAGE_TRAINABLE_ID:
        raise RuntimeError(
            f"LoRA training supports Qwen-Image ({_QWEN_IMAGE_TRAINABLE_ID!r}) only "
            f"(got {base_model_id!r})"
        )

    entry = registry.require(base_model_id)
    if str(getattr(entry, "family", "")) != "qwen_image":
        raise RuntimeError(
            f"Qwen-Image training runner expects family qwen_image "
            f"(model {base_model_id!r} is {entry.family!r})"
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

    config = get_config_class("qwen_image")()
    _log(exec_ctx, "info", "Loading Qwen-Image text encoder …")
    _progress(
        exec_ctx,
        step=0,
        total=1,
        message="Loading text encoder …",
        phase="loading_model",
        progress=0.01,
    )
    text_encoder = _load_qwen_text_encoder(
        ctx,
        bundle_root,
        config,
        entry=entry,
        version_key=version_key or None,
    )

    _log(
        exec_ctx,
        "info",
        f"Training crop {resolution[0]}×{resolution[1]} (portrait-biased cover, Qwen-Image VAE grid ÷16) …",
    )
    latents, txt_feats, txt_masks = _encode_dataset(
        ctx=ctx,
        pairs=pairs,
        bundle_root=bundle_root,
        project_root=project_root,
        text_encoder=text_encoder,
        base_model_id=base_model_id,
        train_cfg=cfg,
        preset=request.preset,
        resolution=resolution,
        num_augmentations=train_runtime.num_augmentations,
        exec_ctx=exec_ctx,
    )
    text_encoder.release_weights()
    ctx.clear_cache()

    _log(exec_ctx, "info", "Loading Qwen-Image DiT …")
    _progress(
        exec_ctx,
        step=0,
        total=1,
        message="Loading Qwen-Image DiT …",
        phase="loading_model",
        progress=0.10,
    )
    model = load_image_transformer(
        ctx=ctx,
        family="qwen_image",
        config=config,
        entry=entry,
        version_key=version_key or None,
        project_root=project_root,
        model_cache=None,
        allow_cache=False,
    )
    if model is None:
        raise RuntimeError("Failed to load Qwen-Image transformer from bundle")

    model, train_module = prepare_dit_for_lora_training(
        model,
        apply_lora_to_qwen_dit,
        list_lora_blocks_fn=list_qwen_lora_blocks,
        rank=train_runtime.lora_rank,
        lora_blocks=train_runtime.lora_blocks,
        lora_scale=train_runtime.lora_scale,
        lora_dropout=train_runtime.lora_dropout,
        lora_module_keys=train_runtime.lora_module_keys,
        qlora_bits=train_runtime.qlora_bits,
        grad_checkpoint=train_runtime.grad_checkpoint,
    )
    _strip_dit_lora_paths(train_module)

    xs = mx.concatenate(latents)
    if txt_feats:
        mx.eval(xs, *txt_feats, *txt_masks)
    else:
        mx.eval(xs)
    n_samples = len(latents)
    img_h, img_w = resolution[1], resolution[0]

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

    def sample_batch(indices: list[int]) -> tuple[Any, ...]:
        idx = indices[0]
        return (xs[idx : idx + 1], txt_feats[idx], txt_masks[idx])

    def loss_fn(x0: mx.array, txt: mx.array, mask: mx.array) -> mx.array:
        return _training_loss(
            model,
            x0,
            txt,
            mask,
            image_height=img_h,
            image_width=img_w,
            ctx=ctx,
        )

    def preview_at(step: int) -> None:
        te = _load_qwen_text_encoder(
            ctx,
            bundle_root,
            config,
            entry=entry,
            version_key=version_key or None,
        )
        preview = _generate_progress_image(
            model=model,
            bundle_root=bundle_root,
            project_root=project_root,
            text_encoder=te,
            prompt=progress_prompt,
            resolution=resolution,
            ctx=ctx,
            steps=train_runtime.progress_steps,
        )
        Image.fromarray(preview).save(work_dir / f"{step:07d}_progress.png")
        te.release_weights()
        ctx.clear_cache()

    loss_history = run_dit_lora_train_loop(
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
    )

    final_path = adapter_dir / "final_adapters.safetensors"
    meta = {
        "iteration": train_runtime.iterations,
        "lora_rank": train_runtime.lora_rank,
        "base_model": base_model_id,
        "progress_prompt": progress_prompt,
        "qlora_bits": train_runtime.qlora_bits,
    }
    _save_adapter(final_path, train_module, train_runtime.lora_rank, meta)
    _validate_saved_lora(final_path, lora_blocks=train_runtime.lora_blocks)

    output_name = (request.output_name or f"{base_model_id}-{request.dataset_id}").strip()
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in output_name)[:64]
    loras_dir = path_resolver.get_loras_dir()
    dest_dir = loras_dir / slug
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / "adapter.safetensors"
    import shutil

    shutil.copy2(final_path, dest_file)
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
