"""Transformer / 权重映射 / Text Encoder 注册表 — 新增模型只需在此添加条目。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# (模块路径, 类名)
_TRANSFORMER = {
    "z_image":  ("backend.engine.families.z_image.transformer",  "ZImageTransformer"),
    "flux2":    ("backend.engine.families.flux2.transformer",     "Flux2Transformer"),
    "fibo":     ("backend.engine.families.fibo.transformer",      "FIBOTransformer"),
    "flux1":    ("backend.engine.families.flux1.transformer",     "Flux1Transformer"),
    "qwen_image": ("backend.engine.families.qwen.transformer",    "QwenImageTransformer"),
    "longcat":  ("backend.engine.families.longcat.transformer",   "LongCatTransformer"),
}

_WEIGHT_REMAP = {
    "z_image":  ("backend.engine.families.z_image.weights",  "remap_zimage_weights"),
    "flux2":    ("backend.engine.families.flux2.weights",     "remap_flux2_weights"),
    "flux1":    ("backend.engine.families.flux1.weights",    "remap_flux1_weights"),
    "qwen_image": ("backend.engine.families.qwen.weights",    "remap_qwen_transformer_weights"),
    "longcat":  ("backend.engine.families.longcat.weights",   "remap_longcat_weights"),
}

# encoder_type → (模块路径, 类名)
_TEXT_ENCODER = {
    "flux1":    ("backend.engine.families.flux1.text_encoder",     "Flux1TextEncoder"),
    "flux2":    ("backend.engine.families.flux2.text_encoder",     "Flux2TextEncoder"),
    "z_image":  ("backend.engine.families.z_image.text_encoder",   "ZImageTextEncoder"),
    "qwen_image": ("backend.engine.families.qwen.text_encoder", "QwenImageTextEncoder"),
    "fibo":     ("backend.engine.families.fibo.text_encoder", "FiboTextEncoder"),
    "qwen25vl": ("backend.engine.families.longcat.text_encoder", "LongCatTextEncoder"),
}

_IMAGE_LORA_MERGE = {
    "flux1": ("backend.engine.families.flux1.lora_mlx", "merge_flux1_lora_adapters"),
    "flux2": ("backend.engine.families.flux2.lora_mlx", "merge_flux2_lora_adapters"),
    "z_image": ("backend.engine.families.z_image.lora_mlx", "merge_z_image_lora_adapters"),
    "qwen_image": ("backend.engine.families.qwen.lora_mlx", "merge_qwen_image_lora_adapters"),
}


def get_transformer_class(family: str):
    import importlib
    entry = _TRANSFORMER.get(family)
    if entry is None:
        raise RuntimeError(f"Unknown image model family: {family}")
    return getattr(importlib.import_module(entry[0]), entry[1])


def get_weight_remap(family: str):
    import importlib
    entry = _WEIGHT_REMAP.get(family)
    if entry is None:
        return None
    return getattr(importlib.import_module(entry[0]), entry[1])


def get_text_encoder(encoder_type: str):
    import importlib
    entry = _TEXT_ENCODER.get(encoder_type)
    if entry is None:
        raise RuntimeError(f"Unknown encoder type: {encoder_type}")
    return getattr(importlib.import_module(entry[0]), entry[1])


def get_image_lora_merge(family: str):
    import importlib

    entry = _IMAGE_LORA_MERGE.get(family)
    if entry is None:
        return None
    return getattr(importlib.import_module(entry[0]), entry[1])


def merge_image_lora_adapters(
    *,
    family: str,
    model: Any,
    adapters: list[Any],
    base_model_id: str,
    project_root: Path,
    registry: Any,
    ctx: Any,
    on_log: Any | None = None,
) -> None:
    merge_fn = get_image_lora_merge(family)
    if merge_fn is None:
        supported = ", ".join(sorted(_IMAGE_LORA_MERGE.keys()))
        raise RuntimeError(
            "LoRA adapters require in-engine merging; supported image families are "
            f"{supported} (this model is family={family!r}). Remove adapters or switch model."
        )
    kwargs = dict(
        model=model,
        adapters=adapters,
        base_model_id=base_model_id,
        project_root=project_root,
        registry=registry,
        ctx=ctx,
        on_log=on_log,
    )
    if family == "z_image":
        kwargs["patch_size"] = int(getattr(getattr(model, "config", None), "patch_size", 2) or 2)
    merge_fn(**kwargs)


def encode_prompt_with_image_text_encoder(
    ctx: Any,
    text: str,
    *,
    encoder_type: str,
    bundle_root: Path,
    config: Any,
) -> tuple[Any, Any | None, Any | None]:
    """Instantiate registry text encoder for ``encoder_type`` and encode one prompt.

    Centralizes bundle path resolution and ``config`` → encoder kwargs mapping so
    ``ImagePipeline`` does not branch on ``encoder_type`` for construction.

    Returns:
        ``(embeddings, attention_mask, pooled_embeds)``.
        * mask is set for ``qwen_image``;
        * ``pooled_embeds`` for ``flux1`` (CLIP pooled, not a mask).
    """
    enc_cls = get_text_encoder(encoder_type)
    enc_kwargs: dict[str, Any] = {}
    out_layers = getattr(config, "text_encoder_out_layers", None)
    if out_layers is not None:
        enc_kwargs["hidden_state_layers"] = tuple(out_layers)
    enc_kwargs["enable_thinking"] = getattr(config, "enable_thinking", False)

    if encoder_type == "flux1":
        enc = enc_cls(
            ctx,
            bundle_root,
            max_seq_len=getattr(config, "max_seq_len", 512),
            text_dim=getattr(config, "text_dim", 4096),
            pooled_dim=getattr(config, "pooled_dim", 768),
        )
    else:
        enc_dir = bundle_root / "text_encoder"
        tok_dir = bundle_root / "tokenizer"
        if not tok_dir.exists():
            tok_dir = enc_dir
        if not enc_dir.exists():
            enc_dir = bundle_root
            tok_dir = bundle_root
        enc = enc_cls(ctx, str(enc_dir), tokenizer_path=str(tok_dir), **enc_kwargs)

    out = enc.encode([text])
    if isinstance(out, tuple):
        if len(out) == 2 and encoder_type == "flux1":
            return out[0], None, out[1]
        if len(out) == 2 and encoder_type == "fibo":
            return out[0], out[1], None
        if len(out) != 2:
            raise RuntimeError(
                f"Text encoder {encoder_type!r} returned a tuple of len {len(out)}; expected 2."
            )
        return out[0], out[1], None
    return out, None, None


# =========================================================================
# 视频模型注册表
# =========================================================================

_VIDEO_TRANSFORMER = {
    "wan":       ("backend.engine.families.wan.transformer",       "WanTransformer"),
    "ltx":       ("backend.engine.families.ltx.transformer",       "LTXTransformer"),
    "hunyuan":   ("backend.engine.families.hunyuan.transformer",   "HunyuanVideoTransformer"),
}

_VIDEO_WEIGHT_REMAP = {
    "wan":       ("backend.engine.families.wan.weights",       "remap_wan_weights"),
    "ltx":       ("backend.engine.families.ltx.weights",       "remap_ltx_weights"),
    "hunyuan":   ("backend.engine.families.hunyuan.weights",   "remap_hunyuan_weights"),
}

# family → (module, factory_fn) for VideoPipeline Shape C (in-repo family generator)
_VIDEO_GENERATION_FACTORY = {
    "ltx": ("backend.engine.families.ltx.generation", "create_ltx23_generator"),
}

_VIDEO_TEXT_ENCODER = {
    "t5":              ("backend.engine.common.text_encoders", "T5Encoder"),
    "hunyuan_video_dual": ("backend.engine.families.hunyuan.text_encoder", "HunyuanVideoTextEncoder"),
}


def get_video_text_encoder_class(encoder_type: str):
    import importlib
    entry = _VIDEO_TEXT_ENCODER.get(encoder_type)
    if entry is None:
        raise RuntimeError(f"Unknown video text encoder type: {encoder_type}")
    return getattr(importlib.import_module(entry[0]), entry[1])


def encode_video_prompt(
    ctx: Any,
    text: str,
    *,
    encoder_type: str,
    bundle_root: Path,
    config: Any,
) -> tuple[Any, Any | None, Any | None, Any | None, Any | None, Any | None]:
    """Encode one video prompt. Returns ``(e1, mask1, e2, mask2, pooled, extra)``.

    T5: ``(embeds, None, None, None, None, None)``.
    Hunyuan dual: ``(qwen_embeds, qwen_mask, byt5_embeds, byt5_mask, None, None)``.
    """
    enc_cls = get_video_text_encoder_class(encoder_type)
    if encoder_type == "t5":
        from backend.engine.common.bundle_layout import t5_encoder_bundle_paths

        t5_dir, t5_tok_dir = t5_encoder_bundle_paths(bundle_root)
        max_seq = int(getattr(config, "max_text_seq_length", 512))
        enc = enc_cls(ctx, t5_dir, max_seq_len=max_seq, tokenizer_path=t5_tok_dir)
        return enc.encode([text]), None, None, None, None, None

    if encoder_type == "hunyuan_video_dual":
        from backend.engine.families.hunyuan.text_encoder import get_hunyuan_text_encoder

        enc = get_hunyuan_text_encoder(ctx, bundle_root, config)
        e1, m1, e2, m2 = enc.encode([text])
        return e1, m1, e2, m2, None, None

    raise RuntimeError(f"Unsupported video encoder_type: {encoder_type!r}")


def get_video_transformer_class(family: str):
    import importlib
    entry = _VIDEO_TRANSFORMER.get(family)
    if entry is None:
        raise RuntimeError(f"Unknown video model family: {family}")
    return getattr(importlib.import_module(entry[0]), entry[1])


def get_video_weight_remap(family: str):
    import importlib
    entry = _VIDEO_WEIGHT_REMAP.get(family)
    if entry is None:
        return None
    return getattr(importlib.import_module(entry[0]), entry[1])


def get_video_generation_factory(family: str):
    import importlib
    entry = _VIDEO_GENERATION_FACTORY.get(family)
    if entry is None:
        supported = ", ".join(sorted(_VIDEO_GENERATION_FACTORY.keys()))
        raise RuntimeError(
            f"VideoPipeline: no generation factory for family {family!r}; supported: {supported}"
        )
    return getattr(importlib.import_module(entry[0]), entry[1])


# =========================================================================
# 音频模型注册表
# =========================================================================

_AUDIO_TRANSFORMER = {
    "ace_step": ("backend.engine.families.ace_step.transformer", "AceStepTransformer"),
}

_AUDIO_WEIGHT_REMAP = {
    "ace_step": ("backend.engine.families.ace_step.weights", "remap_ace_step_weights"),
}

# family → (module, run_method_name) for MusicPipeline Shape C
_AUDIO_GENERATION_FACTORY = {
    "ace_step": ("backend.engine.families.ace_step.generation", "create_ace_step_generator"),
    "heartmula": ("backend.engine.families.heartmula.generation", "create_heartmula_generator"),
}

# family → (module, bundle_is_ready_fn_name) optional readiness hook
_AUDIO_BUNDLE_READY = {
    "heartmula": ("backend.engine.families.heartmula.bundle", "bundle_is_ready"),
}


def get_audio_transformer_class(family: str):
    import importlib
    entry = _AUDIO_TRANSFORMER.get(family)
    if entry is None:
        raise RuntimeError(f"Unknown audio model family: {family}")
    return getattr(importlib.import_module(entry[0]), entry[1])


def get_audio_weight_remap(family: str):
    import importlib
    entry = _AUDIO_WEIGHT_REMAP.get(family)
    if entry is None:
        return None
    return getattr(importlib.import_module(entry[0]), entry[1])


def get_audio_generation_factory(family: str):
    import importlib
    entry = _AUDIO_GENERATION_FACTORY.get(family)
    if entry is None:
        supported = ", ".join(sorted(_AUDIO_GENERATION_FACTORY.keys()))
        raise RuntimeError(
            f"MusicPipeline: unsupported audio family {family!r}; supported: {supported}"
        )
    return getattr(importlib.import_module(entry[0]), entry[1])


def check_audio_bundle_ready(family: str, bundle_path: Any) -> bool | None:
    """Return ``True``/``False`` when a family hook exists; ``None`` if no extra check."""
    import importlib

    entry = _AUDIO_BUNDLE_READY.get(family)
    if entry is None:
        return None
    fn = getattr(importlib.import_module(entry[0]), entry[1])
    return bool(fn(bundle_path))
