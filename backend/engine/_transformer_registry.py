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
}

_WEIGHT_REMAP = {
    "z_image":  ("backend.engine.families.z_image.weights",  "remap_zimage_weights"),
    "flux2":    ("backend.engine.families.flux2.weights",     "remap_flux2_weights"),
    "flux1":    ("backend.engine.families.flux1.weights",    "remap_flux1_weights"),
    "qwen_image": ("backend.engine.families.qwen.weights",    "remap_qwen_transformer_weights"),
}

# encoder_type → (模块路径, 类名)
_TEXT_ENCODER = {
    "flux1":    ("backend.engine.families.flux1.text_encoder",     "Flux1TextEncoder"),
    "flux2":    ("backend.engine.families.flux2.text_encoder",     "Flux2TextEncoder"),
    "z_image":  ("backend.engine.families.z_image.text_encoder",   "ZImageTextEncoder"),
    "qwen_image": ("backend.engine.families.qwen.text_encoder", "QwenImageTextEncoder"),
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
    "cogvideox": ("backend.engine.families.cogvideox.transformer", "CogVideoXTransformer"),
}

_VIDEO_WEIGHT_REMAP = {
    "wan":       ("backend.engine.families.wan.weights",       "remap_wan_weights"),
    "ltx":       ("backend.engine.families.ltx.weights",       "remap_ltx_weights"),
    "cogvideox": ("backend.engine.families.cogvideox.weights", "remap_cogvideox_weights"),
}


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


# =========================================================================
# 音频模型注册表
# =========================================================================

_AUDIO_TRANSFORMER = {
    "ace_step": ("backend.engine.families.ace_step.transformer", "AceStepTransformer"),
}

_AUDIO_WEIGHT_REMAP = {
    "ace_step": ("backend.engine.families.ace_step.weights", "remap_ace_step_weights"),
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
