"""Media codec dispatch — text encoders & VAE (registry-facing entry).

**Not** a parallel ``engine/codecs/`` wrapper tree (governance forbids that).
Reusable implementations live in:

- ``backend/engine/common/codecs/text_encoders/`` — T5, CLIP, Qwen3, …
- ``backend/engine/common/codecs/vae/`` — standard AutoencoderKL encode/decode

Dispatch tables:

- ``vae_codec_registry`` — VAE encode/decode/preview by diffusers ``_class_name``
- ``_transformer_registry`` — text encoders + image/video prompt encoding
"""
from __future__ import annotations

from backend.engine._transformer_registry import (
    augment_image_generation_request,
    attach_image_conditioning,
    attach_image_edit_extra_cond,
    audio_lyrics_metadata,
    encode_image_text_conditioning,
    encode_prompt_with_image_text_encoder,
    encode_video_hunyuan_dual_cfg_batch,
    encode_video_prompt,
    get_text_encoder,
    get_video_text_encoder_class,
)
from backend.engine.vae_codec_registry import (
    decode_vae_preview,
    get_vae_decode_handler,
    get_vae_encode_handler,
    get_vae_preview_decode_handler,
    get_vae_preview_warmup_handler,
    qwen_pack_latents_nchw,
    qwen_unpack_latents_nchw,
    registered_vae_decode_classes,
    registered_vae_encode_classes,
    warmup_vae_preview,
)

__all__ = [
    "augment_image_generation_request",
    "attach_image_conditioning",
    "attach_image_edit_extra_cond",
    "audio_lyrics_metadata",
    "decode_vae_preview",
    "encode_image_text_conditioning",
    "encode_prompt_with_image_text_encoder",
    "encode_video_hunyuan_dual_cfg_batch",
    "encode_video_prompt",
    "get_text_encoder",
    "get_vae_decode_handler",
    "get_vae_encode_handler",
    "get_vae_preview_decode_handler",
    "get_vae_preview_warmup_handler",
    "get_video_text_encoder_class",
    "qwen_pack_latents_nchw",
    "qwen_unpack_latents_nchw",
    "registered_vae_decode_classes",
    "registered_vae_encode_classes",
    "warmup_vae_preview",
]
