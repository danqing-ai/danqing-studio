"""VAE：解码器 / 编码器 / diffusers 键映射。

MLX 分块编解码见 ``mlx_tiling``（避免在 ``import backend.engine.common`` 时加载 ``mlx``）。"""

from .decoder import (
    VAEDecoder,
    apply_flux2_latent_preprocess_if_enabled,
    build_standard_vae_preview_session,
    create_loaded_vae_decoder,
    flux2_preprocess_latents_for_decode,
    flux2_quant_preprocess_gate,
    infer_latent_channels,
    load_vae_weight_dict,
    release_vae_decoder_memory,
    read_vae_dir_config,
    reshape_packed_latents_to_nchw,
    vae_forward_to_pil,
    vae_output_to_uint8_hwc,
)
from .encoder import VAEEncoder
from .weight_remap import prepare_vae_encoder_weight_items, load_vae_decoder_from_weights, remap_vae_weights

__all__ = [
    "VAEDecoder",
    "VAEEncoder",
    "apply_flux2_latent_preprocess_if_enabled",
    "build_standard_vae_preview_session",
    "create_loaded_vae_decoder",
    "flux2_preprocess_latents_for_decode",
    "flux2_quant_preprocess_gate",
    "infer_latent_channels",
    "load_vae_decoder_from_weights",
    "load_vae_weight_dict",
    "release_vae_decoder_memory",
    "prepare_vae_encoder_weight_items",
    "read_vae_dir_config",
    "remap_vae_weights",
    "reshape_packed_latents_to_nchw",
    "vae_forward_to_pil",
    "vae_output_to_uint8_hwc",
]
