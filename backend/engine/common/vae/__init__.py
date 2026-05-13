"""VAE：解码器 / 编码器 / diffusers 键映射。

MLX 分块编解码见 ``mlx_tiling``（避免在 ``import backend.engine.common`` 时加载 ``mlx``）。"""

from .decoder import VAEDecoder, vae_output_to_uint8_hwc
from .encoder import VAEEncoder
from .weight_remap import prepare_vae_encoder_weight_items, remap_vae_weights

__all__ = [
    "VAEDecoder",
    "VAEEncoder",
    "prepare_vae_encoder_weight_items",
    "remap_vae_weights",
    "vae_output_to_uint8_hwc",
]
