"""
Model configuration dataclasses — architecture hyperparameters per model family.

Reference implementations of ModelConfig and config system.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


# =========================================================================
# Image models
# =========================================================================


@dataclass
class Flux1Config:
    """Flux.1 series (schnell / dev / fill / depth / kontext / redux / controlnet)
    
    Architecture: MM-DiT (dual-stream / joint attention) + T5 + CLIP
    """
    in_channels: int = 16            # VAE latent channels（DiT 内 2×2 pack 后为 64 维 token）
    out_channels: int = 64           # ``proj_out`` 每 token 维度，unpack 回 16ch 空间格
    hidden_dim: int = 3072           # hidden dimension
    num_heads: int = 24
    num_joint_layers: int = 19       # diffusers ``num_layers`` (FluxTransformerBlock)
    num_single_layers: int = 38      # diffusers ``num_single_layers`` (FluxSingleTransformerBlock)
    text_dim: int = 4096             # T5 output dim
    clip_dim: int = 0                # diffusers DiT 无 CLIP token 支路（pooled 见 pooled_dim）
    pooled_dim: int = 768            # CLIP pooled → time_text_embed.text_embedder
    encoder_type: str = "flux1"      # T5 + CLIP pooled（见 families/flux1/text_encoder.py）
    latent_noise_packed: bool = True  # 初始噪声在 packed [B, (H//16)*(W//16), 64] 上采样（对齐 mflux）
    max_seq_len: int = 512           # max text token count
    rope_dim: int = 64
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    # Variant flags
    supports_guidance: bool = False   # schnell=False; dev/krea 由 registry 置 True
    supports_img2img: bool = True
    supports_mask: bool = False       # Fill / Depth need this
    supports_controlnet: bool = False
    vae_scale: int = 8               # VAE latent downsampling factor

    def __post_init__(self):
        pass


@dataclass
class Flux2Config:
    """Flux.2 Klein series (4B / 9B / base)
    
    Architecture: MM-DiT dual-stream + AdaLayerNormContinuous + Qwen3 text encoder
    """
    in_channels: int = 128           # Flux2 uses 128 (not 64 like Flux1)
    out_channels: int = 128
    inner_dim: int = 4096            # 9B: 4096; 4B: 3072
    num_heads: int = 32              # 9B: 32; 4B: 24
    attn_head_dim: int = 128
    num_layers: int = 8              # joint blocks (9B: 8)
    num_single_layers: int = 24      # single blocks (9B: 24)
    joint_attention_dim: int = 12288 # Qwen3 output dim
    max_seq_len: int = 512
    mlp_ratio: float = 3.0
    qk_norm: bool = True
    supports_guidance: bool = True
    supports_img2img: bool = True
    supports_edit: bool = False
    encoder_type: str = "flux2"
    text_encoder_out_layers: tuple = (9, 18, 27)  # Flux2 Qwen3 takes 3 layers concatenated
    enable_thinking: bool = False     # Reference: Flux2KleinWeightDefinition explicitly disables
    vae_scale: int = 16              # Flux2 uses 16x tile, not 8x


@dataclass
class QwenImageConfig:
    """Qwen-Image series (txt2img / edit)
    
    Architecture: DiT + Qwen VL vision encoder.
    Transformer ``proj_out`` uses ``patch_size**2 * out_channels``; packed latent token dim is 64 (= 16 VAE latent ch × 4 patch),
    so ``out_channels`` must be **16** (matches diffusers ``QwenImageTransformer2DModel`` default), not 64.
    """
    in_channels: int = 64
    out_channels: int = 16
    hidden_dim: int = 3072
    num_heads: int = 24
    num_layers: int = 60
    text_dim: int = 3584             # Qwen text dim
    vl_dim: int = 3584               # Qwen VL encoder dim
    max_seq_len: int = 2048
    rope_dim: int = 64
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    supports_guidance: bool = True
    supports_img2img: bool = True
    vae_scale: int = 16
    encoder_type: str = "qwen_image"


@dataclass
class FIBOConfig:
    """FIBO series (FIBO / FIBO-Lite / FIBO-Edit)
    
    Architecture: DiT + JSON structured prompt encoding
    """
    in_channels: int = 64
    out_channels: int = 64
    hidden_dim: int = 3072
    num_heads: int = 24
    num_layers: int = 32
    text_dim: int = 4096
    max_seq_len: int = 512
    rope_dim: int = 64
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    supports_guidance: bool = True
    supports_img2img: bool = True
    # FIBO-specific: JSON structured prompt（管线侧按字符串送 T5；与 mflux 一致用合法 JSON 更佳）
    structured_prompt: bool = True
    encoder_type: str = "t5"
    vae_scale: int = 8


@dataclass
class ZImageConfig:
    """Z-Image series (Z-Image / Z-Image-Turbo)
    
    Architecture: ZImageTransformer + Qwen3 text encoder (cap_feat_dim=2560).
    Reference: models/z_image/model/z_image_transformer/transformer.py
    """
    in_channels: int = 16             # VAE latent channels (16, not 64 like Flux)
    out_channels: int = 16
    dim: int = 3840                   # hidden dimension (much larger than Flux1's 3072)
    hidden_dim: int = 3840            # alias for compat
    num_heads: int = 30               # n_heads=30
    num_layers: int = 30              # n_layers=30
    num_refiner_layers: int = 2       # noise_refiner + context_refiner layer count
    text_dim: int = 2560              # Qwen3 cap_feat_dim (not T5)
    cap_feat_dim: int = 2560          # caption feature dimension
    clip_dim: int = 0                 # No CLIP encoder
    max_seq_len: int = 512
    rope_dim: int = 32                # one of axis_dims (32+48+48 3D RoPE)
    rope_theta: float = 256.0
    patch_size: int = 2               # Z-Image uses 2x2 patch (Flux1 uses 1x1)
    t_scale: float = 1000.0           # timestep scaling
    norm_eps: float = 1e-5
    qk_norm: bool = True
    supports_guidance: bool = True    # Z-Image=True, Z-Image-Turbo=False
    supports_img2img: bool = False
    encoder_type: str = "z_image"       # ZImageTextEncoder
    text_encoder_out_layers: Optional[tuple] = None  # flux2=(9,18,27), z_image=None
    enable_thinking: bool = True       # z_image uses True, flux2 uses False
    vae_scale: int = 8
    # Match mflux ``ZImageLatentCreator.create_noise`` (``ModelConfig.precision`` = bf16).
    latent_noise_dtype: str = "bfloat16"


@dataclass
class SeedVR2Config:
    """SeedVR2 super-resolution model (3B / 7B)
    
    Architecture: unconditional DiT + low-res image condition injection
    """
    in_channels: int = 64           # low-res latent
    out_channels: int = 64          # high-res latent
    hidden_dim: int = 1024          # 3B; 7B = 1536
    num_heads: int = 16             # 3B; 7B = 24
    num_layers: int = 24            # 3B; 7B = 32
    text_dim: int = 0               # unconditional, no text encoder
    rope_dim: int = 64
    mlp_ratio: float = 4.0
    qk_norm: bool = False
    supports_guidance: bool = False  # unconditional model
    supports_img2img: bool = False
    # Super-resolution specific
    vae_scale: int = 8
    scale_factor: int = 2            # upscale factor
    tile_size: int = 1024            # tiled super-resolution
    denoise_strength: float = 0.3    # default denoising strength


# =========================================================================
# Audio models
# =========================================================================


@dataclass
class AceStepConfig:
    """ACE-Step series — music generation via DiT + VAE.

    Architecture: Qwen3-based decoder-only DiT (sliding-window self-attn +
    cross-attn) + AdaLN modulation + Oobleck VAE.

    Defaults target XL SFT (4B); bundle config.json is authoritative at load time.
    """
    # DiT decoder (XL SFT defaults; 2B base uses 2048 / 24 / 16)
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    in_channels: int = 192                  # DiT input channels
    audio_acoustic_hidden_dim: int = 64     # output channels (= VAE latent dim)
    patch_size: int = 2
    sliding_window: int = 128
    layer_types: tuple | None = None        # e.g. ("sliding_attention", "full_attention", ...)
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 32768
    is_turbo: bool = False                  # turbo: 8-step, no CFG
    turbo_infer_steps: int = 8
    turbo_shift: float = 1.0

    # VAE
    vae_encoder_hidden_size: int = 128
    vae_downsampling_ratios: tuple = (2, 4, 4, 8, 8)
    vae_channel_multiples: tuple = (1, 2, 4, 8, 16)
    vae_decoder_channels: int = 128
    vae_decoder_input_channels: int = 64     # latent dim into VAE decoder
    audio_channels: int = 2                  # stereo output

    # Text encoder
    encoder_type: str = "qwen3_embedding"    # Qwen3-Embedding-0.6B

    # Pipeline
    supports_guidance: bool = True           # CFG / APG
    supports_img2img: bool = False
    default_infer_steps: int = 50            # base/sft; turbo = 8
    default_shift: float = 3.0

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = tuple(
                "sliding_attention" if bool((i + 1) % 2) else "full_attention"
                for i in range(self.num_hidden_layers)
            )


# =========================================================================
# Video models
# =========================================================================


@dataclass
class LTXConfig:
    """LTX Video series (LTX-2 / LTX-2.3)
    
    Architecture: single-stream spatiotemporal DiT + T5 text encoder.
    Matches diffusers LTXVideoTransformer3DModel (hidden=2048, heads=32, layers=28).
    Pipeline: distilled (fixed sigma) / dev (dynamic CFG) / dev_two_stage.
    """
    dim: int = 2048                  # hidden dimension (32 heads × 64 head_dim)
    depth: int = 28                  # Transformer layer count
    num_heads: int = 32
    head_dim: int = 64               # dim // num_heads
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    # Input dimensions
    dim_in: int = 128                # VAE latent channels (3D)
    dim_out: int = 128
    text_dim: int = 4096             # T5
    max_seq_len: int = 512
    time_dim: int = 256
    # Spatiotemporal parameters
    patch_size: int = 1
    temporal_patch_size: int = 1
    # Pipeline
    supports_guidance: bool = True   # dev mode; distilled=False
    supports_img2img: bool = True
    # Latent grid vs pixels — matches diffusers ``LTXPipeline`` defaults
    # (``vae.spatial_compression_ratio`` / ``vae.temporal_compression_ratio``).
    vae_scale: int = 32
    temporal_vae_scale: int = 8
    default_scheduler: str = "flow_match_euler"


@dataclass
class WanConfig:
    """Wan Video series (Wan2.1 / Wan2.2)
    
    Architecture: dual-model (high/low noise) spatiotemporal DiT + T5
    """
    dim: int = 3584                  # 14B; 1.3B = 1536
    depth: int = 32                  # 14B; 1.3B = 24  
    num_heads: int = 28              # 14B; 1.3B = 12
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    # Input dimensions
    dim_in: int = 128                # VAE latent channels (3D)
    dim_out: int = 128
    text_dim: int = 4096             # T5
    max_seq_len: int = 512
    # Spatiotemporal parameters
    patch_size: tuple = (1, 2, 2)    # T, H, W patch
    rope_dim: int = 64
    temporal_rope_dim: int = 32
    # Temporal
    temporal_attn_every: int = 2
    # Dual model
    dual_model: bool = True          # high_noise + low_noise
    # Pipeline
    supports_guidance: bool = True
    supports_img2img: bool = True
    supports_lora: bool = True       # supports high/low noise LoRA tag routing
    # Scheduler
    default_scheduler: str = "unipc"


@dataclass
class CogVideoXConfig:
    """CogVideoX-5b / diffusers `CogVideoXTransformer3DModel` defaults (ZhipuAI CogVideoX-5b compatible).

    `sample_*` follow diffusers naming (latent grid defaults before spatial VAE upscale math in pipeline).
    """

    inner_dim: int = 1920                # num_attention_heads * attention_head_dim (30 * 64)
    num_attention_heads: int = 30
    attention_head_dim: int = 64
    in_channels: int = 16
    out_channels: int = 16
    flip_sin_to_cos: bool = True
    freq_shift: float = 0.0
    time_embed_dim: int = 512
    text_dim: int = 4096                 # T5-XXL — alias text_embed_dim conceptually
    num_layers: int = 30
    dropout: float = 0.0
    attention_bias: bool = True
    attention_out_bias: bool = True
    norm_eps: float = 1e-5
    ff_bias: bool = True
    ff_inner_dim: int | None = None      # default 4 * inner_dim in FF if None

    sample_width: int = 90               # latent space width (training default)
    sample_height: int = 60              # latent space height
    sample_frames: int = 49              # pixel-frame default driving positional slot math (diffusers quirk)
    patch_size: int = 2
    patch_size_t: int | None = None
    temporal_compression_ratio: int = 4
    max_text_seq_length: int = 226
    spatial_interpolation_scale: float = 1.875
    temporal_interpolation_scale: float = 1.0
    patch_bias: bool = True

    use_rotary_positional_embeddings: bool = False

    dim_in: int = 16                     # latent channels (Pipeline noise shape)
    supports_guidance: bool = True
    supports_img2img: bool = True
    default_scheduler: str = "dpm++"

    temporal_vae_scale: int = 4          # pixel_frames → latent_frames for VideoPipeline noise shape
    vae_scale: int = 8                   # spatial latent scaling vs pixels (registry may override)


    def __post_init__(self) -> None:
        if self.ff_inner_dim is None:
            object.__setattr__(self, "ff_inner_dim", int(self.inner_dim * 4))


def merge_cogvideox_transformer_config_from_bundle(config: CogVideoXConfig, bundle_root: Path | None) -> None:
    """Override ``CogVideoXConfig`` from ``<bundle>/transformer/config.json`` (diffusers layout).

    Public HF / zai-org CogVideoX-5b checkpoints use 48 heads × 64 = 3072 ``inner_dim`` and 42 layers;
    defaults in this repo match the older 30-layer Zhipu snapshot unless merged here.
    """
    if bundle_root is None:
        return
    cfg_path = bundle_root / "transformer" / "config.json"
    if not cfg_path.is_file():
        return
    try:
        data: dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"CogVideoX: cannot read transformer config {cfg_path}: {e}") from e

    _BOOL_KEYS = (
        "flip_sin_to_cos",
        "attention_bias",
        "use_rotary_positional_embeddings",
    )
    _INT_KEYS = (
        "num_attention_heads",
        "attention_head_dim",
        "in_channels",
        "out_channels",
        "num_layers",
        "max_text_seq_length",
        "patch_size",
        "sample_frames",
        "sample_height",
        "sample_width",
        "temporal_compression_ratio",
    )
    _FLOAT_KEYS = (
        "dropout",
        "norm_eps",
        "spatial_interpolation_scale",
        "temporal_interpolation_scale",
        "freq_shift",
    )
    for k in _BOOL_KEYS:
        if k in data:
            setattr(config, k, bool(data[k]))
    for k in _INT_KEYS:
        if k in data:
            setattr(config, k, int(data[k]))
    for k in _FLOAT_KEYS:
        if k in data:
            setattr(config, k, float(data[k]))
    if "text_embed_dim" in data:
        setattr(config, "text_dim", int(data["text_embed_dim"]))
    if "time_embed_dim" in data:
        setattr(config, "time_embed_dim", int(data["time_embed_dim"]))
    if "patch_bias" in data:
        setattr(config, "patch_bias", bool(data["patch_bias"]))
    if "patch_size_t" in data:
        pt = data["patch_size_t"]
        object.__setattr__(config, "patch_size_t", int(pt) if pt is not None else None)

    na = int(getattr(config, "num_attention_heads", 30))
    hd = int(getattr(config, "attention_head_dim", 64))
    inner = na * hd
    object.__setattr__(config, "inner_dim", inner)
    object.__setattr__(config, "ff_inner_dim", int(inner * 4))
    object.__setattr__(config, "dim_in", int(getattr(config, "in_channels", 16)))


# =========================================================================
# Config registry: family → config class
# =========================================================================

FAMILY_CONFIG_MAP: dict[str, type] = {
    # Image
    "flux1": Flux1Config,
    "flux2": Flux2Config,
    "qwen_image": QwenImageConfig,
    "fibo": FIBOConfig,
    "z_image": ZImageConfig,
    "seedvr2": SeedVR2Config,
    # Audio
    "ace_step": AceStepConfig,
    # Video
    "ltx": LTXConfig,
    "wan": WanConfig,
    "cogvideox": CogVideoXConfig,
}


def get_config_class(family: str) -> type:
    """Get config dataclass by family name."""
    cls = FAMILY_CONFIG_MAP.get(family)
    if cls is None:
        raise KeyError(f"unknown model family: {family}")
    return cls
