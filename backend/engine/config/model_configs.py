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
    passes_guidance_in_kwargs: bool = True
    preserve_guidance_when_disabled: bool = True
    cfg_negative_eligible: bool = False

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
    requires_sigma_shift: bool = True
    supports_img2img: bool = True
    supports_edit: bool = False
    encoder_type: str = "flux2"
    text_encoder_out_layers: tuple = (9, 18, 27)  # Flux2 Qwen3 takes 3 layers concatenated
    enable_thinking: bool = False     # Reference: Flux2KleinWeightDefinition explicitly disables
    vae_scale: int = 16              # Flux2 uses 16x tile, not 8x
    latent_noise_dtype: str = "bfloat16"
    noise_sample_fp32: bool = True
    vae_preview_warmup: bool = True


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
    encoder_step_kwargs: str = "qwen_image"
    # Qwen-Image-Edit（独立权重）：VL 图文编码 + VAE 参考 latent 序列拼接
    edit_use_vl_vision: bool = False
    edit_conditioning_latent_concat: bool = False


@dataclass
class FIBOConfig:
    """FIBO series (FIBO / FIBO-Lite / FIBO-Edit)

    Architecture: Bria4Transformer2DModel — Joint MM-DiT + Single DiT.
    """
    in_channels: int = 48
    out_channels: int = 48
    hidden_dim: int = 3072
    num_heads: int = 24
    head_dim: int = 128
    num_joint_layers: int = 8
    num_single_layers: int = 38
    text_dim: int = 4096
    text_encoder_dim: int = 2048
    max_seq_len: int = 2048
    rope_dim: int = 64
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    supports_guidance: bool = True
    supports_img2img: bool = True
    structured_prompt: bool = True
    encoder_type: str = "fibo"
    skip_negative_when_structured_prompt: bool = True
    use_mlx_cfg_fusion: bool = True
    vae_scale: int = 16
    text_encoder_mask_key: str = "text_encoder_layers"
    # FIBO-Edit: mflux concat VAE-packed source latents on the sequence axis (not img2img blend).
    edit_conditioning_concat: bool = False
    edit_rmbg_composite_output: bool = False


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
    requires_sigma_shift: bool = True
    supports_img2img: bool = False
    encoder_type: str = "z_image"       # ZImageTextEncoder
    text_encoder_out_layers: Optional[tuple] = None  # flux2=(9,18,27), z_image=None
    enable_thinking: bool = True       # z_image uses True, flux2 uses False
    vae_scale: int = 8
    # Match mflux ``ZImageLatentCreator.create_noise`` (``ModelConfig.precision`` = bf16).
    latent_noise_dtype: str = "bfloat16"
    noise_sample_fp32: bool = True
    z_image_noise_layout: bool = True
    use_mlx_compile: bool = False        # keep numerical path close to reference; prioritize parity
    use_mlx_cfg_fusion: bool = False     # disable fused CFG fast-path; use explicit cond/uncond forwards


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


@dataclass
class LongCatConfig:
    """LongCat-Image — MM-DiT + Qwen2.5-VL.

    Reference: meituan-longcat/LongCat-Image transformer/config.json
    dim = num_heads * head_dim = 24 * 128 = 3072
    in_channels=16: VAE latent channels (2x2 patchify → 64-dim tokens in DiT)
    """
    in_channels: int = 16
    out_channels: int = 16
    hidden_dim: int = 3072
    num_heads: int = 24
    attn_head_dim: int = 128
    num_joint_layers: int = 10
    num_single_layers: int = 20
    text_dim: int = 3584
    pooled_proj_dim: int = 3584
    max_seq_len: int = 512
    rope_dim: int = 64
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    supports_guidance: bool = True
    supports_img2img: bool = True
    encoder_type: str = "qwen25vl"
    vae_scale: int = 8


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


@dataclass
class HeartMulaConfig:
    """HeartMuLa-oss-3B-happy-new-year — autoregressive LM + HeartCodec (12.5 Hz).

    Registry model id: ``heartmula-oss-3b-happy-new-year``. Bundle layout matches
    HeartMuLa/heartlib (HeartMuLaGen root + HeartMuLa-oss-3B/ + HeartCodec-oss/).
    """
    sample_rate: int = 48_000
    frame_rate: float = 12.5
    default_duration_seconds: float = 30.0
    # Product cap (registry ``duration.max``); long MLX runs need large KV cache.
    max_duration_seconds: float = 300.0
    default_cfg_scale: float = 1.5
    cfg_scale_min: float = 1.0
    cfg_scale_max: float = 3.0
    default_temperature: float = 1.0
    temperature_min: float = 0.5
    temperature_max: float = 1.5
    default_topk: int = 50
    topk_min: int = 10
    topk_max: int = 100
    codec_ode_steps: int = 20
    codec_ode_steps_min: int = 4
    codec_ode_steps_max: int = 32
    codec_guidance_scale: float = 1.25
    codec_guidance_min: float = 1.0
    codec_guidance_max: float = 2.0
    long_form_temperature: float = 1.04
    long_form_temperature_min: float = 0.9
    long_form_temperature_max: float = 1.5
    long_form_topk: int = 60
    long_form_topk_min: int = 20
    long_form_topk_max: int = 150
    supports_guidance: bool = True
    supports_img2img: bool = False


# =========================================================================
# Video models
# =========================================================================


@dataclass
class LTXConfig:
    """LTX 2.3 video (Lightricks LTX-2.3, dgrauet MLX bundles).

    Shape C family generator: 48-layer A/V DiT + Gemma 3 + two-stage upscale.
    Registry ``step_distill`` selects distilled vs dev pipelines.
    """
    video_pipeline_shape: str = "family_generator"
    encoder_type: str = "gemma"
    dim: int = 2048
    depth: int = 48
    num_heads: int = 32
    head_dim: int = 64
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    dim_in: int = 128
    dim_out: int = 128
    text_dim: int = 0
    max_seq_len: int = 512
    max_text_seq_length: int = 128
    t5_attention_mask: bool = False
    time_dim: int = 256
    patch_size: int = 1
    temporal_patch_size: int = 1
    supports_guidance: bool = True
    step_distill: bool = False
    supports_img2img: bool = True
    vae_scale: int = 32
    temporal_vae_scale: int = 8
    default_scheduler: str = "flow_match_euler"
    uses_mlx_forge_weight_restore: bool = False
    validate_ltx_block_depth: bool = False
    geometry_check: str = "generic"
    post_denoise_clear_cache: bool = False
    uses_ltx_flat_vae_decoder: bool = False
    video_vae_backend: str = ""
    video_i2v_style: str = "ltx23"
    bundle_config_merger: str = ""
    release_t5_after_encode: bool = False
    cfg_negative_prompt_style: str = "default"
    scheduler_bundle_extras: str = ""
    gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit"
    ltx_low_memory: bool = True
    low_ram_streaming: bool = False
    ltx_stage2_steps: int = 3


@dataclass
class WanConfig:
    """Wan Video series (Wan2.1 / Wan2.2).

    Defaults target **Wan2.2-TI2V-5B**; larger variants override via ``merge_wan_config_from_bundle``.
    """
    dim: int = 3072
    depth: int = 30
    num_heads: int = 24
    ffn_dim: int = 14336
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    cross_attn_norm: bool = True
    eps: float = 1e-6
    freq_dim: int = 256
    # Input dimensions
    dim_in: int = 48
    dim_out: int = 48
    text_dim: int = 4096
    max_seq_len: int = 512
    text_len: int = 512
    # Spatiotemporal parameters
    patch_size: tuple = (1, 2, 2)
    window_size: tuple = (-1, -1)
    rope_dim: int = 64
    temporal_rope_dim: int = 32
    temporal_attn_every: int = 2
    # Dual model (14B high/low noise); TI2V 5B is single-model
    dual_model: bool = False
    expand_timesteps: bool = True  # I2V only: per-token timesteps via wan_expand_timesteps in before_denoise
    # Pipeline / VAE
    vae_scale: int = 16
    temporal_vae_scale: int = 4
    vae_z_dim: int = 48
    supports_guidance: bool = True
    supports_img2img: bool = True
    supports_lora: bool = True
    default_scheduler: str = "wan_flow_unipc"
    num_train_timesteps: int = 1000
    use_mlx_compile: bool = False  # compile partial forward + text KV cache caused denoise drift vs mlx-video
    vae_spatial_tiling: bool = False  # 默认整幅 decode；分块拼接在 TI2V 5B 分辨率下会出 seam
    uses_wan_t5_bundle: bool = True
    uses_wan_shift: bool = True
    snap_pixel_dims: bool = True
    validate_umt5_embeddings: bool = True
    post_denoise_clear_cache: bool = False
    geometry_check: str = "wan"
    uses_wan_vae_bundle: bool = True
    video_vae_backend: str = "wan"
    video_i2v_style: str = "wan"
    bundle_config_merger: str = "wan"
    release_t5_after_encode: bool = True
    cfg_negative_prompt_style: str = "wan"
    scheduler_bundle_extras: str = "wan_flow_unipc"


def merge_wan_config_from_bundle(config: WanConfig, bundle_root: Path | None) -> None:
    """Override ``WanConfig`` from ``config.json`` at bundle root or ``transformer/config.json``."""
    if bundle_root is None:
        return
    candidates = [bundle_root / "config.json", bundle_root / "transformer" / "config.json"]
    cfg_path = next((p for p in candidates if p.is_file()), None)
    if cfg_path is None:
        return
    try:
        data: dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Wan: cannot read config {cfg_path}: {e}") from e

    key_map = {
        "dim": "dim",
        "hidden_size": "dim",
        "num_layers": "depth",
        "num_heads": "num_heads",
        "num_attention_heads": "num_heads",
        "ffn_dim": "ffn_dim",
        "in_dim": "dim_in",
        "in_channels": "dim_in",
        "out_dim": "dim_out",
        "out_channels": "dim_out",
        "text_dim": "text_dim",
        "text_len": "text_len",
        "freq_dim": "freq_dim",
        "qk_norm": "qk_norm",
        "cross_attn_norm": "cross_attn_norm",
        "eps": "eps",
        "dual_model": "dual_model",
        "expand_timesteps": "expand_timesteps",
        "vae_scale": "vae_scale",
        "temporal_vae_scale": "temporal_vae_scale",
    }
    for src, dst in key_map.items():
        if src in data:
            val = data[src]
            if isinstance(val, bool):
                setattr(config, dst, bool(val))
            elif isinstance(val, float):
                setattr(config, dst, float(val))
            else:
                setattr(config, dst, int(val))
    if "patch_size" in data:
        ps = data["patch_size"]
        if isinstance(ps, (list, tuple)) and len(ps) == 3:
            object.__setattr__(config, "patch_size", tuple(int(x) for x in ps))
    if "model_type" in data and str(data["model_type"]).lower() in ("ti2v", "t2v", "i2v"):
        object.__setattr__(config, "model_type", str(data["model_type"]).lower())


def merge_wan_vae_config_from_bundle(config: WanConfig, bundle_root: Path | None) -> None:
    """Apply ``vae/config.json`` (diffusers layout) → ``WanConfig`` pipeline scalars."""
    if bundle_root is None:
        return
    vae_cfg_path = bundle_root / "vae" / "config.json"
    if not vae_cfg_path.is_file():
        return
    try:
        data: dict[str, Any] = json.loads(vae_cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Wan: cannot read VAE config {vae_cfg_path}: {e}") from e

    if "scale_factor_spatial" in data:
        config.vae_scale = int(data["scale_factor_spatial"])
    if "scale_factor_temporal" in data:
        config.temporal_vae_scale = int(data["scale_factor_temporal"])
    z_dim = data.get("z_dim")
    if z_dim is not None:
        z = int(z_dim)
        config.vae_z_dim = z
        config.dim_in = z
        config.dim_out = z


def merge_wan_bundle_config(config: WanConfig, bundle_root: Path | None) -> None:
    """Merge transformer + VAE JSON from a Wan model bundle."""
    merge_wan_config_from_bundle(config, bundle_root)
    merge_wan_vae_config_from_bundle(config, bundle_root)


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
    supports_img2img: bool = False
    default_scheduler: str = "cogvideox_dpm"

    temporal_vae_scale: int = 4          # pixel_frames → latent_frames for VideoPipeline noise shape
    vae_scale: int = 8                   # spatial latent scaling vs pixels (registry may override)
    latent_noise_dtype: str = "bfloat16" # MLX denoise activations (scheduler keeps FP32 updates)
    use_mlx_compile: bool = True         # ``mx.compile`` DiT forward after weight load (MLX only)
    post_denoise_clear_cache: bool = True
    geometry_check: str = "cogvideox"
    uses_prediction_type: bool = True
    video_vae_backend: str = "cogvideox"
    video_i2v_style: str = "concat"
    bundle_config_merger: str = "cogvideox"
    release_t5_after_encode: bool = True
    cfg_negative_prompt_style: str = "default"
    scheduler_bundle_extras: str = "cogvideox_dpm"

    def __post_init__(self) -> None:
        if self.ff_inner_dim is None:
            object.__setattr__(self, "ff_inner_dim", int(self.inner_dim * 4))


@dataclass
class HunyuanVideoConfig:
    """HunyuanVideo-1.5 — diffusers ``HunyuanVideo15Transformer3DModel`` / 480p T2V defaults."""

    inner_dim: int = 2048
    num_attention_heads: int = 16
    attention_head_dim: int = 128
    in_channels: int = 65
    out_channels: int = 32
    dim_in: int = 32
    num_layers: int = 54
    num_refiner_layers: int = 2
    mlp_ratio: float = 4.0
    patch_size: int = 1
    patch_size_t: int = 1
    qk_norm: str = "rms_norm"
    text_embed_dim: int = 3584
    text_embed_2_dim: int = 1472
    image_embed_dim: int = 1152
    rope_theta: float = 256.0
    rope_axes_dim: tuple[int, ...] = (16, 56, 56)
    target_size: int = 640
    task_type: str = "t2v"
    use_meanflow: bool = False

    encoder_type: str = "hunyuan_video_dual"
    text_dim: int = 3584
    mllm_max_length: int = 1000
    byt5_max_length: int = 256
    prompt_template_crop_start: int = 108
    vision_num_semantic_tokens: int = 256

    supports_guidance: bool = True
    supports_img2img: bool = True
    default_scheduler: str = "flow_match_euler"
    temporal_vae_scale: int = 4
    vae_scale: int = 16
    min_unified_memory_gb: int = 32
    step_distill: bool = False
    text_encoder_device: str = "auto"
    text_encoder_qwen_local: str = ""
    text_encoder_byt5_local: str = ""
    text_encoder_release_after_encode: bool = True
    vae_temporal_chunk_size: int = 8
    vae_spatial_tiling: bool = False
    inject_text_encoder_paths: bool = True
    post_denoise_clear_cache: bool = True
    geometry_check: str = "generic"
    uses_hunyuan_vae_bundle: bool = True
    video_vae_backend: str = "hunyuan"
    video_i2v_style: str = "hunyuan"
    bundle_config_merger: str = "hunyuan"
    release_t5_after_encode: bool = False
    cfg_negative_prompt_style: str = "default"
    scheduler_bundle_extras: str = ""
    default_encoder_type: str = "hunyuan_video_dual"


def merge_hunyuan_transformer_config_from_bundle(config: HunyuanVideoConfig, bundle_root: Path | None) -> None:
    """Override ``HunyuanVideoConfig`` from ``<bundle>/transformer/config.json``."""
    if bundle_root is None:
        return
    cfg_path = bundle_root / "transformer" / "config.json"
    if not cfg_path.is_file():
        raise RuntimeError(
            f"HunyuanVideo: missing transformer config {cfg_path}. "
            "Install the full diffusers bundle (transformer/config.json + weight shards)."
        )
    try:
        data: dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"HunyuanVideo: cannot read transformer config {cfg_path}: {e}") from e

    _INT_KEYS = (
        "num_attention_heads", "attention_head_dim", "in_channels", "out_channels",
        "num_layers", "num_refiner_layers", "patch_size", "patch_size_t",
        "text_embed_dim", "text_embed_2_dim", "image_embed_dim", "target_size",
    )
    _FLOAT_KEYS = ("mlp_ratio", "rope_theta")
    _STR_KEYS = ("qk_norm", "task_type")
    _BOOL_KEYS = ("use_meanflow",)
    for k in _INT_KEYS:
        if k in data:
            setattr(config, k, int(data[k]))
    for k in _FLOAT_KEYS:
        if k in data:
            setattr(config, k, float(data[k]))
    for k in _STR_KEYS:
        if k in data:
            setattr(config, k, str(data[k]))
    for k in _BOOL_KEYS:
        if k in data:
            setattr(config, k, bool(data[k]))
    if "rope_axes_dim" in data:
        object.__setattr__(config, "rope_axes_dim", tuple(int(x) for x in data["rope_axes_dim"]))

    na = int(getattr(config, "num_attention_heads", 16))
    hd = int(getattr(config, "attention_head_dim", 128))
    object.__setattr__(config, "inner_dim", na * hd)
    object.__setattr__(config, "dim_in", int(getattr(config, "out_channels", 32)))
    object.__setattr__(config, "text_dim", int(getattr(config, "text_embed_dim", 3584)))


def merge_cogvideox_transformer_config_from_bundle(config: CogVideoXConfig, bundle_root: Path | None) -> None:
    """Override ``CogVideoXConfig`` from ``<bundle>/transformer/config.json`` (diffusers layout).

    Public HF / zai-org CogVideoX-5b checkpoints use 48 heads × 64 = 3072 ``inner_dim`` and 42 layers;
    defaults in this repo match the older 30-layer Zhipu snapshot unless merged here.
    """
    if bundle_root is None:
        return
    cfg_path = bundle_root / "transformer" / "config.json"
    if not cfg_path.is_file():
        raise RuntimeError(
            f"CogVideoX: missing transformer config {cfg_path}. "
            "Install the full diffusers bundle (transformer/config.json + weight shards)."
        )
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


def cogvideox_scheduler_kwargs_from_bundle(bundle_root: Path) -> dict[str, Any]:
    """Build ``CogVideoXDPMScheduler`` ctor kwargs from ``<bundle>/scheduler/scheduler_config.json``.

    Official CogVideoX-5b bundles ship DDIM config (``v_prediction``, ``trailing``, ``snr_shift_scale=1``,
    ``rescale_betas_zero_snr=true``). Diffusers swaps to ``CogVideoXDPMScheduler.from_config(...)`` — we must
    pass the same fields or denoised latents are wrong (patch-grid VAE output).
    """
    cfg_path = bundle_root / "scheduler" / "scheduler_config.json"
    if not cfg_path.is_file():
        raise RuntimeError(
            f"CogVideoX: missing scheduler config {cfg_path}. "
            "Install the full diffusers bundle (scheduler/scheduler_config.json)."
        )
    try:
        data: dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"CogVideoX: cannot read scheduler config {cfg_path}: {e}") from e

    kwargs: dict[str, Any] = {}
    for key in ("num_train_timesteps", "steps_offset"):
        if key in data:
            kwargs[key] = int(data[key])
    for key in ("beta_start", "beta_end", "snr_shift_scale"):
        if key in data:
            kwargs[key] = float(data[key])
    for key in ("beta_schedule", "prediction_type", "timestep_spacing"):
        if key in data:
            kwargs[key] = str(data[key])
    for key in ("set_alpha_to_one", "rescale_betas_zero_snr"):
        if key in data:
            kwargs[key] = bool(data[key])
    if "prediction_type" not in kwargs:
        raise RuntimeError(f"CogVideoX: scheduler config {cfg_path} missing prediction_type")
    return kwargs


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
    "longcat": LongCatConfig,
    # Audio
    "ace_step": AceStepConfig,
    "heartmula": HeartMulaConfig,
    # Video
    "ltx": LTXConfig,
    "wan": WanConfig,
    "cogvideox": CogVideoXConfig,
    "hunyuan": HunyuanVideoConfig,
}

IMAGE_FAMILY_REUSE_CONTRACT = frozenset({"flux1", "flux2", "z_image", "qwen_image", "fibo", "seedvr2", "longcat"})


def get_config_class(family: str) -> type:
    """Get config dataclass by family name."""
    cls = FAMILY_CONFIG_MAP.get(family)
    if cls is None:
        raise KeyError(f"unknown model family: {family}")
    return cls


def assert_image_family_contract(family: str, config: Any) -> None:
    """Engine-side governance for image families entering ``ImagePipeline``."""
    if family not in IMAGE_FAMILY_REUSE_CONTRACT:
        raise RuntimeError(
            f"Image family {family!r} is not declared in IMAGE_FAMILY_REUSE_CONTRACT. "
            "Register family/config/transformer instead of adding pipeline special-case branches."
        )
    encoder_type = str(getattr(config, "encoder_type", "") or "")
    if not encoder_type:
        raise RuntimeError(
            f"Image family {family!r} config must declare encoder_type for registry-driven text encoding."
        )
