"""
模型配置 dataclass — 每个模型族的架构超参数。

参考 mflux 项目的 ModelConfig 和 mlx-video 的 config 体系。
"""
from dataclasses import dataclass
from typing import Optional


# =========================================================================
# 图像模型
# =========================================================================


@dataclass
class Flux1Config:
    """Flux.1 系列 (schnell / dev / fill / depth / kontext / redux / controlnet)
    
    架构: MM-DiT (双流 / 联合注意力) + T5 + CLIP
    """
    in_channels: int = 64            # VAE latent channels
    out_channels: int = 64
    hidden_dim: int = 3072           # 隐藏维度
    num_heads: int = 24
    num_single_layers: int = 19      # 单流 block 数
    num_joint_layers: int = 38       # 双流 (MM-DiT) block 数
    text_dim: int = 4096             # T5 输出维度
    clip_dim: int = 768              # CLIP 输出维度
    pooled_dim: int = 768            # CLIP pooled
    max_seq_len: int = 512           # 文本最大 token 数
    rope_dim: int = 64
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    # 变体标识
    supports_guidance: bool = True    # schnell=False, dev=True
    supports_img2img: bool = True
    supports_mask: bool = False       # Fill / Depth 需要
    supports_controlnet: bool = False
    vae_scale: int = 8               # VAE latent 下采样倍数

    def __post_init__(self):
        pass


@dataclass
class Flux2Config:
    """Flux.2 Klein 系列 (4B / 9B / base)
    
    架构: MM-DiT 双流 + AdaLayerNormContinuous + Qwen3 文本编码器
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
    text_encoder_out_layers: tuple = (9, 18, 27)  # Flux2 Qwen3 取 3 层拼接
    enable_thinking: bool = False     # mflux Flux2KleinWeightDefinition 显式禁用
    vae_scale: int = 16              # Flux2 用 16x tile，非 8x


@dataclass
class QwenImageConfig:
    """Qwen-Image 系列 (txt2img / edit)
    
    架构: DiT + Qwen VL 视觉编码器
    """
    in_channels: int = 64
    out_channels: int = 64
    hidden_dim: int = 3584
    num_heads: int = 28
    num_layers: int = 28
    text_dim: int = 3584             # Qwen text dim
    vl_dim: int = 3584               # Qwen VL encoder dim
    max_seq_len: int = 2048
    rope_dim: int = 64
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    supports_guidance: bool = True
    supports_img2img: bool = True
    vae_scale: int = 8




@dataclass
class FIBOConfig:
    """FIBO 系列 (FIBO / FIBO-Lite / FIBO-Edit)
    
    架构: DiT + JSON 结构化 prompt 编码
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
    # FIBO 特有: JSON 结构化 prompt
    structured_prompt: bool = True
    vae_scale: int = 8


@dataclass
class ZImageConfig:
    """Z-Image 系列 (Z-Image / Z-Image-Turbo)
    
    架构: ZImageTransformer + Qwen3 文本编码器 (cap_feat_dim=2560)。
    参考 mflux models/z_image/model/z_image_transformer/transformer.py
    """
    in_channels: int = 16             # VAE latent channels (16, not 64 like Flux)
    out_channels: int = 16
    dim: int = 3840                   # 隐藏维度 (远大于 Flux1 的 3072)
    hidden_dim: int = 3840            # alias for compat
    num_heads: int = 30               # n_heads=30
    num_layers: int = 30              # n_layers=30
    num_refiner_layers: int = 2       # noise_refiner + context_refiner 层数
    text_dim: int = 2560              # Qwen3 cap_feat_dim (不是 T5)
    cap_feat_dim: int = 2560          # caption feature dimension
    clip_dim: int = 0                 # No CLIP encoder
    max_seq_len: int = 512
    rope_dim: int = 32                # axes_dims 之一 (32+48+48 三维 RoPE)
    rope_theta: float = 256.0
    patch_size: int = 2               # Z-Image 用 2x2 patch (Flux1 用 1x1)
    t_scale: float = 1000.0           # timestep scaling
    norm_eps: float = 1e-5
    qk_norm: bool = True
    supports_guidance: bool = True    # Z-Image=True, Z-Image-Turbo=False
    supports_img2img: bool = False
    encoder_type: str = "z_image"       # ZImageTextEncoder
    text_encoder_out_layers: Optional[tuple] = None  # flux2=(9,18,27), z_image=None
    enable_thinking: bool = True       # z_image uses True, flux2 uses False
    vae_scale: int = 8


@dataclass
class SeedVR2Config:
    """SeedVR2 超分模型 (3B / 7B)
    
    架构: 无条件 DiT + 低分辨率图像条件注入
    """
    in_channels: int = 64           # 低分辨率 latent
    out_channels: int = 64          # 高分辨率 latent
    hidden_dim: int = 1024          # 3B; 7B = 1536
    num_heads: int = 16             # 3B; 7B = 24
    num_layers: int = 24            # 3B; 7B = 32
    text_dim: int = 0               # 无条件，无文本编码器
    rope_dim: int = 64
    mlp_ratio: float = 4.0
    qk_norm: bool = False
    supports_guidance: bool = False  # 无条件模型
    supports_img2img: bool = False
    # 超分特有
    vae_scale: int = 8
    scale_factor: int = 2            # 放大倍数
    tile_size: int = 1024            # 分块超分
    denoise_strength: float = 0.3    # 默认去噪强度


# =========================================================================
# 视频模型
# =========================================================================


@dataclass
class LTXConfig:
    """LTX Video 系列 (LTX-2 / LTX-2.3)
    
    架构: 单流时空 DiT + T5 文本编码器
    管线: distilled (固定 sigma) / dev (动态 CFG) / dev_two_stage
    """
    dim: int = 3072                  # 隐藏维度
    depth: int = 28                  # Transformer 层数
    num_heads: int = 24
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    # 输入维度
    dim_in: int = 128                # VAE latent channels (3D)
    dim_out: int = 128
    text_dim: int = 4096             # T5
    max_seq_len: int = 512
    time_dim: int = 256
    # 时空参数
    patch_size: int = 1
    temporal_patch_size: int = 1
    rope_dim: int = 64
    temporal_rope_dim: int = 64
    # 时序
    temporal_attn_every: int = 2     # 每 N 层加时序注意力
    # 音频 (可选)
    audio_dim: int = 0               # 0 = 无音频
    # 管线
    supports_guidance: bool = True   # dev 模式; distilled=False
    supports_img2img: bool = True


@dataclass
class WanConfig:
    """Wan Video 系列 (Wan2.1 / Wan2.2)
    
    架构: 双模型 (高/低噪声) 时空 DiT + T5
    """
    dim: int = 3584                  # 14B; 1.3B = 1536
    depth: int = 32                  # 14B; 1.3B = 24  
    num_heads: int = 28              # 14B; 1.3B = 12
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    # 输入维度
    dim_in: int = 128                # VAE latent channels (3D)
    dim_out: int = 128
    text_dim: int = 4096             # T5
    max_seq_len: int = 512
    # 时空参数
    patch_size: tuple = (1, 2, 2)    # T, H, W patch
    rope_dim: int = 64
    temporal_rope_dim: int = 32
    # 时序
    temporal_attn_every: int = 2
    # 双模型
    dual_model: bool = True          # high_noise + low_noise
    # 管线
    supports_guidance: bool = True
    supports_img2img: bool = True
    supports_lora: bool = True       # 支持 high/low 噪声 LoRA 标签路由
    # 调度器
    default_scheduler: str = "unipc"


@dataclass
class CogVideoXConfig:
    """CogVideoX 系列（将来）
    
    架构: 3D Causal VAE + 时空 Expert Transformer + T5
    """
    dim: int = 3072
    depth: int = 30
    num_heads: int = 24
    num_experts: int = 1             # 1 = dense; >=2 = MoE
    top_k: int = 1
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    dim_in: int = 128
    dim_out: int = 128
    text_dim: int = 4096
    max_seq_len: int = 226
    patch_size: tuple = (1, 2, 2)
    temporal_rope_dim: int = 32
    rope_dim: int = 64
    temporal_attn_every: int = 1
    supports_guidance: bool = True
    supports_img2img: bool = True
    default_scheduler: str = "dpm++"


@dataclass
class LongCatConfig:
    """LongCat-Image — MM-DiT + Qwen2.5-VL。
    
    参考: meituan-longcat/LongCat-Image transformer/config.json
    dim = num_heads * head_dim = 24 * 128 = 3072
    
    in_channels=16: VAE latent channels (transformer 内部做 2x2 patchify 成 64-dim)
    """
    in_channels: int = 16
    out_channels: int = 16
    hidden_dim: int = 3072            # 24*128
    num_heads: int = 24
    attn_head_dim: int = 128
    num_joint_layers: int = 10
    num_single_layers: int = 20
    text_dim: int = 3584              # joint_attention_dim (Qwen2.5-VL)
    pooled_proj_dim: int = 3584       # pooled_projection_dim
    max_seq_len: int = 512
    rope_dim: int = 64
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    supports_guidance: bool = False   # guidance_embeds=false
    supports_img2img: bool = True
    encoder_type: str = "qwen25vl"
    vae_scale: int = 8


# =========================================================================
# 配置注册表：family → config class
# =========================================================================

FAMILY_CONFIG_MAP: dict[str, type] = {
    # 图像
    "flux1": Flux1Config,
    "flux2": Flux2Config,
    "qwen_image": QwenImageConfig,
    "fibo": FIBOConfig,
    "z_image": ZImageConfig,
    "seedvr2": SeedVR2Config,
    "longcat": LongCatConfig,
    # 视频
    "ltx": LTXConfig,
    "wan": WanConfig,
    "cogvideox": CogVideoXConfig,
}


def get_config_class(family: str) -> type:
    """根据 family 名获取配置 dataclass。"""
    cls = FAMILY_CONFIG_MAP.get(family)
    if cls is None:
        raise KeyError(f"unknown model family: {family}")
    return cls
