"""
通用组件 — 所有模型和管线共享。
"""
from .schedulers import (
    Scheduler, 
    FlowMatchEulerScheduler,
    LinearScheduler, 
    UniPCScheduler,
    DPMPlusPlusScheduler,
    SeedVR2EulerScheduler,
    get_scheduler,
)
from .norm import RMSNorm, LayerNorm, GroupNorm, AdaLayerNorm
from .activations import silu, gelu
from .attention import SelfAttention, CrossAttention, TemporalAttention
from .embeddings import (
    TimestepEmbedding, RoPE2D, RoPE3D,
    PatchEmbed2D, PatchEmbed3D,
)
from .weights import (
    parse_size_gb, load_safetensors, save_safetensors,
    LoRAConfig, load_lora_weights, inject_lora, quantize_weights,
)
from .cache import ModelCache
from backend.core.contracts import CancelToken

from .pipeline import DenoisingPipeline, GenerationCancelled
from .text_encoders import T5Encoder, CLIPEncoder, Qwen3TextEncoder

__all__ = [
    # Schedulers
    "Scheduler", "FlowMatchEulerScheduler", "LinearScheduler",
    "UniPCScheduler", "DPMPlusPlusScheduler", "SeedVR2EulerScheduler",
    "get_scheduler",
    # Norm
    "RMSNorm", "LayerNorm", "GroupNorm", "AdaLayerNorm",
    # Activations
    "silu", "gelu",
    # Attention
    "SelfAttention", "CrossAttention", "TemporalAttention",
    # Embeddings
    "TimestepEmbedding", "RoPE2D", "RoPE3D",
    "PatchEmbed2D", "PatchEmbed3D",
    # Weights
    "parse_size_gb", "load_safetensors", "save_safetensors",
    "LoRAConfig", "load_lora_weights", "inject_lora", "quantize_weights",
    # Cache
    "ModelCache",
    # Pipeline
    "DenoisingPipeline", "CancelToken", "GenerationCancelled",
    # Text Encoders
    "T5Encoder", "CLIPEncoder", "Qwen3TextEncoder",
]
