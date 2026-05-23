"""
通用组件 — 所有模型和管线共享。
"""
from .schedulers import (
    Scheduler, 
    FlowMatchEulerScheduler,
    LinearScheduler, 
    UniPCScheduler,
    DPMPlusPlusScheduler,
    CogVideoXDPMScheduler,
    SeedVR2EulerScheduler,
    get_scheduler,
)
from .norm import (
    RMSNorm,
    LayerNorm,
    GroupNorm,
    AdaLayerNorm,
    apply_rms_norm,
    apply_layer_norm_fp32,
    apply_scale_shift,
    apply_ada_layer_norm_continuous,
    apply_ada_layer_norm_zero,
    apply_ada_layer_norm_zero_single,
    split_last_dim_chunks,
    unpack_modulation_2way,
    unpack_modulation_3way,
    unpack_modulation_4way,
    unpack_modulation_6way,
    unpack_modulation_2table,
    unpack_modulation_6table,
)
from .activations import silu, gelu
from .attention import (
    SelfAttention,
    CrossAttention,
    TemporalAttention,
    attention_blhd,
    attention_bhsd,
    attention_bhsd_to_blhd,
    scaled_dot_product_attention_bhsd_mx,
    scaled_dot_product_attention_bhsd_torch,
    repeat_kv_heads_mx,
    repeat_kv_heads_torch,
    build_key_padding_mask_from_lengths,
    build_causal_attention_mask,
    build_padding_attention_bias,
    resolve_blhd_attention_mask,
    build_bidirectional_bool_attention_mask,
    left_pad_token_mask,
    apply_binary_mask_bias,
    build_causal_with_padding_bias,
    build_causal_with_offset_bias,
    build_window_with_padding_bias,
    build_window_with_padding_bias_torch,
    build_frame_prefix_causal_bias,
)
from .embeddings import (
    TimestepEmbedding, sinusoidal_embedding_1d, RoPE2D, RoPE3D,
    factorized_rope_params, factorized_rope_concat_params, factorized_rope_precompute_cos_sin, factorized_rope_apply,
    build_position_ids_2d, build_position_ids_3d_axes,
    pad_ragged_2d_sequences, pad_ragged_1d_sequences,
    pad_len_to_multiple, build_tail_pad_mask, pad_tail_with_last, apply_pad_token,
    apply_complex_rope_bshd, apply_complex_rope_from_cis_bshd,
    PatchEmbed2D, PatchEmbed3D,
)
from .weights import (
    parse_size_gb, load_safetensors, save_safetensors,
    LoRAConfig, load_lora_weights, inject_lora, quantize_weights,
)
from .vae import remap_vae_weights
from .cache import ModelCache
from backend.core.contracts import CancelToken

from .pipeline import DenoisingPipeline, GenerationCancelled
from .text_encoders import T5Encoder, CLIPEncoder

__all__ = [
    # Schedulers
    "Scheduler", "FlowMatchEulerScheduler", "LinearScheduler",
    "UniPCScheduler", "DPMPlusPlusScheduler", "CogVideoXDPMScheduler", "SeedVR2EulerScheduler",
    "get_scheduler",
    # Norm
    "RMSNorm", "LayerNorm", "GroupNorm", "AdaLayerNorm",
    "apply_rms_norm", "apply_layer_norm_fp32", "apply_scale_shift", "apply_ada_layer_norm_continuous",
    "apply_ada_layer_norm_zero", "apply_ada_layer_norm_zero_single",
    "split_last_dim_chunks", "unpack_modulation_2way", "unpack_modulation_3way", "unpack_modulation_4way",
    "unpack_modulation_6way", "unpack_modulation_2table", "unpack_modulation_6table",
    # Activations
    "silu", "gelu",
    # Attention
    "SelfAttention", "CrossAttention", "TemporalAttention",
    "attention_blhd",
    "attention_bhsd",
    "attention_bhsd_to_blhd",
    "scaled_dot_product_attention_bhsd_mx",
    "scaled_dot_product_attention_bhsd_torch",
    "repeat_kv_heads_mx",
    "repeat_kv_heads_torch",
    "build_key_padding_mask_from_lengths", "build_causal_attention_mask",
    "build_padding_attention_bias", "resolve_blhd_attention_mask", "build_bidirectional_bool_attention_mask",
    "left_pad_token_mask", "apply_binary_mask_bias",
    "build_causal_with_padding_bias",
    "build_causal_with_offset_bias",
    "build_window_with_padding_bias",
    "build_window_with_padding_bias_torch",
    "build_frame_prefix_causal_bias",
    # Embeddings
    "TimestepEmbedding", "sinusoidal_embedding_1d", "RoPE2D", "RoPE3D",
    "factorized_rope_params", "factorized_rope_concat_params", "factorized_rope_precompute_cos_sin", "factorized_rope_apply",
    "build_position_ids_2d", "build_position_ids_3d_axes",
    "pad_ragged_2d_sequences", "pad_ragged_1d_sequences",
    "pad_len_to_multiple", "build_tail_pad_mask", "pad_tail_with_last", "apply_pad_token",
    "apply_complex_rope_bshd", "apply_complex_rope_from_cis_bshd",
    "PatchEmbed2D", "PatchEmbed3D",
    # Weights
    "parse_size_gb", "load_safetensors", "save_safetensors",
    "LoRAConfig", "load_lora_weights", "inject_lora", "quantize_weights",
    # Cache
    "ModelCache",
    # Pipeline
    "DenoisingPipeline", "CancelToken", "GenerationCancelled",
    # Text Encoders (family-specific encoders live under ``engine.families.*``)
    "T5Encoder", "CLIPEncoder",
]
