from __future__ import annotations

"""SeedVR2 权重与 bundle：``ModelConfig``、``load_flat_bundle``、键映射与 ``WeightLoader`` 组件定义。

扁平 bundle 键 → MLX 模块树映射见 ``SeedVR2WeightDefinition*`` 引用的 ``SeedVR2WeightMapping``。
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, List

import mlx.core as mx
import mlx.nn as nn

from backend.engine.common.bundle_weight_mapping import WeightMapping, WeightTarget, WeightTransforms
from backend.engine.common.bundle_weights import ComponentDefinition, LoadedWeights, TokenizerDefinition, WeightLoader


class ModelConfig:
    """与上游扁平 safetensors 布局对齐的最小配置。"""

    precision: Any = mx.bfloat16

    def __init__(
        self,
        priority: int,
        aliases: list[str],
        model_name: str,
        base_model: str | None,
        controlnet_model: str | None,
        custom_transformer_model: str | None,
        num_train_steps: int | None,
        max_sequence_length: int | None,
        supports_guidance: bool | None,
        requires_sigma_shift: bool | None,
        transformer_overrides: dict | None = None,
        text_encoder_overrides: dict | None = None,
        sigma_base_shift: float = 0.5,
        sigma_max_shift: float = 1.15,
        sigma_base_seq_len: int = 256,
        sigma_max_seq_len: int = 4096,
        sigma_shift_terminal: float | None = None,
    ):
        self.aliases = aliases
        self.model_name = model_name
        self.base_model = base_model
        self.controlnet_model = controlnet_model
        self.custom_transformer_model = custom_transformer_model
        self.num_train_steps = num_train_steps
        self.max_sequence_length = max_sequence_length
        self.supports_guidance = supports_guidance
        self.requires_sigma_shift = requires_sigma_shift
        self.priority = priority
        self.transformer_overrides = transformer_overrides or {}
        self.text_encoder_overrides = text_encoder_overrides or {}
        self.sigma_base_shift = sigma_base_shift
        self.sigma_max_shift = sigma_max_shift
        self.sigma_base_seq_len = sigma_base_seq_len
        self.sigma_max_seq_len = sigma_max_seq_len
        self.sigma_shift_terminal = sigma_shift_terminal

    @staticmethod
    @lru_cache
    def seedvr2_3b() -> "ModelConfig":
        return AVAILABLE_MODELS["seedvr2-3b"]

    @staticmethod
    @lru_cache
    def seedvr2_7b() -> "ModelConfig":
        return AVAILABLE_MODELS["seedvr2-7b"]


AVAILABLE_MODELS = {
    "seedvr2-3b": ModelConfig(
        priority=22,
        aliases=["seedvr2-3b", "seedvr2"],
        model_name="numz/SeedVR2_comfyUI",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=None,
        max_sequence_length=None,
        supports_guidance=True,
        requires_sigma_shift=None,
    ),
    "seedvr2-7b": ModelConfig(
        priority=23,
        aliases=["seedvr2-7b", "seedvr2-7B"],
        model_name="numz/SeedVR2_comfyUI",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=None,
        max_sequence_length=None,
        supports_guidance=True,
        requires_sigma_shift=None,
        transformer_overrides={
            "vid_dim": 3072,
            "heads": 24,
            "num_layers": 36,
            "mm_layers": 36,
            "rope_dim": 64,
            "rope_on_text": False,
            "rope_freqs_for": "pixel",
            "mlp_type": "normal",
            "use_output_ada": False,
            "last_layer_vid_only": False,
        },
    ),
}


class SeedVR2WeightMapping(WeightMapping):
    @staticmethod
    def get_transformer_mapping(num_blocks: int = 32) -> List[WeightTarget]:
        return [
            WeightTarget(
                to_pattern="vid_in.proj.weight",
                from_pattern=["vid_in.proj.weight"],
            ),
            WeightTarget(
                to_pattern="vid_in.proj.bias",
                from_pattern=["vid_in.proj.bias"],
            ),
            WeightTarget(
                to_pattern="txt_in.weight",
                from_pattern=["txt_in.weight"],
            ),
            WeightTarget(
                to_pattern="txt_in.bias",
                from_pattern=["txt_in.bias"],
            ),
            WeightTarget(
                to_pattern="emb_in.proj_in.weight",
                from_pattern=["emb_in.proj_in.weight"],
            ),
            WeightTarget(
                to_pattern="emb_in.proj_in.bias",
                from_pattern=["emb_in.proj_in.bias"],
            ),
            WeightTarget(
                to_pattern="emb_in.proj_hid.weight",
                from_pattern=["emb_in.proj_hid.weight"],
            ),
            WeightTarget(
                to_pattern="emb_in.proj_hid.bias",
                from_pattern=["emb_in.proj_hid.bias"],
            ),
            WeightTarget(
                to_pattern="emb_in.proj_out.weight",
                from_pattern=["emb_in.proj_out.weight"],
            ),
            WeightTarget(
                to_pattern="emb_in.proj_out.bias",
                from_pattern=["emb_in.proj_out.bias"],
            ),
            WeightTarget(
                to_pattern="blocks.{block}.attn.proj_qkv_vid.weight",
                from_pattern=[
                    "blocks.{block}.attn.proj_qkv.vid.weight",
                    "blocks.{block}.attn.proj_qkv.all.weight",
                ],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.attn.proj_qkv_txt.weight",
                from_pattern=[
                    "blocks.{block}.attn.proj_qkv.txt.weight",
                    "blocks.{block}.attn.proj_qkv.all.weight",
                ],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.attn.norm_q_vid.weight",
                from_pattern=[
                    "blocks.{block}.attn.norm_q.vid.weight",
                    "blocks.{block}.attn.norm_q.all.weight",
                ],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.attn.norm_q_txt.weight",
                from_pattern=[
                    "blocks.{block}.attn.norm_q.txt.weight",
                    "blocks.{block}.attn.norm_q.all.weight",
                ],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.attn.norm_k_vid.weight",
                from_pattern=[
                    "blocks.{block}.attn.norm_k.vid.weight",
                    "blocks.{block}.attn.norm_k.all.weight",
                ],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.attn.norm_k_txt.weight",
                from_pattern=[
                    "blocks.{block}.attn.norm_k.txt.weight",
                    "blocks.{block}.attn.norm_k.all.weight",
                ],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.attn.proj_out_vid.weight",
                from_pattern=[
                    "blocks.{block}.attn.proj_out.vid.weight",
                    "blocks.{block}.attn.proj_out.all.weight",
                ],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.attn.proj_out_vid.bias",
                from_pattern=[
                    "blocks.{block}.attn.proj_out.vid.bias",
                    "blocks.{block}.attn.proj_out.all.bias",
                ],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.attn.proj_out_txt.weight",
                from_pattern=[
                    "blocks.{block}.attn.proj_out.txt.weight",
                    "blocks.{block}.attn.proj_out.all.weight",
                ],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.attn.proj_out_txt.bias",
                from_pattern=[
                    "blocks.{block}.attn.proj_out.txt.bias",
                    "blocks.{block}.attn.proj_out.all.bias",
                ],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.attn.rope.freqs",
                from_pattern=["blocks.{block}.attn.rope.rope.freqs"],
                max_blocks=num_blocks,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.mlp.vid.proj_in.weight",
                from_pattern=["blocks.{block}.mlp.vid.proj_in.weight"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.mlp.vid.proj_in.bias",
                from_pattern=["blocks.{block}.mlp.vid.proj_in.bias"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.mlp.vid.proj_in_gate.weight",
                from_pattern=["blocks.{block}.mlp.vid.proj_in_gate.weight"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.mlp.vid.proj_out.weight",
                from_pattern=["blocks.{block}.mlp.vid.proj_out.weight"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.mlp.vid.proj_out.bias",
                from_pattern=["blocks.{block}.mlp.vid.proj_out.bias"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.mlp.txt.proj_in.weight",
                from_pattern=["blocks.{block}.mlp.txt.proj_in.weight"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.mlp.txt.proj_in.bias",
                from_pattern=["blocks.{block}.mlp.txt.proj_in.bias"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.mlp.txt.proj_in_gate.weight",
                from_pattern=["blocks.{block}.mlp.txt.proj_in_gate.weight"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.mlp.txt.proj_out.weight",
                from_pattern=["blocks.{block}.mlp.txt.proj_out.weight"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.mlp.txt.proj_out.bias",
                from_pattern=["blocks.{block}.mlp.txt.proj_out.bias"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.mlp.all.proj_in.weight",
                from_pattern=["blocks.{block}.mlp.all.proj_in.weight"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.mlp.all.proj_in.bias",
                from_pattern=["blocks.{block}.mlp.all.proj_in.bias"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.mlp.all.proj_in_gate.weight",
                from_pattern=["blocks.{block}.mlp.all.proj_in_gate.weight"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.mlp.all.proj_out.weight",
                from_pattern=["blocks.{block}.mlp.all.proj_out.weight"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.mlp.all.proj_out.bias",
                from_pattern=["blocks.{block}.mlp.all.proj_out.bias"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.ada.params_vid.attn_shift",
                from_pattern=["blocks.{block}.ada.vid.attn_shift"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.ada.params_vid.attn_scale",
                from_pattern=["blocks.{block}.ada.vid.attn_scale"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.ada.params_vid.attn_gate",
                from_pattern=["blocks.{block}.ada.vid.attn_gate"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.ada.params_vid.mlp_shift",
                from_pattern=["blocks.{block}.ada.vid.mlp_shift"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.ada.params_vid.mlp_scale",
                from_pattern=["blocks.{block}.ada.vid.mlp_scale"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.ada.params_vid.mlp_gate",
                from_pattern=["blocks.{block}.ada.vid.mlp_gate"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.ada.params_txt.attn_shift",
                from_pattern=["blocks.{block}.ada.txt.attn_shift"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.ada.params_txt.attn_scale",
                from_pattern=["blocks.{block}.ada.txt.attn_scale"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.ada.params_txt.attn_gate",
                from_pattern=["blocks.{block}.ada.txt.attn_gate"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.ada.params_txt.mlp_shift",
                from_pattern=["blocks.{block}.ada.txt.mlp_shift"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.ada.params_txt.mlp_scale",
                from_pattern=["blocks.{block}.ada.txt.mlp_scale"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.ada.params_txt.mlp_gate",
                from_pattern=["blocks.{block}.ada.txt.mlp_gate"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.ada.params_all.attn_shift",
                from_pattern=["blocks.{block}.ada.all.attn_shift"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.ada.params_all.attn_scale",
                from_pattern=["blocks.{block}.ada.all.attn_scale"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.ada.params_all.attn_gate",
                from_pattern=["blocks.{block}.ada.all.attn_gate"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.ada.params_all.mlp_shift",
                from_pattern=["blocks.{block}.ada.all.mlp_shift"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.ada.params_all.mlp_scale",
                from_pattern=["blocks.{block}.ada.all.mlp_scale"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="blocks.{block}.ada.params_all.mlp_gate",
                from_pattern=["blocks.{block}.ada.all.mlp_gate"],
                max_blocks=num_blocks,
                required=False,
            ),
            WeightTarget(
                to_pattern="vid_out_norm.weight",
                from_pattern=["vid_out_norm.weight"],
            ),
            WeightTarget(
                to_pattern="out_shift",
                from_pattern=["vid_out_ada.out_shift"],
            ),
            WeightTarget(
                to_pattern="out_scale",
                from_pattern=["vid_out_ada.out_scale"],
            ),
            WeightTarget(
                to_pattern="vid_out.proj.weight",
                from_pattern=["vid_out.proj.weight"],
            ),
            WeightTarget(
                to_pattern="vid_out.proj.bias",
                from_pattern=["vid_out.proj.bias"],
            ),
        ]

    @staticmethod
    def get_vae_mapping() -> List[WeightTarget]:
        return [
            WeightTarget(
                to_pattern="encoder.conv_in.weight",
                from_pattern=["encoder.conv_in.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.conv_in.bias",
                from_pattern=["encoder.conv_in.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.{block}.resnets.{res}.conv1.weight",
                from_pattern=["encoder.down_blocks.{block}.resnets.{res}.conv1.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.{block}.resnets.{res}.conv1.bias",
                from_pattern=["encoder.down_blocks.{block}.resnets.{res}.conv1.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.{block}.resnets.{res}.conv2.weight",
                from_pattern=["encoder.down_blocks.{block}.resnets.{res}.conv2.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.{block}.resnets.{res}.conv2.bias",
                from_pattern=["encoder.down_blocks.{block}.resnets.{res}.conv2.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.{block}.resnets.{res}.norm1.weight",
                from_pattern=["encoder.down_blocks.{block}.resnets.{res}.norm1.weight"],
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.{block}.resnets.{res}.norm1.bias",
                from_pattern=["encoder.down_blocks.{block}.resnets.{res}.norm1.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.{block}.resnets.{res}.norm2.weight",
                from_pattern=["encoder.down_blocks.{block}.resnets.{res}.norm2.weight"],
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.{block}.resnets.{res}.norm2.bias",
                from_pattern=["encoder.down_blocks.{block}.resnets.{res}.norm2.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.{block}.resnets.{res}.conv_shortcut.weight",
                from_pattern=["encoder.down_blocks.{block}.resnets.{res}.conv_shortcut.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
                required=False,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.{block}.resnets.{res}.conv_shortcut.bias",
                from_pattern=["encoder.down_blocks.{block}.resnets.{res}.conv_shortcut.bias"],
                required=False,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.{block}.downsamplers.0.conv.weight",
                from_pattern=["encoder.down_blocks.{block}.downsamplers.0.conv.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
                required=False,
            ),
            WeightTarget(
                to_pattern="encoder.down_blocks.{block}.downsamplers.0.conv.bias",
                from_pattern=["encoder.down_blocks.{block}.downsamplers.0.conv.bias"],
                required=False,
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.resnets.{i}.conv1.weight",
                from_pattern=["encoder.mid_block.resnets.{i}.conv1.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.resnets.{i}.conv1.bias",
                from_pattern=["encoder.mid_block.resnets.{i}.conv1.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.resnets.{i}.conv2.weight",
                from_pattern=["encoder.mid_block.resnets.{i}.conv2.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.resnets.{i}.conv2.bias",
                from_pattern=["encoder.mid_block.resnets.{i}.conv2.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.resnets.{i}.norm1.weight",
                from_pattern=["encoder.mid_block.resnets.{i}.norm1.weight"],
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.resnets.{i}.norm1.bias",
                from_pattern=["encoder.mid_block.resnets.{i}.norm1.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.resnets.{i}.norm2.weight",
                from_pattern=["encoder.mid_block.resnets.{i}.norm2.weight"],
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.resnets.{i}.norm2.bias",
                from_pattern=["encoder.mid_block.resnets.{i}.norm2.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.attentions.0.group_norm.weight",
                from_pattern=["encoder.mid_block.attentions.0.group_norm.weight"],
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.attentions.0.group_norm.bias",
                from_pattern=["encoder.mid_block.attentions.0.group_norm.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.attentions.0.to_q.weight",
                from_pattern=["encoder.mid_block.attentions.0.to_q.weight"],
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.attentions.0.to_q.bias",
                from_pattern=["encoder.mid_block.attentions.0.to_q.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.attentions.0.to_k.weight",
                from_pattern=["encoder.mid_block.attentions.0.to_k.weight"],
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.attentions.0.to_k.bias",
                from_pattern=["encoder.mid_block.attentions.0.to_k.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.attentions.0.to_v.weight",
                from_pattern=["encoder.mid_block.attentions.0.to_v.weight"],
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.attentions.0.to_v.bias",
                from_pattern=["encoder.mid_block.attentions.0.to_v.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.attentions.0.to_out.0.weight",
                from_pattern=["encoder.mid_block.attentions.0.to_out.0.weight"],
            ),
            WeightTarget(
                to_pattern="encoder.mid_block.attentions.0.to_out.0.bias",
                from_pattern=["encoder.mid_block.attentions.0.to_out.0.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.conv_norm_out.weight",
                from_pattern=["encoder.conv_norm_out.weight"],
            ),
            WeightTarget(
                to_pattern="encoder.conv_norm_out.bias",
                from_pattern=["encoder.conv_norm_out.bias"],
            ),
            WeightTarget(
                to_pattern="encoder.conv_out.weight",
                from_pattern=["encoder.conv_out.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="encoder.conv_out.bias",
                from_pattern=["encoder.conv_out.bias"],
            ),
            WeightTarget(
                to_pattern="decoder.conv_in.weight",
                from_pattern=["decoder.conv_in.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="decoder.conv_in.bias",
                from_pattern=["decoder.conv_in.bias"],
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.resnets.{i}.conv1.weight",
                from_pattern=["decoder.mid_block.resnets.{i}.conv1.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.resnets.{i}.conv1.bias",
                from_pattern=["decoder.mid_block.resnets.{i}.conv1.bias"],
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.resnets.{i}.conv2.weight",
                from_pattern=["decoder.mid_block.resnets.{i}.conv2.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.resnets.{i}.conv2.bias",
                from_pattern=["decoder.mid_block.resnets.{i}.conv2.bias"],
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.resnets.{i}.norm1.weight",
                from_pattern=["decoder.mid_block.resnets.{i}.norm1.weight"],
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.resnets.{i}.norm1.bias",
                from_pattern=["decoder.mid_block.resnets.{i}.norm1.bias"],
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.resnets.{i}.norm2.weight",
                from_pattern=["decoder.mid_block.resnets.{i}.norm2.weight"],
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.resnets.{i}.norm2.bias",
                from_pattern=["decoder.mid_block.resnets.{i}.norm2.bias"],
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.attentions.0.group_norm.weight",
                from_pattern=["decoder.mid_block.attentions.0.group_norm.weight"],
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.attentions.0.group_norm.bias",
                from_pattern=["decoder.mid_block.attentions.0.group_norm.bias"],
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.attentions.0.to_q.weight",
                from_pattern=["decoder.mid_block.attentions.0.to_q.weight"],
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.attentions.0.to_q.bias",
                from_pattern=["decoder.mid_block.attentions.0.to_q.bias"],
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.attentions.0.to_k.weight",
                from_pattern=["decoder.mid_block.attentions.0.to_k.weight"],
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.attentions.0.to_k.bias",
                from_pattern=["decoder.mid_block.attentions.0.to_k.bias"],
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.attentions.0.to_v.weight",
                from_pattern=["decoder.mid_block.attentions.0.to_v.weight"],
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.attentions.0.to_v.bias",
                from_pattern=["decoder.mid_block.attentions.0.to_v.bias"],
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.attentions.0.to_out.0.weight",
                from_pattern=["decoder.mid_block.attentions.0.to_out.0.weight"],
            ),
            WeightTarget(
                to_pattern="decoder.mid_block.attentions.0.to_out.0.bias",
                from_pattern=["decoder.mid_block.attentions.0.to_out.0.bias"],
            ),
            WeightTarget(
                to_pattern="decoder.up_blocks.{block}.resnets.{res}.conv1.weight",
                from_pattern=["decoder.up_blocks.{block}.resnets.{res}.conv1.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="decoder.up_blocks.{block}.resnets.{res}.conv1.bias",
                from_pattern=["decoder.up_blocks.{block}.resnets.{res}.conv1.bias"],
            ),
            WeightTarget(
                to_pattern="decoder.up_blocks.{block}.resnets.{res}.conv2.weight",
                from_pattern=["decoder.up_blocks.{block}.resnets.{res}.conv2.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="decoder.up_blocks.{block}.resnets.{res}.conv2.bias",
                from_pattern=["decoder.up_blocks.{block}.resnets.{res}.conv2.bias"],
            ),
            WeightTarget(
                to_pattern="decoder.up_blocks.{block}.resnets.{res}.norm1.weight",
                from_pattern=["decoder.up_blocks.{block}.resnets.{res}.norm1.weight"],
            ),
            WeightTarget(
                to_pattern="decoder.up_blocks.{block}.resnets.{res}.norm1.bias",
                from_pattern=["decoder.up_blocks.{block}.resnets.{res}.norm1.bias"],
            ),
            WeightTarget(
                to_pattern="decoder.up_blocks.{block}.resnets.{res}.norm2.weight",
                from_pattern=["decoder.up_blocks.{block}.resnets.{res}.norm2.weight"],
            ),
            WeightTarget(
                to_pattern="decoder.up_blocks.{block}.resnets.{res}.norm2.bias",
                from_pattern=["decoder.up_blocks.{block}.resnets.{res}.norm2.bias"],
            ),
            WeightTarget(
                to_pattern="decoder.up_blocks.{block}.resnets.{res}.conv_shortcut.weight",
                from_pattern=["decoder.up_blocks.{block}.resnets.{res}.conv_shortcut.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
                required=False,
            ),
            WeightTarget(
                to_pattern="decoder.up_blocks.{block}.resnets.{res}.conv_shortcut.bias",
                from_pattern=["decoder.up_blocks.{block}.resnets.{res}.conv_shortcut.bias"],
                required=False,
            ),
            WeightTarget(
                to_pattern="decoder.up_blocks.{block}.upsamplers.0.conv.weight",
                from_pattern=["decoder.up_blocks.{block}.upsamplers.0.conv.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
                required=False,
            ),
            WeightTarget(
                to_pattern="decoder.up_blocks.{block}.upsamplers.0.conv.bias",
                from_pattern=["decoder.up_blocks.{block}.upsamplers.0.conv.bias"],
                required=False,
            ),
            WeightTarget(
                to_pattern="decoder.up_blocks.{block}.upsamplers.0.upscale_conv.weight",
                from_pattern=["decoder.up_blocks.{block}.upsamplers.0.upscale_conv.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
                required=False,
            ),
            WeightTarget(
                to_pattern="decoder.up_blocks.{block}.upsamplers.0.upscale_conv.bias",
                from_pattern=["decoder.up_blocks.{block}.upsamplers.0.upscale_conv.bias"],
                required=False,
            ),
            WeightTarget(
                to_pattern="decoder.conv_norm_out.weight",
                from_pattern=["decoder.conv_norm_out.weight"],
            ),
            WeightTarget(
                to_pattern="decoder.conv_norm_out.bias",
                from_pattern=["decoder.conv_norm_out.bias"],
            ),
            WeightTarget(
                to_pattern="decoder.conv_out.weight",
                from_pattern=["decoder.conv_out.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            ),
            WeightTarget(
                to_pattern="decoder.conv_out.bias",
                from_pattern=["decoder.conv_out.bias"],
            ),
        ]


# ----- WeightLoader / WeightApplier 组件定义（原 ``runtime/.../seedvr2_weight_definition.py``） -----


class SeedVR2WeightDefinition3B:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="transformer",
                hf_subdir=".",
                num_blocks=32,
                loading_mode="mlx_native",
                mapping_getter=lambda: SeedVR2WeightMapping.get_transformer_mapping(num_blocks=32),
                weight_files=["seedvr2_ema_3b_fp16.safetensors"],
            ),
            ComponentDefinition(
                name="vae",
                hf_subdir=".",
                num_blocks=4,
                loading_mode="mlx_native",
                mapping_getter=SeedVR2WeightMapping.get_vae_mapping,
                weight_files=["ema_vae_fp16.safetensors"],
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return []

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "seedvr2_ema_3b_fp16.safetensors",
            "ema_vae_fp16.safetensors",
        ]


class SeedVR2WeightDefinition7B:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="transformer",
                hf_subdir=".",
                num_blocks=36,
                loading_mode="mlx_native",
                mapping_getter=lambda: SeedVR2WeightMapping.get_transformer_mapping(num_blocks=36),
                weight_files=["seedvr2_ema_7b_fp16.safetensors"],
            ),
            ComponentDefinition(
                name="vae",
                hf_subdir=".",
                num_blocks=4,
                loading_mode="mlx_native",
                mapping_getter=SeedVR2WeightMapping.get_vae_mapping,
                weight_files=["ema_vae_fp16.safetensors"],
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return []

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "seedvr2_ema_7b_fp16.safetensors",
            "ema_vae_fp16.safetensors",
        ]


class SeedVR2WeightDefinition:
    @staticmethod
    def resolve(model_config) -> type[SeedVR2WeightDefinition3B | SeedVR2WeightDefinition7B]:
        aliases = {a.lower() for a in getattr(model_config, "aliases", [])}
        if "seedvr2-7b" in aliases:
            return SeedVR2WeightDefinition7B
        return SeedVR2WeightDefinition3B

    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return SeedVR2WeightDefinition3B.get_components()

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return []

    @staticmethod
    def get_download_patterns() -> List[str]:
        return SeedVR2WeightDefinition3B.get_download_patterns()


def load_flat_bundle(
    bundle_path: str | Path,
    model_config: ModelConfig,
) -> tuple[LoadedWeights, type]:
    """从目录加载 ``safetensors``，返回 ``LoadedWeights`` 与解析出的权重定义类。"""
    definition_cls = SeedVR2WeightDefinition.resolve(model_config)
    weights = WeightLoader.load(
        weight_definition=definition_cls,
        model_path=str(bundle_path),
    )
    return weights, definition_cls
