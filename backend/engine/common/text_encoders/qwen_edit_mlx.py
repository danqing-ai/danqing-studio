"""Qwen-Image-Edit MLX：VL vision 栈 + 图文 prompt 编码（对齐 mflux QwenImageEdit）。"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx import nn
from PIL import Image

# --- from qwen_image_processor.py ---
from typing import Optional, Union

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class QwenImageProcessor:
    def __init__(
        self,
        min_pixels: int = 56 * 56,
        max_pixels: int = 28 * 28 * 1280,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        image_mean: Optional[list[float]] = None,
        image_std: Optional[list[float]] = None,
    ):
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD

    def _preprocess(
        self,
        image: Image.Image,
        resized_height: Optional[int] = None,
        resized_width: Optional[int] = None,
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        height, width = image.size[1], image.size[0]

        if resized_height is None or resized_width is None:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=self.patch_size * self.merge_size,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
        if (height, width) != (resized_height, resized_width):
            image = image.resize((resized_width, resized_height), Image.BICUBIC)

        image_np = np.array(image).astype(np.float32)
        image_np = image_np / 255.0

        mean_np = np.array(self.image_mean, dtype=np.float32)
        std_np = np.array(self.image_std, dtype=np.float32)
        image_np = (image_np - mean_np) / std_np

        image_np = image_np.transpose(2, 0, 1)
        patches = image_np[np.newaxis]  # Shape: (1, channel, height, width)

        if patches.shape[0] % self.temporal_patch_size != 0:
            repeats = np.repeat(
                patches[-1][np.newaxis],
                self.temporal_patch_size - (patches.shape[0] % self.temporal_patch_size),
                axis=0,
            )
            patches = np.concatenate([patches, repeats], axis=0)

        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h = resized_height // self.patch_size
        grid_w = resized_width // self.patch_size

        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )

        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)

        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w,
            channel * self.temporal_patch_size * self.patch_size * self.patch_size,
        )

        return flatten_patches, (grid_t, grid_h, grid_w)

    def preprocess(
        self,
        images: Union[Image.Image, list[Image.Image]],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(images, list):
            images = [images]

        pixel_values_list = []
        vision_grid_thws = []

        for image in images:
            patches, image_grid_thw = self._preprocess(image)
            pixel_values_list.append(patches)
            vision_grid_thws.append([image_grid_thw[0], image_grid_thw[1], image_grid_thw[2]])

        # Concatenate all patches from all images along the patch dimension
        pixel_values = np.concatenate(pixel_values_list, axis=0) if pixel_values_list else np.array([])

        vision_grid_thws = np.array(vision_grid_thws)

        return pixel_values, vision_grid_thws

    def get_number_of_image_patches(
        self,
        height: int,
        width: int,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ) -> int:
        min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        max_pixels = max_pixels if max_pixels is not None else self.max_pixels

        factor = self.patch_size * self.merge_size
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        grid_h = resized_height // self.patch_size
        grid_w = resized_width // self.patch_size
        return grid_h * grid_w

# --- from qwen_vision_rotary_embedding.py ---
import mlx.core as mx
from mlx import nn


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self.inv_freq = inv_freq

    def __call__(self, max_grid_size: int) -> mx.array:
        positions = mx.arange(max_grid_size, dtype=mx.float32)
        freqs = mx.outer(positions, self.inv_freq)
        return freqs

# --- from qwen_vision_patch_embed.py ---
import mlx.core as mx
from mlx import nn


class VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1280,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=[temporal_patch_size, patch_size, patch_size],
            stride=[temporal_patch_size, patch_size, patch_size],
            bias=False,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(
            batch_size, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = hidden_states.transpose(0, 2, 3, 4, 1)
        output = self.proj(hidden_states)
        return output.reshape(batch_size, self.embed_dim)

# --- from qwen_vision_mlp.py ---
import mlx.core as mx
from mlx import nn


class VisionMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=True)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=True)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        gate = nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

# --- from qwen_vision_attention.py ---
import mlx.core as mx
from mlx import nn


class VisionAttention(nn.Module):
    def __init__(self, embed_dim: int = 1280, num_heads: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def _rotate_half(self, x: mx.array) -> mx.array:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return mx.concatenate([-x2, x1], axis=-1)

    def _apply_rope(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        orig_dtype = x.dtype
        x = x.astype(mx.float32)
        cos_expanded = mx.expand_dims(cos, axis=0).astype(mx.float32)
        sin_expanded = mx.expand_dims(sin, axis=0).astype(mx.float32)
        rotated = (x * cos_expanded) + (self._rotate_half(x) * sin_expanded)
        return rotated.astype(orig_dtype)

    def __call__(self, x: mx.array, position_embeddings=None, cu_seqlens=None) -> mx.array:
        seq_len, embed_dim = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = mx.split(qkv, 3, axis=1)
        q = q.squeeze(1).transpose(1, 0, 2)
        k = k.squeeze(1).transpose(1, 0, 2)
        v = v.squeeze(1).transpose(1, 0, 2)

        if position_embeddings is not None:
            cos_emb, sin_emb = position_embeddings
            if cos_emb.shape[0] != seq_len:
                cos_emb = cos_emb[:seq_len]
                sin_emb = sin_emb[:seq_len]

            q = self._apply_rope(q, cos_emb, sin_emb)
            k = self._apply_rope(k, cos_emb, sin_emb)

        scale = 1.0 / (self.head_dim**0.5)

        # Process attention chunks if cu_seqlens is provided (windowed attention)
        if cu_seqlens is not None and len(cu_seqlens) > 2:
            lengths = [int((cu_seqlens[i + 1] - cu_seqlens[i]).item()) for i in range(len(cu_seqlens) - 1)]

            attn_outputs = []
            offset = 0
            for length in lengths:
                q_chunk = mx.expand_dims(q[:, offset : offset + length, :], axis=0)
                k_chunk = mx.expand_dims(k[:, offset : offset + length, :], axis=0)
                v_chunk = mx.expand_dims(v[:, offset : offset + length, :], axis=0)
                offset += length
                out = mx.fast.scaled_dot_product_attention(q_chunk, k_chunk, v_chunk, scale=scale)
                attn_outputs.append(out.squeeze(0))

            attn_output = mx.concatenate(attn_outputs, axis=1)  # [heads, seq, head_dim]
        else:
            # Full attention (no chunking)
            q_4d = mx.expand_dims(q, axis=0)
            k_4d = mx.expand_dims(k, axis=0)
            v_4d = mx.expand_dims(v, axis=0)
            attn_output = mx.fast.scaled_dot_product_attention(q_4d, k_4d, v_4d, scale=scale)
            attn_output = attn_output.squeeze(0)  # [heads, seq, head_dim]

        # Reshape and project
        attn_output = attn_output.transpose(1, 0, 2).reshape(seq_len, embed_dim)  # [seq, embed_dim]
        return self.proj(attn_output)

# --- from qwen_vision_block.py ---
import mlx.core as mx
from mlx import nn



class VisionBlock(nn.Module):
    def __init__(self, embed_dim: int = 1280, num_heads: int = 16, mlp_ratio: float = 2.671875):
        super().__init__()
        self.norm1 = nn.RMSNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.RMSNorm(embed_dim, eps=1e-6)
        self.attn = VisionAttention(embed_dim, num_heads)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = VisionMLP(embed_dim, mlp_hidden_dim)

    def __call__(self, x: mx.array, position_embeddings=None, cu_seqlens=None) -> mx.array:
        normed1 = self.norm1(x)
        attn_out = self.attn(normed1, position_embeddings, cu_seqlens)
        x = x + attn_out
        normed2 = self.norm2(x)
        mlp_out = self.mlp(normed2)
        x = x + mlp_out
        return x

# --- from qwen_patch_merger.py ---
import mlx.core as mx
from mlx import nn


class PatchMerger(nn.Module):
    def __init__(self, context_dim: int, hidden_size: int, spatial_merge_size: int = 2):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size_merged = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.RMSNorm(context_dim, eps=1e-6)
        self.mlp_0 = nn.Linear(self.hidden_size_merged, self.hidden_size_merged, bias=True)
        self.mlp_1 = nn.Linear(self.hidden_size_merged, hidden_size, bias=True)

    def __call__(self, x: mx.array, grid_thw: mx.array) -> mx.array:
        if not hasattr(self, "_weights_logged"):
            self._weights_logged = True
        x = self.ln_q(x)
        merged_patches = []
        offset = 0
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            num_patches_this_image = t * h * w
            x_this_image = x[offset : offset + num_patches_this_image]
            x_merged = x_this_image.reshape(-1, self.hidden_size_merged)
            merged_patches.append(x_merged)
            offset += num_patches_this_image

        x = mx.concatenate(merged_patches, axis=0)
        x = self.mlp_0(x)
        x = nn.gelu(x)
        x = self.mlp_1(x)
        return x

# --- from qwen_vision_transformer.py ---
import mlx.core as mx
import numpy as np
from mlx import nn



class VisionTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1280,
        depth: int = 32,
        num_heads: int = 16,
        mlp_ratio: float = 2.671875,
        hidden_size: int = 3584,
        spatial_merge_size: int = 2,
        window_size: int = 112,
        fullatt_block_indexes: list = None,
    ):
        super().__init__()

        self.patch_embed = VisionPatchEmbed(patch_size, temporal_patch_size, in_channels, embed_dim)
        self.spatial_merge_size = spatial_merge_size
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes or [7, 15, 23, 31]
        self.spatial_merge_unit = spatial_merge_size * spatial_merge_size
        self.patch_size = patch_size

        head_dim = embed_dim // num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = [VisionBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        self.merger = PatchMerger(embed_dim, hidden_size, spatial_merge_size)

    def get_window_index(self, grid_thw: mx.array):
        window_index = []
        cu_window_seqlens = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.patch_size // self.spatial_merge_size

        for t, grid_h, grid_w in grid_thw:
            t, grid_h, grid_w = int(t), int(grid_h), int(grid_w)
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size

            index = mx.arange(t * llm_grid_h * llm_grid_w).reshape(t, llm_grid_h, llm_grid_w)

            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size

            index_padded = mx.pad(index, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=-100)

            index_padded = index_padded.reshape(
                t, num_windows_h, vit_merger_window_size, num_windows_w, vit_merger_window_size
            )
            index_padded = mx.transpose(index_padded, (0, 1, 3, 2, 4)).reshape(
                t, num_windows_h * num_windows_w, vit_merger_window_size, vit_merger_window_size
            )

            seqlens = mx.sum((index_padded != -100).astype(mx.int32), axis=(2, 3)).reshape(-1)
            index_padded_flat = index_padded.reshape(-1)

            index_padded_np = np.array(index_padded_flat)
            index_new = mx.array(index_padded_np[index_padded_np != -100])

            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = mx.cumsum(seqlens) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += t * llm_grid_h * llm_grid_w

        window_index = mx.concatenate(window_index, axis=0)
        cu_window_seqlens = mx.array(cu_window_seqlens, dtype=mx.int32)

        return window_index, cu_window_seqlens

    def rot_pos_emb(self, grid_thw: mx.array) -> mx.array:
        pos_ids = []
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            hpos_ids = mx.repeat(mx.arange(h)[..., None], w, axis=1)
            wpos_ids = mx.repeat(mx.arange(w)[None, ...], h, axis=0)
            merge_h = h // self.spatial_merge_size
            merge_w = w // self.spatial_merge_size
            hpos_ids = hpos_ids.reshape(merge_h, self.spatial_merge_size, merge_w, self.spatial_merge_size)
            wpos_ids = wpos_ids.reshape(merge_h, self.spatial_merge_size, merge_w, self.spatial_merge_size)
            hpos_ids = mx.transpose(hpos_ids, (0, 2, 1, 3))
            wpos_ids = mx.transpose(wpos_ids, (0, 2, 1, 3))
            hpos_ids = hpos_ids.reshape(-1)
            wpos_ids = wpos_ids.reshape(-1)
            pos_id_pair = mx.stack([hpos_ids, wpos_ids], axis=-1)
            pos_id_pair = mx.tile(pos_id_pair, (t, 1))
            pos_ids.append(pos_id_pair)
        pos_ids = mx.concatenate(pos_ids, axis=0)
        max_grid_size = int(mx.max(grid_thw[:, 1:]).item())
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        h_indices = pos_ids[:, 0].astype(mx.int32)
        w_indices = pos_ids[:, 1].astype(mx.int32)
        h_emb = rotary_pos_emb_full[h_indices]
        w_emb = rotary_pos_emb_full[w_indices]
        rotary_pos_emb = mx.stack([h_emb, w_emb], axis=1)
        rotary_pos_emb = rotary_pos_emb.reshape(rotary_pos_emb.shape[0], -1)
        return rotary_pos_emb

    def __call__(self, pixel_values: mx.array, grid_thw: mx.array) -> mx.array:
        hidden_states = self.patch_embed(pixel_values)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens_unique = [cu_window_seqlens[0].item()]
        for i in range(1, len(cu_window_seqlens)):
            if cu_window_seqlens[i].item() != cu_window_seqlens_unique[-1]:
                cu_window_seqlens_unique.append(cu_window_seqlens[i].item())
        cu_window_seqlens = mx.array(cu_window_seqlens_unique, dtype=mx.int32)
        seq_len = hidden_states.shape[0]
        cu_seqlens = []
        offset = 0
        for t, h, w in grid_thw:
            t, h, w = int(t), int(h), int(w)
            length = t * h * w
            offset += length
            cu_seqlens.append(offset)
        cu_seqlens = mx.array([0] + cu_seqlens, dtype=mx.int32)
        seq_len = hidden_states.shape[0]
        num_groups = seq_len // self.spatial_merge_unit
        hidden_states_grouped = hidden_states.reshape(num_groups, self.spatial_merge_unit, -1)
        hidden_states_grouped = hidden_states_grouped[window_index.astype(mx.int32), :, :]
        hidden_states = hidden_states_grouped.reshape(seq_len, -1)
        rotary_pos_emb_grouped = rotary_pos_emb.reshape(num_groups, self.spatial_merge_unit, -1)
        rotary_pos_emb_grouped = rotary_pos_emb_grouped[window_index.astype(mx.int32), :, :]
        rotary_pos_emb = rotary_pos_emb_grouped.reshape(seq_len, -1)
        emb = mx.concatenate([rotary_pos_emb, rotary_pos_emb], axis=-1)
        position_embeddings = (mx.cos(emb), mx.sin(emb))
        for layer_num, block in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            hidden_states = block(hidden_states, position_embeddings, cu_seqlens_now)
        hidden_states = self.merger(hidden_states, grid_thw)
        reverse_indices = mx.argsort(window_index.astype(mx.int32))
        hidden_states = hidden_states[reverse_indices.astype(mx.int32), :]
        return hidden_states

# --- from qwen_vision_language_encoder.py ---
import mlx.core as mx
from mlx import nn



class QwenVisionLanguageEncoder(nn.Module):
    def __init__(self, encoder=None):
        from backend.engine.common.text_encoders.qwen_image_mlx import QwenEncoder

        super().__init__()
        self.encoder = encoder or QwenEncoder()
        self.edit_template_start_idx = 64

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        hidden_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        trimmed = []
        batch = hidden_states.shape[0]
        for i in range(batch):
            valid_len = int(mx.sum(attention_mask[i]).item())
            trimmed.append(hidden_states[i, :valid_len, :])
        drop_idx = self.edit_template_start_idx
        trimmed_after_drop = [t[drop_idx:] if t.shape[0] > drop_idx else t for t in trimmed]
        trimmed = trimmed_after_drop
        max_len = max(t.shape[0] for t in trimmed) if trimmed else 0
        hidden_dim = hidden_states.shape[2]
        padded_embeds = []
        padded_masks = []
        for t in trimmed:
            cur_len = t.shape[0]
            if cur_len < max_len:
                pad_e = mx.zeros((max_len - cur_len, hidden_dim), dtype=t.dtype)
                t_pad = mx.concatenate([t, pad_e], axis=0)
                pad_m = mx.concatenate(
                    [mx.ones(cur_len, dtype=mx.int32), mx.zeros(max_len - cur_len, dtype=mx.int32)], axis=0
                )
            else:
                t_pad = t
                pad_m = mx.ones(cur_len, dtype=mx.int32)
            padded_embeds.append(t_pad)
            padded_masks.append(pad_m)
        prompt_embeds = mx.stack(padded_embeds, axis=0) if padded_embeds else hidden_states
        encoder_attention_mask = mx.stack(padded_masks, axis=0) if padded_masks else attention_mask
        return prompt_embeds, encoder_attention_mask

# --- from qwen_vision_language_processor.py ---
from typing import Optional, Union

import mlx.core as mx
import numpy as np
from PIL import Image



class QwenVisionLanguageProcessor:
    def __init__(
        self,
        tokenizer,
        image_processor: Optional[QwenImageProcessor] = None,
        image_token: str = "<|image_pad|>",
        video_token: str = "<|video_pad|>",
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor or QwenImageProcessor()

        self.image_token = image_token
        self.video_token = video_token
        self.image_token_id = (
            tokenizer.image_token_id
            if hasattr(tokenizer, "image_token_id")
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if hasattr(tokenizer, "video_token_id")
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )

    def __call__(
        self,
        images: Optional[Union[Image.Image, list[Image.Image]]] = None,
        text: Optional[Union[str, list[str]]] = None,
        padding: bool = True,
        return_tensors: Optional[str] = None,
    ) -> dict:
        image_inputs = {}
        if images is not None:
            pixel_values, image_grid_thw = self.image_processor.preprocess(images)
            image_inputs = {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }

        if text is not None:
            if not isinstance(text, list):
                text = [text]
            text = text.copy()

            if images is not None:
                merge_length = self.image_processor.merge_size**2
                index = 0
                for i in range(len(text)):
                    while self.image_token in text[i]:
                        if index < len(image_grid_thw):
                            num_image_tokens = int(np.prod(image_grid_thw[index])) // merge_length
                            text[i] = text[i].replace(
                                self.image_token,
                                "<|placeholder|>" * num_image_tokens,
                                1,
                            )
                            index += 1
                        else:
                            break
                    text[i] = text[i].replace("<|placeholder|>", self.image_token)

            text_inputs = self.tokenizer(
                text,
                padding=padding,
                return_tensors="pt" if return_tensors == "pt" else "np",
            )

            if return_tensors == "pt":
                import torch

                if isinstance(text_inputs["input_ids"], torch.Tensor):
                    input_ids = mx.array(text_inputs["input_ids"].numpy())
                    attention_mask = mx.array(text_inputs["attention_mask"].numpy())
                else:
                    input_ids = mx.array(text_inputs["input_ids"])
                    attention_mask = mx.array(text_inputs["attention_mask"])
            else:
                if isinstance(text_inputs["input_ids"], np.ndarray):
                    input_ids = mx.array(text_inputs["input_ids"])
                    attention_mask = mx.array(text_inputs["attention_mask"])
                else:
                    input_ids = mx.array(np.array(text_inputs["input_ids"]))
                    attention_mask = mx.array(np.array(text_inputs["attention_mask"]))
        else:
            input_ids = None
            attention_mask = None

        result = {**image_inputs}
        if input_ids is not None:
            result["input_ids"] = input_ids
        if attention_mask is not None:
            result["attention_mask"] = attention_mask

        return result

# --- from qwen_vision_language_tokenizer.py ---
import math
from pathlib import Path
from typing import Union

import mlx.core as mx
import numpy as np
from PIL import Image



class QwenVisionLanguageTokenizer:
    def __init__(
        self,
        processor: QwenVisionLanguageProcessor,
        max_length: int = 1024,
        use_picture_prefix: bool = True,
    ):
        self.processor = processor
        self.max_length = max_length
        self.use_picture_prefix = use_picture_prefix

        if use_picture_prefix:
            self.edit_template = (
                "<|im_start|>system\n"
                "Describe the key features of the input image (color, shape, size, texture, objects, background), "
                "then explain how the user's text instruction should alter or modify the image. "
                "Generate a new image that meets the user's requirements while maintaining consistency "
                "with the original input where appropriate.<|im_end|>\n"
                "<|im_start|>user\n"
                "{}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        else:
            self.edit_template = (
                "<|im_start|>system\n"
                "Describe the key features of the input image (color, shape, size, texture, objects, background), "
                "then explain how the user's text instruction should alter or modify the image. "
                "Generate a new image that meets the user's requirements while maintaining consistency "
                "with the original input where appropriate.<|im_end|>\n"
                "<|im_start|>user\n"
                "<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        self.edit_template_start_idx = 64

    def tokenize_with_image(
        self,
        prompt: str,
        image: Union[Image.Image, np.ndarray, str, list],
        vl_width: int | None = None,
        vl_height: int | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        # Normalize image to list format
        if not isinstance(image, list):
            images = [image]
        else:
            images = image

        # Format prompt based on tokenizer mode
        if self.use_picture_prefix:
            # Edit format: Add "Picture N:" prefix for each image
            # For multiple images: "Picture 1: ... Picture 2: ... Picture N: ..."
            img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
            base_img_prompt = ""
            for i in range(len(images)):
                base_img_prompt += img_prompt_template.format(i + 1)
            formatted_text = self.edit_template.format(base_img_prompt + prompt)
        else:
            # Regular Edit format: Vision tokens already in template
            # Just format with user prompt directly
            formatted_text = self.edit_template.format(prompt)

        # Process images: convert to PIL Images and resize to CONDITION_IMAGE_SIZE
        CONDITION_IMAGE_SIZE = 384 * 384

        processed_images = []
        for img in images:
            # Convert to PIL Image
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert("RGB")
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            elif not isinstance(img, Image.Image):
                raise ValueError(f"Unsupported image type: {type(img)}")

            # Resize to CONDITION_IMAGE_SIZE (384×384) maintaining aspect ratio
            img_w, img_h = img.size
            ratio = img_w / img_h
            condition_width = math.sqrt(CONDITION_IMAGE_SIZE * ratio)
            condition_height = condition_width / ratio
            condition_width = round(condition_width / 32) * 32
            condition_height = round(condition_height / 32) * 32

            img = img.resize((int(condition_width), int(condition_height)), Image.BICUBIC)
            processed_images.append(img)

        # Use our MLX processor for both text and images
        model_inputs = self.processor(
            text=[formatted_text],
            images=processed_images,
            padding=True,
            return_tensors=None,  # Return numpy/MLX arrays, not PyTorch
        )

        grid_thw = model_inputs["image_grid_thw"][0]
        factor = 14 * 2
        self._vl_image_width = int(grid_thw[2]) * factor
        self._vl_image_height = int(grid_thw[1]) * factor

        # Convert to MLX arrays if needed
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        pixel_values = mx.array(model_inputs["pixel_values"])
        image_grid_thw = mx.array(model_inputs["image_grid_thw"])

        return input_ids, attention_mask, pixel_values, image_grid_thw

    def tokenize_text_only(self, prompt: str) -> tuple[mx.array, mx.array]:
        # Use the regular text-only template
        text_template = (
            "<|im_start|>system\n"
            "Describe the image by detailing the color, shape, size, texture, quantity, text, "
            "spatial relationships of the objects and background:<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        formatted_text = text_template.format(prompt)
        tokens = self.processor.tokenizer(
            formatted_text,
            max_length=self.max_length + 34,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Convert PyTorch tensors to MLX arrays
        input_ids = mx.array(tokens["input_ids"].numpy())
        attention_mask = mx.array(tokens["attention_mask"].numpy())

        return input_ids, attention_mask


def load_qwen_edit_vl_encoder(bundle_root: Path, ctx: Any) -> QwenVisionLanguageEncoder:
    """加载带 ``visual`` 的 Qwen2.5-VL trunk（Edit 图文编码）。"""
    from backend.engine.common.text_encoders.qwen_image_mlx import load_qwen25vl_mlx_encoder

    encoder = load_qwen25vl_mlx_encoder(
        bundle_root,
        ctx=ctx,
        load_fn=getattr(ctx, "load_weights", None),
        strip_visual=False,
    )
    return QwenVisionLanguageEncoder(encoder=encoder)


def build_qwen_edit_vl_tokenizer(tok_root: Path) -> QwenVisionLanguageTokenizer:
    from backend.engine.common.text_encoders.qwen_image_mlx import QwenImageTextEncoder

    hf_tok = QwenImageTextEncoder._load_qwen2_tokenizer(tok_root)
    processor = QwenVisionLanguageProcessor(tokenizer=hf_tok)
    return QwenVisionLanguageTokenizer(processor=processor, max_length=1024, use_picture_prefix=False)


def encode_qwen_edit_prompts_mlx(
    *,
    vl_encoder: QwenVisionLanguageEncoder,
    vl_tokenizer: QwenVisionLanguageTokenizer,
    ctx: Any,
    prompt: str,
    negative_prompt: str,
    source: Image.Image,
    vl_width: int,
    vl_height: int,
) -> tuple[Any, Any, Any, Any]:
    """返回 ``(pos_embeds, pos_mask, neg_embeds, neg_mask)``。"""
    src = source.convert("RGB").resize((vl_width, vl_height), Image.BICUBIC)
    pos_ids, pos_mask, pos_px, pos_grid = vl_tokenizer.tokenize_with_image(
        prompt, src, vl_width=vl_width, vl_height=vl_height
    )
    neg_ids, neg_mask, neg_px, neg_grid = vl_tokenizer.tokenize_with_image(
        negative_prompt or "", src, vl_width=vl_width, vl_height=vl_height
    )
    pos_embeds, pos_attn = vl_encoder(
        input_ids=pos_ids,
        attention_mask=pos_mask,
        pixel_values=pos_px,
        image_grid_thw=pos_grid,
    )
    neg_embeds, neg_attn = vl_encoder(
        input_ids=neg_ids,
        attention_mask=neg_mask,
        pixel_values=neg_px,
        image_grid_thw=neg_grid,
    )
    ctx.eval(pos_embeds, pos_attn, neg_embeds, neg_attn)
    return pos_embeds, pos_attn, neg_embeds, neg_attn

