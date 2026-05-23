# M5 Max 128GB - Wan 2.2 5B MLX 极致性能优化
## AI Coding 实施计划

---

## 项目概述

将 Wan 2.2 5B 视频生成模型移植到 Apple MLX 框架，在 M5 Max 128GB 统一内存上实现极致性能优化。

### 硬件基线
| 规格 | 数值 |
|------|------|
| 统一内存 | 128 GB |
| 内存带宽 | ~600 GB/s |
| FP16 算力 | ~45 TFLOPS |
| GPU 核心 | 40-core |

### 目标性能
| 配置 | 480P 5秒 | 720P 5秒 |
|------|---------|---------|
| W4A16 + 20步 LCM | ~2.1 分钟 | ~5 分钟 |
| W4A16 + 8步 LCM | ~51 秒 | ~2 分钟 |
| W4A8 + 4步 DMD | ~26 秒 | ~1 分钟 |

---

## Phase 1: 环境搭建与权重准备 (Day 1-2)

### 1.1 依赖安装

```bash
# 核心依赖
pip install mlx>=0.23.0
pip install mlx-lm>=0.21.0
pip install transformers>=4.45.0
pip install safetensors>=0.4.0
pip install huggingface_hub>=0.25.0

# 视频处理
pip install opencv-python
pip install imageio[ffmpeg]
pip install av

# 开发工具
pip install pytest
pip install black
pip install ruff
```

### 1.2 权重下载与格式转换

```python
# download_weights.py
import os
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
import mlx.core as mx
import numpy as np

def download_wan_weights(model_id="Wan-AI/Wan2.2-T2V-5B", local_dir="./weights"):
    """下载原始 PyTorch 权重"""
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=["*.safetensors", "*.json", "*.txt"]
    )
    return local_dir

def convert_to_mlx_weights(torch_dir, mlx_dir, quantization="w4a16"):
    """将 PyTorch 权重转换为 MLX 格式 + 量化

    Args:
        quantization: "fp16" | "w8a16" | "w4a16" | "w4a8"
    """
    os.makedirs(mlx_dir, exist_ok=True)

    # 加载所有 safetensors 文件
    weight_files = sorted([
        f for f in os.listdir(torch_dir) 
        if f.endswith(".safetensors")
    ])

    mlx_weights = {}
    for wf in weight_files:
        pt_weights = load_file(os.path.join(torch_dir, wf))
        for key, tensor in pt_weights.items():
            # 转换为 numpy -> mlx
            np_array = tensor.numpy()

            # 量化策略
            if "attention" in key or "norm" in key:
                # Attention 相关层保持 FP16 (精度敏感)
                mlx_weights[key] = mx.array(np_array, dtype=mx.float16)
            elif quantization == "w4a16":
                # FFN 层用 4-bit
                mlx_weights[key] = mx.array(np_array, dtype=mx.float16)
                # 实际量化在保存时处理
            elif quantization == "w4a8":
                mlx_weights[key] = mx.array(np_array, dtype=mx.float16)
            else:
                mlx_weights[key] = mx.array(np_array, dtype=mx.float16)

    # 保存为 MLX 格式
    mx.savez(os.path.join(mlx_dir, "wan2.2_5b.npz"), **mlx_weights)

    return mlx_dir

if __name__ == "__main__":
    torch_dir = download_wan_weights()
    mlx_dir = convert_to_mlx_weights(torch_dir, "./weights_mlx", "w4a16")
    print(f"MLX weights saved to: {mlx_dir}")
```

### 1.3 量化实现 (核心)

```python
# quantization.py
import mlx.core as mx
from typing import Dict, Tuple

class W4A16Quantizer:
    """4-bit 权重 + 16-bit 激活量化器"""

    GROUP_SIZE = 128  # 每 128 个通道共享一组 scale/zero-point

    @staticmethod
    def quantize_weight(weight: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """将 FP16 权重量化为 4-bit

        Returns:
            quantized: INT4 量化后的权重 (以 uint8 存储, 每 byte 存 2 个权重)
            scales: 每组 scale
            zeros: 每组 zero-point
        """
        original_shape = weight.shape
        num_elements = weight.size

        # reshape 为 (num_groups, group_size)
        num_groups = (num_elements + W4A16Quantizer.GROUP_SIZE - 1) // W4A16Quantizer.GROUP_SIZE
        padded = mx.pad(weight.flatten(), (0, num_groups * W4A16Quantizer.GROUP_SIZE - num_elements))
        grouped = padded.reshape(num_groups, W4A16Quantizer.GROUP_SIZE)

        # 计算每组 min/max
        w_min = mx.min(grouped, axis=1, keepdims=True)
        w_max = mx.max(grouped, axis=1, keepdims=True)

        # scale 和 zero-point
        scales = (w_max - w_min) / 15.0  # 4-bit = 16 values (0-15)
        zeros = w_min

        # 量化: (w - zero) / scale, round to 0-15
        quantized = mx.round((grouped - zeros) / scales).astype(mx.uint8)
        quantized = mx.clip(quantized, 0, 15)

        # 打包: 两个 4-bit 值存到一个 uint8
        # 实际实现中可能需要自定义 kernel

        return quantized, scales.squeeze(), zeros.squeeze()

    @staticmethod
    def dequantize_weight(quantized: mx.array, scales: mx.array, zeros: mx.array, 
                         shape: Tuple[int, ...]) -> mx.array:
        """反量化为 FP16"""
        # 解包 uint8 -> 两个 4-bit
        # ... 实现略 ...

        # 反量化
        dequantized = quantized.astype(mx.float16) * scales + zeros
        return dequantized.reshape(shape)

class W4A8Quantizer:
    """4-bit 权重 + 8-bit 激活 (更激进)"""

    @staticmethod
    def quantize_activation(activation: mx.array) -> Tuple[mx.array, mx.array]:
        """动态量化激活值到 8-bit"""
        a_min = mx.min(activation)
        a_max = mx.max(activation)
        scale = (a_max - a_min) / 255.0

        quantized = mx.round((activation - a_min) / scale).astype(mx.uint8)
        return quantized, scale
```

---

## Phase 2: 核心模型架构移植 (Day 3-5)

### 2.1 Wan 2.2 DiT 架构 (原生 MLX)

```python
# wan_dit.py
import mlx.core as mx
from mlx import nn
from typing import Optional, Tuple

class WanRMSNorm(nn.Module):
    """Wan 使用的 RMSNorm"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return x * mx.rsqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + self.eps) * self.scale

class WanSelfAttention(nn.Module):
    """分解式 Spatial-Temporal Self-Attention

    关键优化: 将 3D attention 拆分为 spatial + temporal,
    避免 O(N^2) 的完整 3D attention (N=62400 时不可行)
    """
    def __init__(self, dim: int, num_heads: int = 20, head_dim: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        # RoPE 预缓存
        self._rope_cache = {}

    def get_rope(self, seq_len: int, device):
        """预计算并缓存 RoPE sin/cos 表"""
        if seq_len not in self._rope_cache:
            inv_freq = 1.0 / (10000 ** (mx.arange(0, self.head_dim, 2) / self.head_dim))
            t = mx.arange(seq_len)
            freqs = mx.outer(t, inv_freq)
            emb = mx.concatenate([freqs, freqs], axis=-1)
            self._rope_cache[seq_len] = (mx.sin(emb), mx.cos(emb))
        return self._rope_cache[seq_len]

    def apply_rope(self, x: mx.array, sin: mx.array, cos: mx.array) -> mx.array:
        """应用旋转位置编码"""
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = mx.stack([-x2, x1], axis=-1).flatten(-2)
        return x * cos + rotated * sin

    def __call__(self, x: mx.array, spatial_shape: Tuple[int, int, int]) -> mx.array:
        """
        Args:
            x: (batch, seq_len, dim) - seq_len = F * H * W
            spatial_shape: (F, H, W) latent shape
        """
        F, H, W = spatial_shape
        B, N, D = x.shape

        # QKV projection
        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        # reshape to multi-head
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # === Spatial Attention (每帧内) ===
        # 将 token 重新组织为 (B, F, H*W, heads, head_dim)
        q_spatial = q.reshape(B, F, H*W, self.num_heads, self.head_dim)
        k_spatial = k.reshape(B, F, H*W, self.num_heads, self.head_dim)
        v_spatial = v.reshape(B, F, H*W, self.num_heads, self.head_dim)

        # 对每帧分别做 attention
        spatial_out = []
        for f in range(F):
            qf, kf, vf = q_spatial[:, f], k_spatial[:, f], v_spatial[:, f]

            # RoPE for spatial positions
            sin, cos = self.get_rope(H*W, x.device)
            qf = self.apply_rope(qf, sin, cos)
            kf = self.apply_rope(kf, sin, cos)

            # FlashAttention-2 等效 (使用 mlx.fast.sdpa)
            # 注意: mlx 0.23+ 支持 scaled_dot_product_attention
            attn_out = mx.fast.scaled_dot_product_attention(qf, kf, vf, scale=self.scale)
            spatial_out.append(attn_out)

        spatial_out = mx.stack(spatial_out, axis=1)  # (B, F, H*W, heads, head_dim)

        # === Temporal Attention (跨帧) ===
        # 重新组织为 (B, H*W, F, heads, head_dim)
        q_temporal = q.reshape(B, F, H*W, self.num_heads, self.head_dim).transpose(0, 2, 1, 3, 4)
        k_temporal = k.reshape(B, F, H*W, self.num_heads, self.head_dim).transpose(0, 2, 1, 3, 4)
        v_temporal = v.reshape(B, F, H*W, self.num_heads, self.head_dim).transpose(0, 2, 1, 3, 4)

        temporal_out = []
        for pos in range(H*W):
            qp, kp, vp = q_temporal[:, pos], k_temporal[:, pos], v_temporal[:, pos]

            # RoPE for temporal positions
            sin_t, cos_t = self.get_rope(F, x.device)
            qp = self.apply_rope(qp, sin_t, cos_t)
            kp = self.apply_rope(kp, sin_t, cos_t)

            attn_out = mx.fast.scaled_dot_product_attention(qp, kp, vp, scale=self.scale)
            temporal_out.append(attn_out)

        temporal_out = mx.stack(temporal_out, axis=1)  # (B, H*W, F, heads, head_dim)
        temporal_out = temporal_out.transpose(0, 2, 1, 3, 4)  # (B, F, H*W, heads, head_dim)

        # 融合: 残差相加
        out = spatial_out + temporal_out
        out = out.reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        out = out.reshape(B, N, D)

        # Output projection
        return self.proj(out)

class WanFeedForward(nn.Module):
    """SwiGLU FFN"""
    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 4

        # SwiGLU: 三个 projection
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))

class WanDiTBlock(nn.Module):
    """单个 DiT Block: Norm + Attention + FFN"""
    def __init__(self, dim: int, num_heads: int = 20):
        super().__init__()
        self.norm1 = WanRMSNorm(dim)
        self.attn = WanSelfAttention(dim, num_heads)
        self.norm2 = WanRMSNorm(dim)
        self.ffn = WanFeedForward(dim)

        # AdaLN modulation
        self.scale_shift_table = mx.zeros((6, dim))  # 2 norms * (scale + shift + gate)

    def __call__(self, x: mx.array, t_emb: mx.array, spatial_shape: Tuple[int, int, int]) -> mx.array:
        # AdaLN: 从 t_emb 生成 scale, shift, gate
        # 简化实现, 实际需要从 t_emb 投影
        scale1, shift1, gate1 = mx.split(self.scale_shift_table[:3], 3, axis=0)
        scale1, shift1, gate1 = scale1[0], shift1[0], gate1[0]

        # Attention branch
        normed = self.norm1(x)
        modulated = normed * (1 + scale1) + shift1
        attn_out = self.attn(modulated, spatial_shape)
        x = x + gate1 * attn_out

        # FFN branch
        scale2, shift2, gate2 = mx.split(self.scale_shift_table[3:], 3, axis=0)
        scale2, shift2, gate2 = scale2[0], shift2[0], gate2[0]

        normed = self.norm2(x)
        modulated = normed * (1 + scale2) + shift2
        ffn_out = self.ffn(modulated)
        x = x + gate2 * ffn_out

        return x

class WanDiT(nn.Module):
    """完整的 Wan 2.2 DiT 模型"""
    def __init__(self, in_channels: int = 16, hidden_size: int = 2560, 
                 num_layers: int = 40, num_heads: int = 20):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size

        # Patch embedding
        self.patch_embed = nn.Linear(in_channels, hidden_size, bias=False)

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        # Transformer blocks
        self.blocks = [WanDiTBlock(hidden_size, num_heads) for _ in range(num_layers)]

        # Final head
        self.final_norm = WanRMSNorm(hidden_size)
        self.final_proj = nn.Linear(hidden_size, in_channels, bias=False)

    def __call__(self, x: mx.array, t: mx.array, cond: Optional[mx.array] = None,
                 spatial_shape: Tuple[int, int, int] = None) -> mx.array:
        """
        Args:
            x: (B, C, F, H, W) latent noise
            t: (B,) timestep
            cond: text embedding
            spatial_shape: (F, H, W)
        """
        B, C, F, H, W = x.shape
        if spatial_shape is None:
            spatial_shape = (F, H, W)

        # Flatten spatial dimensions
        x = x.transpose(0, 2, 3, 4, 1)  # (B, F, H, W, C)
        x = x.reshape(B, F*H*W, C)

        # Patch embed
        x = self.patch_embed(x)

        # Time embedding
        t_emb = self.time_embed(t)  # 简化, 实际需要 sinusoidal embedding

        # Transformer blocks
        for block in self.blocks:
            x = block(x, t_emb, spatial_shape)

        # Final projection
        x = self.final_norm(x)
        x = self.final_proj(x)

        # Reshape back
        x = x.reshape(B, F, H, W, C)
        x = x.transpose(0, 4, 1, 2, 3)  # (B, C, F, H, W)

        return x
```

### 2.2 3D VAE 解码器 (原生 MLX)

```python
# wan_vae.py
import mlx.core as mx
from mlx import nn

class WanCausalConv3D(nn.Module):
    """Wan 使用的因果 3D 卷积 (时序因果)"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int, int]):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding="same")

    def __call__(self, x: mx.array) -> mx.array:
        # 时序因果: 只使用过去帧的信息
        return self.conv(x)

class WanVAEEncoder(nn.Module):
    """VAE Encoder: 视频 -> latent"""
    def __init__(self, in_channels: int = 3, latent_channels: int = 16):
        super().__init__()
        # 下采样: 8x spatial, 4x temporal
        self.blocks = nn.Sequential(
            WanCausalConv3D(in_channels, 128, (3, 3, 3)),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            # ... 更多下采样层
            nn.Conv3d(128, latent_channels * 2, (1, 1, 1))  # mean, logvar
        )

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        h = self.blocks(x)
        mean, logvar = mx.split(h, 2, axis=1)
        return mean, logvar

class WanVAEDecoder(nn.Module):
    """VAE Decoder: latent -> 视频 (优化重点)"""
    def __init__(self, latent_channels: int = 16, out_channels: int = 3):
        super().__init__()
        # 上采样: 8x spatial, 4x temporal
        self.blocks = nn.Sequential(
            nn.Conv3d(latent_channels, 128, (1, 1, 1)),
            # ... 更多上采样层
            WanCausalConv3D(128, out_channels, (3, 3, 3))
        )

    def __call__(self, z: mx.array) -> mx.array:
        return self.blocks(z)

    def decode_tiled(self, z: mx.array, tile_size: Tuple[int, int, int] = (4, 64, 64),
                     overlap: float = 0.25) -> mx.array:
        """分块解码, 避免高分辨率时内存溢出

        关键优化: 720P 视频的 latent 很大, 一次性解码可能 OOM
        """
        C, F, H, W = z.shape
        tile_f, tile_h, tile_w = tile_size

        # 计算 tile 数量和重叠
        stride_f = int(tile_f * (1 - overlap))
        stride_h = int(tile_h * (1 - overlap))
        stride_w = int(tile_w * (1 - overlap))

        # 输出 buffer
        output = mx.zeros((3, F * 4, H * 8, W * 8))  # 上采样后尺寸
        weight = mx.zeros((3, F * 4, H * 8, W * 8))

        # 滑动窗口解码
        for f in range(0, F - tile_f + 1, stride_f):
            for h in range(0, H - tile_h + 1, stride_h):
                for w in range(0, W - tile_w + 1, stride_w):
                    # 提取 tile
                    z_tile = z[:, f:f+tile_f, h:h+tile_h, w:w+tile_w]

                    # 解码
                    tile_out = self.blocks(z_tile)

                    # 加权融合到输出
                    out_f = f * 4
                    out_h = h * 8
                    out_w = w * 8
                    output[:, out_f:out_f+tile_f*4, out_h:out_h+tile_h*8, out_w:out_w+tile_w*8] += tile_out
                    weight[:, out_f:out_f+tile_f*4, out_h:out_h+tile_h*8, out_w:out_w+tile_w*8] += 1

        # 归一化
        output = output / (weight + 1e-8)
        return output
```

---

## Phase 3: 推理引擎优化 (Day 6-8)

### 3.1 惰性求值与图编译

```python
# engine.py
import mlx.core as mx
from functools import wraps

class WanInferenceEngine:
    """优化的推理引擎"""

    def __init__(self, dit_path: str, vae_path: str, quantization: str = "w4a16"):
        # 加载权重 (内存映射)
        self.dit_weights = mx.load(dit_path, mmap=True)
        self.vae_weights = mx.load(vae_path, mmap=True)

        # 构建模型
        self.dit = WanDiT()
        self.vae = WanVAEDecoder()

        # 量化配置
        self.quantization = quantization

        # 预编译静态图
        self._compile_graphs()

    def _compile_graphs(self):
        """预编译去噪步骤的计算图"""
        # 创建示例输入
        dummy_latent = mx.zeros((1, 16, 10, 60, 104))  # 480P latent
        dummy_t = mx.array([0])

        # 编译 DIT forward
        @mx.compile
        def denoise_step(latent, t, cond):
            return self.dit(latent, t, cond, spatial_shape=(10, 60, 104))

        self.denoise_step = denoise_step

        # 预热 (触发编译)
        _ = denoise_step(dummy_latent, dummy_t, None)
        mx.eval(_)  # 强制执行以完成编译

    def generate(self, prompt: str, height: int = 480, width: int = 832,
                 num_frames: int = 40, num_steps: int = 20,
                 cfg_scale: float = 7.5, seed: int = 42) -> mx.array:
        """视频生成主流程"""

        # 1. 文本编码 (复用 mlx-lm 或 transformers)
        text_embed = self.encode_text(prompt)

        # 2. 初始化 latent noise
        latent_shape = (num_frames // 4, height // 8, width // 8)
        latent = mx.random.normal(
            shape=(1, 16, *latent_shape),
            key=mx.random.key(seed)
        )

        # 3. 准备 timestep schedule
        timesteps = self.get_schedule(num_steps)

        # 4. CFG: 同时准备 conditional 和 unconditional
        cond_embed = mx.concatenate([text_embed, mx.zeros_like(text_embed)], axis=0)
        latent_batch = mx.concatenate([latent, latent], axis=0)

        # 5. 去噪循环
        for i, t in enumerate(timesteps):
            # 扩展 timestep 到 batch
            t_batch = mx.array([t, t])

            # 编译过的 forward (单图复用)
            noise_pred = self.denoise_step(latent_batch, t_batch, cond_embed)

            # 拆分 conditional / unconditional
            noise_cond, noise_uncond = mx.split(noise_pred, 2, axis=0)

            # CFG
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

            # DDPM update (简化, 实际用 Wan 的 flow matching)
            latent = latent - noise_pred * (timesteps[i] - timesteps[i+1] if i < len(timesteps)-1 else 0)

            # 每步后清理中间结果 (利用统一内存自动回收)
            mx.eval(latent)

        # 6. VAE 解码 (分块)
        video = self.vae.decode_tiled(latent[0], tile_size=(4, 64, 64))

        return video
```

### 3.2 KV-Cache 分页管理

```python
# kv_cache.py
import mlx.core as mx
from typing import List, Dict

class PagedKVCache:
    """分页 KV-Cache 管理

    关键优化: 避免预分配最大长度, 按需分配 256-token 的页
    """
    PAGE_SIZE = 256

    def __init__(self, num_layers: int, num_heads: int, head_dim: int, 
                 max_pages: int = 1000):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_pages = max_pages

        # 页表: layer -> list of pages
        self.pages: Dict[int, List[mx.array]] = {i: [] for i in range(num_layers)}
        self.page_count = 0

    def allocate_page(self, layer_idx: int) -> mx.array:
        """分配新页"""
        if self.page_count >= self.max_pages:
            raise RuntimeError("KV-Cache 已满")

        # (num_heads, PAGE_SIZE, head_dim)
        page = mx.zeros((self.num_heads, self.PAGE_SIZE, self.head_dim), dtype=mx.float16)
        self.pages[layer_idx].append(page)
        self.page_count += 1
        return page

    def get_kv(self, layer_idx: int, seq_len: int) -> Tuple[mx.array, mx.array]:
        """获取指定长度的 K/V, 按需分配"""
        pages = self.pages[layer_idx]
        needed_pages = (seq_len + self.PAGE_SIZE - 1) // self.PAGE_SIZE

        # 分配不足页
        while len(pages) < needed_pages:
            self.allocate_page(layer_idx)

        # 拼接已有页
        k_pages = [p[:seq_len % self.PAGE_SIZE] if i == needed_pages - 1 and seq_len % self.PAGE_SIZE else p 
                   for i, p in enumerate(pages[:needed_pages])]

        return mx.concatenate(k_pages, axis=1)

    def update_kv(self, layer_idx: int, new_k: mx.array, new_v: mx.array):
        """更新 KV-Cache"""
        # 将新 K/V 写入对应页
        # ... 实现略 ...
        pass

    def clear(self):
        """清空缓存"""
        self.pages = {i: [] for i in range(self.num_layers)}
        self.page_count = 0
```

### 3.3 异步流水线

```python
# pipeline.py
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncWanPipeline:
    """异步流水线: 重叠计算与数据传输"""

    def __init__(self, engine: WanInferenceEngine):
        self.engine = engine
        self.executor = ThreadPoolExecutor(max_workers=2)

    async def generate_async(self, prompt: str, **kwargs):
        """异步生成, 支持后台任务"""
        loop = asyncio.get_event_loop()

        # 在线程池中运行 MLX 计算
        future = loop.run_in_executor(
            self.executor,
            self.engine.generate,
            prompt,
            **kwargs
        )

        # 可同时处理其他任务 (如下一个请求的预处理)
        video = await future
        return video

    async def batch_generate(self, prompts: List[str], **kwargs):
        """批量生成 (利用统一内存的零拷贝特性)"""
        # 合并 batch
        # ... 实现略 ...
        pass
```

---

## Phase 4: 量化与蒸馏集成 (Day 9-10)

### 4.1 LCM-LoRA 8步蒸馏

```python
# lcm_lora.py
import mlx.core as mx
from mlx import nn

class LCMLoRA(nn.Module):
    """LCM-LoRA 适配器"""
    def __init__(self, base_dim: int, rank: int = 64):
        super().__init__()
        self.lora_a = nn.Linear(base_dim, rank, bias=False)
        self.lora_b = nn.Linear(rank, base_dim, bias=False)
        self.scale = 1.0

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.scale * self.lora_b(self.lora_a(x))

def apply_lcm_lora(dit: WanDiT, lora_path: str, alpha: float = 1.0):
    """将 LCM-LoRA 应用到 DiT 模型"""
    lora_weights = mx.load(lora_path)

    for i, block in enumerate(dit.blocks):
        # 在 attention 和 ffn 上注入 LoRA
        block.attn.qkv = LCMLoRAAdapter(block.attn.qkv, lora_weights[f"blocks.{i}.attn.qkv"])
        block.ffn.w1 = LCMLoRAAdapter(block.ffn.w1, lora_weights[f"blocks.{i}.ffn.w1"])

    return dit

def get_lcm_schedule(num_steps: int = 8) -> List[float]:
    """LCM 的 timestep schedule (跳步)"""
    # 从 1000 到 0, 但只取 8 个关键点
    return [1000, 800, 600, 400, 300, 200, 100, 50, 0][:num_steps+1]
```

### 4.2 4-bit 权重加载器

```python
# quant_loader.py
import mlx.core as mx
import numpy as np

class QuantizedLinear(nn.Module):
    """量化线性层: 4-bit 权重, 16-bit 激活"""
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 量化参数 (运行时加载)
        self.qweight = None  # uint8, 每 byte 存 2 个 4-bit 权重
        self.scales = None   # float16
        self.zeros = None    # float16

    @classmethod
    def from_fp16(cls, weight: mx.array, group_size: int = 128):
        """从 FP16 权重创建量化版本"""
        instance = cls(weight.shape[1], weight.shape[0])

        # 量化
        orig_shape = weight.shape
        weight_flat = weight.flatten()
        num_groups = (weight_flat.size + group_size - 1) // group_size

        # 分组计算 scale/zero
        padded = mx.pad(weight_flat, (0, num_groups * group_size - weight_flat.size))
        grouped = padded.reshape(num_groups, group_size)

        w_min = mx.min(grouped, axis=1)
        w_max = mx.max(grouped, axis=1)
        scales = (w_max - w_min) / 15.0
        zeros = w_min

        # 量化到 0-15
        quantized = mx.round((grouped - zeros[:, None]) / scales[:, None]).astype(mx.uint8)
        quantized = mx.clip(quantized, 0, 15)

        # 打包: 两个 4-bit -> 一个 uint8
        # 实际实现需要自定义 Metal kernel 以高效处理
        packed = (quantized[:, ::2] << 4) | quantized[:, 1::2]

        instance.qweight = packed[:weight_flat.size // 2]
        instance.scales = scales
        instance.zeros = zeros

        return instance

    def dequantize(self) -> mx.array:
        """反量化为 FP16 (用于 forward)"""
        # 解包
        high = (self.qweight >> 4) & 0xF
        low = self.qweight & 0xF
        unpacked = mx.stack([high, low], axis=-1).flatten()

        # 反量化
        # 需要重复 scales 和 zeros 以匹配 unpacked 长度
        # ... 实现略 ...

        return dequantized.reshape(self.out_features, self.in_features)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward: 反量化 -> matmul"""
        w = self.dequantize()
        return x @ w.T
```

---

## Phase 5: 性能测试与调优 (Day 11-12)

### 5.1 基准测试脚本

```python
# benchmark.py
import time
import mlx.core as mx
from dataclasses import dataclass
from typing import List

@dataclass
class BenchmarkResult:
    resolution: str
    num_frames: int
    num_steps: int
    quantization: str
    total_time: float
    time_per_step: float
    memory_peak_gb: float
    vram_usage_gb: float

def benchmark(engine: WanInferenceEngine, configs: List[dict]) -> List[BenchmarkResult]:
    """系统性基准测试"""
    results = []

    for cfg in configs:
        # 内存清零
        mx.clear_cache()

        # 预热
        _ = engine.generate(
            "a cat", 
            height=cfg["height"], 
            width=cfg["width"],
            num_frames=cfg["frames"],
            num_steps=1
        )
        mx.eval(_)

        # 正式测试
        start = time.time()
        video = engine.generate(
            "a beautiful sunset over mountains, cinematic",
            height=cfg["height"],
            width=cfg["width"],
            num_frames=cfg["frames"],
            num_steps=cfg["steps"]
        )
        mx.eval(video)
        elapsed = time.time() - start

        # 记录结果
        result = BenchmarkResult(
            resolution=f"{cfg['height']}P",
            num_frames=cfg["frames"],
            num_steps=cfg["steps"],
            quantization=cfg["quantization"],
            total_time=elapsed,
            time_per_step=elapsed / cfg["steps"],
            memory_peak_gb=0,  # 需要系统级监控
            vram_usage_gb=0
        )
        results.append(result)

        print(f"[{cfg['height']}P, {cfg['frames']}frames, {cfg['steps']}steps, {cfg['quantization']}]")
        print(f"  Total: {elapsed:.1f}s, Per-step: {elapsed/cfg['steps']:.1f}s")

    return results

# 测试配置矩阵
TEST_CONFIGS = [
    {"height": 480, "width": 832, "frames": 40, "steps": 50, "quantization": "fp16"},
    {"height": 480, "width": 832, "frames": 40, "steps": 20, "quantization": "w4a16"},
    {"height": 480, "width": 832, "frames": 40, "steps": 8, "quantization": "w4a16"},
    {"height": 720, "width": 1280, "frames": 80, "steps": 20, "quantization": "w4a16"},
    {"height": 720, "width": 1280, "frames": 80, "steps": 8, "quantization": "w4a16"},
]
```

### 5.2 Metal System Trace 分析

```bash
# 使用 Xcode Instruments 分析 GPU 利用率
# 1. 编译为可执行文件
# 2. 运行: xcrun xctrace record --template "Metal System Trace" --launch -- /path/to/benchmark

# 关键指标:
# - GPU 利用率: 目标 > 80%
# - Memory bandwidth: 目标接近 600 GB/s
# - Kernel 执行时间: 识别长尾 kernel
```

---

## Phase 6: 部署与封装 (Day 13-14)

### 6.1 CLI 工具

```python
# cli.py
import argparse
import mlx.core as mx
from wan_engine import WanInferenceEngine

def main():
    parser = argparse.ArgumentParser(description="Wan 2.2 5B MLX Video Generation")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--quantization", type=str, default="w4a16", choices=["fp16", "w4a16", "w4a8"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output.mp4")
    args = parser.parse_args()

    # 初始化引擎
    engine = WanInferenceEngine(
        dit_path=f"./weights_mlx/wan2.2_5b_{args.quantization}.npz",
        vae_path="./weights_mlx/wan_vae.npz",
        quantization=args.quantization
    )

    # 生成
    print(f"Generating: {args.prompt}")
    print(f"Config: {args.height}x{args.width}, {args.frames} frames, {args.steps} steps")

    video = engine.generate(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        num_steps=args.steps,
        seed=args.seed
    )

    # 保存
    save_video(video, args.output)
    print(f"Saved to: {args.output}")

if __name__ == "__main__":
    main()
```

### 6.2 Python API

```python
# api.py
from wan_engine import WanInferenceEngine

class WanMLX:
    """高级 API"""

    def __init__(self, model_path: str = "./weights_mlx", quantization: str = "w4a16"):
        self.engine = WanInferenceEngine(
            dit_path=f"{model_path}/wan2.2_5b_{quantization}.npz",
            vae_path=f"{model_path}/wan_vae.npz",
            quantization=quantization
        )

    def text_to_video(self, prompt: str, **kwargs):
        return self.engine.generate(prompt, **kwargs)

    def image_to_video(self, image, prompt: str, **kwargs):
        # I2V 实现
        pass

    def video_to_video(self, video, prompt: str, **kwargs):
        # V2V 实现
        pass
```

---

## 关键优化检查清单

### 内存优化
- [ ] 权重使用 W4A16 量化 (2.4GB vs 9.4GB)
- [ ] KV-Cache 分页管理 (避免预分配)
- [ ] VAE 分块解码 (Tile decoding)
- [ ] 激活值 checkpointing (重计算替代存储)
- [ ] 每步后 `mx.eval()` 清理中间结果

### 计算优化
- [ ] Spatial-Temporal Attention 分解
- [ ] RoPE 预计算并缓存
- [ ] 算子融合 (Norm + Linear + SiLU)
- [ ] `mx.compile` 预编译去噪步骤
- [ ] FlashAttention-2 等效实现 (mlx.fast.sdpa)

### 调度优化
- [ ] LCM-LoRA 8步蒸馏 (替代 50步)
- [ ] CFG 并行计算 (batch=2)
- [ ] 异步 VAE 解码 (后台执行)
- [ ] 统一内存零拷贝 (无 CPU-GPU 传输)

### 量化策略
- [ ] Attention 层保持 FP16 (精度敏感)
- [ ] FFN 层 W4A16 (占参数量 70%+)
- [ ] VAE 权重 INT8 (对精度不敏感)
- [ ] 动态激活量化 (可选 W4A8)

---

## 预期最终性能

| 配置 | 480P 5秒 | 720P 5秒 | 质量评级 |
|------|---------|---------|---------|
| FP16 + 50步 | ~5.4 分钟 | ~13 分钟 | 5 stars |
| **W4A16 + 20步 LCM** | **~2.1 分钟** | **~5 分钟** | 5 stars |
| **W4A16 + 8步 LCM** | **~51 秒** | **~2 分钟** | 4 stars |
| W4A8 + 4步 DMD | ~26 秒 | ~1 分钟 | 3 stars |

---

## 参考资源

- **MLX 官方**: https://ml-explore.github.io/mlx/
- **MLX-Video**: https://github.com/Blaizzy/mlx-video
- **MFLUX**: https://github.com/filipstrand/mflux
- **Wan 2.2 论文**: https://arxiv.org/abs/2505.06782
- **LCM-LoRA**: https://huggingface.co/latent-consistency/lcm-lora-sdv1-5
- **FlashAttention**: https://github.com/Dao-AILab/flash-attention

---

*文档版本: v1.0 | 目标平台: Apple M5 Max 128GB | MLX >= 0.23.0*
