"""
位置编码与嵌入 — RoPE 2D/3D、Timestep Embedding、Patch Embedding。

所有模型经由 RuntimeContext 创建。
"""
from __future__ import annotations

from typing import Any


class TimestepEmbedding:
    """正弦时间步嵌入 → MLP 投影。

    用于: Flux1 / Flux2 / LTX / Wan 等几乎所有扩散模型。
    """

    def __init__(self, dim: int, ctx: Any, frequency_embedding_size: int = 256):
        self.ctx = ctx
        nn = ctx
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, timesteps):
        """timesteps: [B] 或 [B, 1]"""
        ctx = self.ctx
        half = self.frequency_embedding_size // 2
        freqs = ctx.exp(
            -ctx.log(ctx.full((half,), 10000.0))
            * ctx.arange(half, dtype=timesteps.dtype) / half
        )
        args = ctx.reshape(timesteps, (-1, 1)) * ctx.reshape(freqs, (1, -1))
        embedding = ctx.concat([ctx.cos(args), ctx.sin(args)], axis=-1)
        return self.mlp(embedding)


class RoPE2D:
    """2D 旋转位置编码 (图像模型)。

    用于: Flux1 / Flux2 / Qwen / FIBO / Z-Image / SeedVR2。
    """

    def __init__(self, dim: int, ctx: Any, base: float = 10000.0):
        self.ctx = ctx
        self.dim = dim
        self.base = base

    def forward(self, height: int, width: int) -> tuple[Any, Any]:
        """返回 (cos, sin) 用于 Q/K 旋转。"""
        ctx = self.ctx
        half_dim = self.dim // 2
        freqs = ctx.exp(
            -ctx.log(ctx.full((half_dim // 2,), self.base))
            * ctx.arange(half_dim // 2, dtype=ctx.float32()) / (half_dim // 2)
        )

        # 2D grid
        h_pos = ctx.arange(height, dtype=ctx.float32())
        w_pos = ctx.arange(width, dtype=ctx.float32())
        h_grid, w_grid = ctx.meshgrid(h_pos, w_pos)
        h_grid_flat = ctx.reshape(h_grid, (-1, 1))
        w_grid_flat = ctx.reshape(w_grid, (-1, 1))

        h_freqs = ctx.reshape(h_grid_flat * freqs, (1, 1, height * width, half_dim // 2))
        w_freqs = ctx.reshape(w_grid_flat * freqs, (1, 1, height * width, half_dim // 2))

        freqs_concat = ctx.concat([h_freqs, w_freqs], axis=-1)
        cos = ctx.cos(freqs_concat)
        sin = ctx.sin(freqs_concat)
        return cos, sin


class RoPE3D:
    """3D 旋转位置编码 (视频模型)。

    用于: LTX / Wan / CogVideoX。沿 T, H, W 三轴编码。
    """

    def __init__(self, dim: int, ctx: Any, base: float = 10000.0,
                 temporal_dim: int | None = None):
        self.ctx = ctx
        self.dim = dim
        self.base = base
        self.temporal_dim = temporal_dim or dim // 3

    def forward(self, num_frames: int, height: int, width: int) -> tuple[Any, Any]:
        """返回 (cos, sin) 用于 3D Q/K 旋转。"""
        ctx = self.ctx
        spatial_dim = (self.dim - self.temporal_dim) // 2

        # 时间频率
        t_freqs = ctx.exp(
            -ctx.log(ctx.full((self.temporal_dim // 2,), self.base))
            * ctx.arange(self.temporal_dim // 2, dtype=ctx.float32()) / (self.temporal_dim // 2)
        )
        # 空间频率
        s_freqs = ctx.exp(
            -ctx.log(ctx.full((spatial_dim // 2,), self.base))
            * ctx.arange(spatial_dim // 2, dtype=ctx.float32()) / (spatial_dim // 2)
        )

        t_pos = ctx.arange(num_frames, dtype=ctx.float32())
        h_pos = ctx.arange(height, dtype=ctx.float32())
        w_pos = ctx.arange(width, dtype=ctx.float32())

        t_grid, h_grid, w_grid = ctx.meshgrid3d(t_pos, h_pos, w_pos)
        total = num_frames * height * width

        t_flat = ctx.reshape(t_grid, (total, 1))
        h_flat = ctx.reshape(h_grid, (total, 1))
        w_flat = ctx.reshape(w_grid, (total, 1))

        t_freq = ctx.reshape(t_flat * t_freqs, (1, 1, total, self.temporal_dim // 2))
        h_freq = ctx.reshape(h_flat * s_freqs, (1, 1, total, spatial_dim // 2))
        w_freq = ctx.reshape(w_flat * s_freqs, (1, 1, total, spatial_dim // 2))

        freqs = ctx.concat([t_freq, h_freq, w_freq], axis=-1)
        cos = ctx.cos(freqs)
        sin = ctx.sin(freqs)
        return cos, sin


class PatchEmbed2D:
    """2D Patch Embedding (图像模型)。

    用于: Flux1 / Flux2 / Qwen / FIBO / Z-Image VAE latent → 展平为 token 序列。
    """

    def __init__(self, in_channels: int, dim: int, patch_size: int = 1, ctx: Any = None):
        self.ctx = ctx
        nn = ctx
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size,
                              stride=patch_size, bias=True)

    def forward(self, x) -> Any:
        ctx = self.ctx
        x = self.proj(x)  # [B, C, H, W] → [B, dim, H/ps, W/ps]
        B, C, H, W = x.shape
        x = ctx.permute(x, (0, 2, 3, 1))  # [B, H, W, C]
        x = ctx.reshape(x, (B, H * W, C))
        return x

    def __call__(self, x):
        return self.forward(x)


class PatchEmbed3D:
    """3D Patch Embedding (视频模型)。

    用于: LTX / Wan / CogVideoX VAE latent → 展平为时空 token 序列。
    """

    def __init__(self, in_channels: int, dim: int,
                 patch_size: tuple = (1, 2, 2), ctx: Any = None):
        self.ctx = ctx
        nn = ctx
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, dim, kernel_size=patch_size,
                              stride=patch_size, bias=True)

    def forward(self, x) -> Any:
        ctx = self.ctx
        x = self.proj(x)  # [B, C, T, H, W] → [B, dim, T/pt, H/ph, W/pw]
        B, C, T, H, W = x.shape
        total_tokens = T * H * W
        x = ctx.permute(x, (0, 2, 3, 4, 1))  # [B, T, H, W, C]
        x = ctx.reshape(x, (B, total_tokens, C))
        return x

    def __call__(self, x):
        return self.forward(x)
