import mlx.core as mx
from mlx import nn

from .qwen_image_res_block_3d_mlx import QwenImageResBlock3D
from .qwen_image_resample_3d_mlx import QwenImageResample3D


class QwenImageUpBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int = 2, upsample_mode: str = None):
        super().__init__()
        self.resnets = []
        current_dim = in_channels
        for _ in range(num_res_blocks + 1):
            self.resnets.append(QwenImageResBlock3D(current_dim, out_channels))
            current_dim = out_channels

        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = [QwenImageResample3D(out_channels, mode=upsample_mode)]

    def __call__(self, x: mx.array) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x
