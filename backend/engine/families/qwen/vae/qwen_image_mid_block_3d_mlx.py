import mlx.core as mx
from mlx import nn

from .qwen_image_attention_block_3d_mlx import QwenImageAttentionBlock3D
from .qwen_image_res_block_3d_mlx import QwenImageResBlock3D


class QwenImageMidBlock3D(nn.Module):
    def __init__(self, dim: int, num_layers: int = 1):
        super().__init__()
        self.dim = dim
        resnets = [QwenImageResBlock3D(dim, dim)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(QwenImageAttentionBlock3D(dim))
            resnets.append(QwenImageResBlock3D(dim, dim))
        self.attentions = attentions
        self.resnets = resnets

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)
            x = resnet(x)
        return x
