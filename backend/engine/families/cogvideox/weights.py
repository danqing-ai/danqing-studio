"""CogVideoX DiT 权重映射 — diffusers / 单文件导出 → DanQing `CogVideoXTransformer3D` 扁平键。"""
from __future__ import annotations


def remap_cogvideox_weights(weights: dict) -> dict:
    """将 HuggingFace / diffusers 风格 checkpoint 键对齐到本仓库模块。

    常见前缀：
    - ``transformer.``（自包含权重目录外的顶层前缀）
    - ``model.transformer.``（部分合并导出）
    - ``module.``（部分训练导出 / DataParallel）
    """
    import importlib

    mx = importlib.import_module("mlx.core")

    def _patch_embed_conv2d_weight_torch_to_mlx(w):
        """diffusers Conv2d ``[O, I, kH, kW]`` → MLX ``nn.Conv2d`` ``[O, kH, kW, I]`` (NHWC kernel)."""
        if not isinstance(w, mx.array) or w.ndim != 4:
            return w
        return mx.transpose(w, (0, 2, 3, 1))

    remapped: dict = {}
    for key, tensor in weights.items():
        new_key = key
        for prefix in ("transformer.", "model.transformer.", "module."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
                break
        new_key = new_key.replace(".default.", ".")
        if ".lora_" in new_key or "lora_" in new_key:
            continue
        if new_key == "patch_embed.proj.weight":
            tensor = _patch_embed_conv2d_weight_torch_to_mlx(tensor)
        remapped[new_key] = tensor

    return remapped
