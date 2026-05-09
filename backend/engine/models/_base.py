"""
模型基类 — 所有图像/视频 Transformer 的公共接口。

同时提供:
- TransformerBase: 当前使用的简单基类（向后兼容）
- ImageTransformer: 抽象基类（未来迁移目标）
"""
from abc import ABC, abstractmethod
from typing import Any


class TransformerBase:
    """简单基类 — 当前所有模型继承此类。

    Hook 方法（默认空实现，子类选择性覆盖）：
    - after_load_weights()  → LoRA 权重合并
    - prepare_conditioning() → ControlNet / 视觉编码器 / 特有预处理
    - before_denoise()       → ControlNet 条件注入、latent 修改
    - step_callback()        → 每步日志 / 条件随时间变化的注入
    """

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    # ── Hook: 权重加载后 ──────────────────────────────────
    def after_load_weights(self, bundle_root=None):
        """权重加载完成后调用。LoRA / Adapter 在此合并。"""
        pass

    # ── Hook: 去噪前准备 ──────────────────────────────────
    def prepare_conditioning(self, request, bundle_root=None) -> dict:
        """处理模型特有的条件输入（ControlNet 图像编码等）。
        返回额外 cond dict，将传入 before_denoise 和 forward。
        """
        return {}

    # ── Hook: 去噪循环前 ──────────────────────────────────
    def before_denoise(self, latents, timesteps, sigmas, **cond):
        """去噪循环开始前调用。可修改 latents 或注入 ControlNet 信号。"""
        return latents, cond

    # ── Hook: 每步回调 ──────────────────────────────────
    def step_callback(self, step_idx: int, latents, noise_pred):
        """每步去噪后调用。默认空实现。"""
        pass

    def parameters(self):
        """默认参数列表（通过 _param_map）。"""
        if not hasattr(self, '_param_map'):
            self._build_param_map()
        return list(self._param_map.items())

    def load_weights(self, weights: list[tuple[str, Any]], strict: bool = False):
        """默认权重加载（通过 _param_map）。"""
        if not hasattr(self, '_param_map'):
            self._build_param_map()
        loaded = []
        skipped = []
        for key, tensor in weights:
            if key in self._param_map:
                param = self._param_map[key]
                if param.shape == tensor.shape:
                    param[:] = tensor
                    loaded.append(key)
                else:
                    skipped.append(f"{key} shape_mismatch: {param.shape} vs {tensor.shape}")
            else:
                skipped.append(key)
        if strict and skipped:
            raise ValueError(f"Unloaded keys: {skipped[:10]}...")
        return loaded, skipped

    def _build_param_map(self):
        """构建参数映射（默认实现，子类可覆盖）。"""
        if hasattr(self, '_param_map'):
            self._param_map.clear()
        else:
            self._param_map = {}
        _collect_params(self, "", self._param_map)


def _collect_params(obj, prefix: str, result: dict):
    """递归收集 nn.Module 参数。"""
    if hasattr(obj, 'parameters') and callable(obj.parameters):
        try:
            for pname, ptensor in obj.parameters().items():
                result[f"{prefix}.{pname}" if prefix else pname] = ptensor
            return
        except Exception:
            pass
    for attr_name in sorted(dir(obj)):
        if attr_name.startswith('_') or attr_name in ('ctx', 'config', 'freqs_cis', '_param_map'):
            continue
        try:
            attr = getattr(obj, attr_name)
        except Exception:
            continue
        if attr is None or isinstance(attr, (int, float, str, bool, type)):
            continue
        new_prefix = f"{prefix}.{attr_name}" if prefix else attr_name
        if hasattr(attr, 'parameters') and callable(attr.parameters):
            _collect_params(attr, new_prefix, result)
        elif isinstance(attr, (list, tuple)):
            for i, item in enumerate(attr):
                _collect_params(item, f"{new_prefix}.{i}", result)
        elif hasattr(attr, '__dict__') and not isinstance(attr, type):
            _collect_params(attr, new_prefix, result)


class ImageTransformer(ABC):
    """抽象基类 — Pipeline 通过此接口与模型交互（未来迁移目标）。

    新增模型建议继承此类，实现标准的 forward 接口。
    """

    @abstractmethod
    def forward(self, latents: Any, timestep: Any,
                txt_embeds: Any = None, sigmas: Any = None,
                **kwargs) -> Any:
        """前向传播。

        Pipeline 统一调用: model(latents, t, txt_embeds=..., sigmas=...)
        所有 timestep 转换、position IDs 生成等由模型自己处理。

        Returns:
            与 latents 同形状 [B, C, H, W]
        """
        ...

    def prepare_inputs(self, latents: Any, timestep: Any,
                       sigmas: Any = None) -> tuple[Any, dict]:
        """可选：预处理输入。默认直接返回。"""
        return latents, {}

    def postprocess_output(self, output: Any) -> Any:
        """可选：后处理输出。默认直接返回。"""
        return output

    @property
    @abstractmethod
    def config(self) -> Any:
        """模型配置。"""
        ...
