"""CFG dispatch 策略 — 3 种模式 + 工厂函数。

| 策略                    | 触发条件                                    | 行为                     |
|-------------------------|---------------------------------------------|--------------------------|
| ``FusedCfgStrategy``    | ``hasattr(model, 'forward_cfg')`` + MLX     | 单次 forward (Z-Image)   |
| ``BatchedCfgStrategy``  | ``hasattr(model, 'predict_noise_cfg')``     | batch axis 合并 (Wan)    |
| ``DualForwardCfgStrategy`` | fallback                                  | 两次 forward + combine   |

``resolve_cfg_strategy()`` 在 Pipeline (L1) 中调用一次，结果放入 InferenceBundle。

关键：``combine_cfg_noise`` 公式**留在 model (L3)** — L2 只负责 dispatch。
"""
from __future__ import annotations

from typing import Any

from backend.engine.common.ops.cfg_batch import predict_noise_cfg_batched


# ---------------------------------------------------------------------------
# Text-related keys — DualForward 在构建 uncond forward 时需要过滤
# ---------------------------------------------------------------------------

_TEXT_RELATED_KEYS = frozenset({
    "txt_embeds", "neg_embeds", "redux_txt_embeds", "fill_static_packed",
    "txt_attn_mask", "txt_embeds_2", "txt_attn_mask_2",
    "pooled_embeds",
})


# ---------------------------------------------------------------------------
# FusedCfgStrategy
# ---------------------------------------------------------------------------

class FusedCfgStrategy:
    """单次 forward_cfg — Z-Image, FIBO 等 MLX 融合 CFG 模型。

    要求 model 实现 ``forward_cfg(latents, t, txt_embeds, neg_embeds, guidance, ...)``
    """

    # forward_cfg 从 model_kwargs 中排除的 key（它们作为独立参数传入）
    _EXCLUDE_KEYS = frozenset({
        "txt_embeds", "neg_embeds", "redux_txt_embeds", "fill_static_packed",
    })

    def predict_noise(
        self,
        model: Any,
        latents: Any,
        t: Any,
        *,
        cond_kwargs: dict[str, Any],
        uncond_kwargs: dict[str, Any] | None,
        guidance: float,
        ctx: Any = None,
        cfg_renorm: bool = False,
        cfg_renorm_min: float = 0.0,
    ) -> Any:
        txt_embeds = cond_kwargs.get("txt_embeds")
        neg_embeds = uncond_kwargs.get("txt_embeds") if uncond_kwargs else None
        cfg_kwargs = {k: v for k, v in cond_kwargs.items() if k not in self._EXCLUDE_KEYS}
        return model.forward_cfg(
            latents, t,
            txt_embeds, neg_embeds, guidance,
            cfg_renorm=cfg_renorm,
            cfg_renorm_min=cfg_renorm_min,
            **cfg_kwargs,
        )


# ---------------------------------------------------------------------------
# BatchedCfgStrategy
# ---------------------------------------------------------------------------

class BatchedCfgStrategy:
    """Batched CFG — 一次 forward 同时处理 cond + uncond（batch axis 合并）。

    要求 model 实现 ``predict_noise_cfg(latents, t, guidance=..., pos_kwargs=..., neg_kwargs=...)``
    或被 ``cfg_batch.predict_noise_cfg_batched`` 消费。

    ``text_keys`` 指定需要在 batch 维度拼接的 text 相关 key。
    """

    def __init__(self, text_keys: frozenset[str] | None = None) -> None:
        self._text_keys = text_keys

    def predict_noise(
        self,
        model: Any,
        latents: Any,
        t: Any,
        *,
        cond_kwargs: dict[str, Any],
        uncond_kwargs: dict[str, Any] | None,
        guidance: float,
        ctx: Any = None,
        cfg_renorm: bool = False,
        cfg_renorm_min: float = 0.0,
    ) -> Any:
        if uncond_kwargs is None or uncond_kwargs.get("txt_embeds") is None:
            return model(latents, t, **cond_kwargs)

        neg_kwargs = uncond_kwargs
        # 优先使用 model 自带的 predict_noise_cfg（如 Wan）
        if hasattr(model, "predict_noise_cfg"):
            return model.predict_noise_cfg(
                latents, t,
                guidance=guidance,
                pos_kwargs=cond_kwargs,
                neg_kwargs=neg_kwargs,
                cfg_renorm=cfg_renorm,
                cfg_renorm_min=cfg_renorm_min,
            )
        # 回退到通用 batched forward
        text_keys = self._text_keys
        if text_keys is None:
            text_keys = _infer_text_keys(cond_kwargs)
        return predict_noise_cfg_batched(
            model, ctx, latents, t,
            guidance=guidance,
            pos_kwargs=cond_kwargs,
            neg_kwargs=neg_kwargs,
            text_keys=text_keys,
            combine_cfg_noise=model.combine_cfg_noise,
            refine_cfg_noise=getattr(model, "refine_cfg_noise", None),
            cfg_renorm=cfg_renorm,
            cfg_renorm_min=cfg_renorm_min,
        )


# ---------------------------------------------------------------------------
# DualForwardCfgStrategy
# ---------------------------------------------------------------------------

class DualForwardCfgStrategy:
    """两次 forward（cond + uncond）后合并 — 通用 fallback。

    当 ``uncond_kwargs`` 为 None 时（无负向 prompt / 不需要 CFG），
    退化为单次 forward 返回 cond 结果。
    """

    def predict_noise(
        self,
        model: Any,
        latents: Any,
        t: Any,
        *,
        cond_kwargs: dict[str, Any],
        uncond_kwargs: dict[str, Any] | None,
        guidance: float,
        ctx: Any = None,
        cfg_renorm: bool = False,
        cfg_renorm_min: float = 0.0,
    ) -> Any:
        noise_cond = model(latents, t, **cond_kwargs, _teacache_branch="cond")

        # 无 CFG — 直接返回
        if uncond_kwargs is None:
            return noise_cond

        # MLX: eval cond branch 防止 lazy graph 堆积
        if ctx is not None and getattr(ctx, "backend", None) == "mlx":
            ctx.eval(noise_cond)

        noise_uncond = model(latents, t, **uncond_kwargs, _teacache_branch="uncond")
        if ctx is not None and getattr(ctx, "backend", None) == "mlx":
            ctx.eval(noise_uncond)

        noise_pred = model.combine_cfg_noise(noise_cond, noise_uncond, guidance)
        if cfg_renorm:
            noise_pred = model.refine_cfg_noise(
                noise_cond, noise_pred, cfg_renorm_min=cfg_renorm_min,
            )
        return noise_pred


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def resolve_cfg_strategy(
    model: Any,
    config: Any,
    ctx: Any,
) -> "FusedCfgStrategy | BatchedCfgStrategy | DualForwardCfgStrategy":
    """根据 model 能力和 backend 选择合适的 CFG dispatch 策略。

    优先级:
    1. ``FusedCfgStrategy`` — model 有 ``forward_cfg`` + MLX backend + config 允许
    2. ``BatchedCfgStrategy`` — model 有 ``predict_noise_cfg``（如 Wan）
    3. ``DualForwardCfgStrategy`` — 通用 fallback
    """
    # 1) Fused CFG — 单次 forward_cfg (Z-Image, FIBO on MLX)
    if (
        callable(getattr(model, "forward_cfg", None))
        and getattr(config, "supports_guidance", False)
        and getattr(config, "use_mlx_cfg_fusion", True)
        and getattr(ctx, "backend", None) == "mlx"
    ):
        return FusedCfgStrategy()

    # 2) Batched CFG — predict_noise_cfg (Wan, Hunyuan, FLUX1, Qwen)
    if (
        callable(getattr(model, "predict_noise_cfg", None))
        and getattr(config, "use_batched_cfg", True)
    ):
        return BatchedCfgStrategy()

    # 3) Dual forward — fallback
    return DualForwardCfgStrategy()


# ---------------------------------------------------------------------------
# 辅助：构建 uncond_kwargs（从 cond_kwargs + neg_embeds + overrides）
# ---------------------------------------------------------------------------

def build_uncond_kwargs(
    cond_kwargs: dict[str, Any],
    neg_embeds: Any,
    overrides: dict[str, Any] | None = None,
    *,
    exclude_keys: frozenset[str] | None = None,
) -> dict[str, Any]:
    """从 cond_kwargs 构建 uncond_kwargs — 替换 text embedding + 应用 overrides。

    ``exclude_keys`` 默认为 ``_TEXT_RELATED_KEYS``。
    """
    _exclude = exclude_keys or _TEXT_RELATED_KEYS
    uncond = {k: v for k, v in cond_kwargs.items() if k not in _exclude}
    uncond["txt_embeds"] = neg_embeds
    if overrides:
        uncond.update(overrides)
    return uncond


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _infer_text_keys(cond_kwargs: dict[str, Any]) -> frozenset[str]:
    """从 cond_kwargs 推断需要 batch 拼接的 text key 集合。"""
    keys = {"txt_embeds"}
    if "txt_attn_mask" in cond_kwargs:
        keys.add("txt_attn_mask")
    if "txt_embeds_2" in cond_kwargs:
        keys.add("txt_embeds_2")
    if "txt_attn_mask_2" in cond_kwargs:
        keys.add("txt_attn_mask_2")
    return frozenset(keys)
