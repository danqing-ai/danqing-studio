"""Layer 2 推理架构 — 协议与数据结构定义。

Pipeline (L1) 通过 ``InferenceBundle`` / ``AudioInferenceBundle`` 将全部上下文
传递给 ``InferenceStrategy`` (L2)。L2 不区分 image/video 媒体 — ``DiffusionInference``
同时服务 ImagePipeline 与 VideoPipeline。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# CfgStrategy — CFG dispatch 策略协议
# ---------------------------------------------------------------------------

@runtime_checkable
class CfgStrategy(Protocol):
    """Classifier-free guidance 噪声预测策略。"""

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
    ) -> Any: ...


# ---------------------------------------------------------------------------
# StepKwargsBuilder — 每步 kwargs 构建协议
# ---------------------------------------------------------------------------

@runtime_checkable
class StepKwargsBuilder(Protocol):
    """构建每步 denoise 的 model forward kwargs。"""

    def build_cond_kwargs(
        self, t: Any, *, step_idx: int, sigmas: Any,
        timestep_embed_value: float | None,
    ) -> dict[str, Any]: ...

    def build_uncond_kwargs(
        self, t: Any, *, step_idx: int, sigmas: Any,
        timestep_embed_value: float | None,
    ) -> dict[str, Any] | None: ...


# ---------------------------------------------------------------------------
# InferenceStrategy — 扩散 L2 协议
# ---------------------------------------------------------------------------

@runtime_checkable
class InferenceStrategy(Protocol):
    """Layer 2 扩散推理策略 — Pipeline (L1) 只看到这个接口。"""

    def run(self, bundle: "InferenceBundle") -> Any:
        """执行推理，返回 denoised latents。"""
        ...


@dataclass
class DenoiseStepResult:
    """每步 denoise 的结构化结果 — pipeline 消费此对象做 preview / progress。"""

    step_idx: int
    total_steps: int
    latents: Any
    noise_pred: Any
    memory_cleared: bool = False


@dataclass
class InferenceBundle:
    """Pipeline (L1) 传给 ``DiffusionInference`` 的全部上下文。"""

    ctx: Any
    model: Any
    config: Any
    scheduler: Any
    timesteps: Any
    sigmas: Any | None = None
    latent_shape: tuple[int, ...] = ()
    seed: int = 0
    guidance: float = 0.0
    cfg_renorm: bool = False
    cfg_renorm_min: float = 0.0
    cancel_token: Any | None = None
    txt_embeds: Any | None = None
    neg_embeds: Any | None = None
    extra_cond: dict[str, Any] = field(default_factory=dict)
    cfg_strategy: CfgStrategy | None = None
    step_kwargs_builder: StepKwargsBuilder | None = None
    pack_fn: Callable | None = None
    unpack_fn: Callable | None = None
    latent_h: int = 0
    latent_w: int = 0
    init_latents: Any | None = None
    init_noise_sigma: float = 1.0
    scale_model_input: bool = False
    step_post_fns: list[Callable] = field(default_factory=list)
    memory_guard: Any | None = None
    on_step_complete: Callable | None = None


# ---------------------------------------------------------------------------
# Audio L2 — 按策略拆分的 spec + bundle
# ---------------------------------------------------------------------------

@dataclass
class FlowMatchingSpec:
    """``FlowMatchingInference`` 专用字段。"""

    timestep_schedule: list[float] | None = None
    init_noise_fn: Callable | None = None
    euler_step_fn: Callable | None = None
    cache_init_fn: Callable | None = None
    before_step_fn: Callable | None = None


@dataclass
class BlockAutoregressiveSpec:
    """``BlockAutoregressiveInference`` 外层 block 循环字段。"""

    num_blocks: int = 0
    seed_fn: Callable | None = None
    setup_fn: Callable | None = None
    before_block_fn: Callable | None = None
    after_block_fn: Callable | None = None
    eos_check_fn: Callable | None = None


@runtime_checkable
class AudioInferenceStrategy(Protocol):
    """Layer 2 音频推理策略协议。"""

    def run(self, bundle: "AudioInferenceBundle") -> Any:
        """执行推理，返回 denoised latents 或结构化结果。"""
        ...


@dataclass
class AudioInferenceBundle:
    """Audio generator → L2 共享上下文。

    各策略只读取自己需要的 ``flow`` / ``block_ar`` 段，避免 flat god-object。
    """

    ctx: Any
    model_forward: Callable
    latent_shape: tuple[int, ...] = ()
    seed: int = 0
    cancel_token: Any | None = None
    on_step_complete: Callable | None = None
    memory_guard: Any | None = None
    eval_fn: Callable | None = None
    flow: FlowMatchingSpec = field(default_factory=FlowMatchingSpec)
    block_ar: BlockAutoregressiveSpec = field(default_factory=BlockAutoregressiveSpec)
