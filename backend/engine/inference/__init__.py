"""Layer 2 推理架构 — 公共 API。

| 策略 | 用途 | 状态 |
|------|------|------|
| ``DiffusionInference`` | 标准 N 步扩散 (image/video) | 已接入 Pipeline |
| ``FlowMatchingInference`` | Euler flow-matching (ACE-Step DiT) | 已接入 |
| ``BlockAutoregressiveInference`` | Block-AR + 内层 CFM (DiffRhythm) | 已接入 |
| ``AutoregressiveInference`` | Token LM 解码 (ACE-Step constrained LM) | 已接入 |
| ``TwoStageInference`` | LM→DiT 两阶段编排 (ACE-Step) | 已接入 |
"""
from backend.engine.inference._protocols import (
    AudioInferenceBundle,
    AudioInferenceStrategy,
    BlockAutoregressiveSpec,
    CfgStrategy,
    DenoiseStepResult,
    FlowMatchingSpec,
    InferenceBundle,
    InferenceStrategy,
    StepKwargsBuilder,
)
from backend.engine.inference.cfg_strategies import (
    BatchedCfgStrategy,
    DualForwardCfgStrategy,
    FusedCfgStrategy,
    build_uncond_kwargs,
    resolve_cfg_strategy,
)
from backend.engine.inference.autoregressive import (
    AutoregressiveBundle,
    AutoregressiveInference,
)
from backend.engine.inference.block_ar import BlockAutoregressiveInference
from backend.engine.inference.audio_edit import run_audio_edit_handler
from backend.engine.inference.audio_waveform import run_audio_waveform
from backend.engine.inference.diffusion import DiffusionInference
from backend.engine.inference.diffusion_bundle import run_diffusion_denoise
from backend.engine.inference.flow_matching import FlowMatchingInference
from backend.engine.inference.image_denoise import run_image_denoise
from backend.engine.inference.optimization_plan import (
    ImageInferencePlan,
    InferencePlan,
    VideoInferencePlan,
    resolve_image_inference_plan,
    resolve_video_inference_plan,
)
from backend.engine.inference.job import JobBundle, run_job
from backend.engine.inference.upscale_job import run_upscale_job
from backend.engine.inference.video_denoise import run_video_denoise
from backend.engine.inference.video_two_stage import run_family_video_generator
from backend.engine.inference.video_upscale_job import run_video_upscale_job
from backend.engine.inference.memory_guard import MemoryGuard
from backend.engine.inference.step_kwargs_builders import (
    FixedStepKwargsBuilder,
    ImageStepKwargsBuilder,
    VideoStepKwargsBuilder,
)
from backend.engine.inference.two_stage import TwoStageBundle, TwoStageInference

# v3 alias — same dataclass until bundle grows extra fields
ParadigmBundle = InferenceBundle

__all__ = [
    "AutoregressiveBundle",
    "AutoregressiveInference",
    "AudioInferenceBundle",
    "AudioInferenceStrategy",
    "BlockAutoregressiveInference",
    "BlockAutoregressiveSpec",
    "CfgStrategy",
    "DenoiseStepResult",
    "DiffusionInference",
    "FlowMatchingInference",
    "FlowMatchingSpec",
    "JobBundle",
    "ParadigmBundle",
    "run_audio_edit_handler",
    "run_audio_waveform",
    "run_diffusion_denoise",
    "run_family_video_generator",
    "run_image_denoise",
    "run_job",
    "run_upscale_job",
    "run_video_denoise",
    "run_video_upscale_job",
    "FixedStepKwargsBuilder",
    "FusedCfgStrategy",
    "BatchedCfgStrategy",
    "DualForwardCfgStrategy",
    "InferenceBundle",
    "InferenceStrategy",
    "ImageStepKwargsBuilder",
    "ImageInferencePlan",
    "VideoInferencePlan",
    "InferencePlan",
    "MemoryGuard",
    "StepKwargsBuilder",
    "TwoStageBundle",
    "TwoStageInference",
    "VideoStepKwargsBuilder",
    "build_uncond_kwargs",
    "resolve_cfg_strategy",
    "resolve_image_inference_plan",
    "resolve_video_inference_plan",
]
