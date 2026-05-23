"""
基准测试用例 — (1) 上游 mflux 参考 CLI vs 丹青 PSNR/SSIM；(2) 无参考时的成片健全性。

mflux 对比仅包含有 **真实 mflux 子命令** 的路径；用例按 ``default_config/models_registry.json`` 中
``engine: danqing-image`` 基础模型的 **fp16 默认版本** 与 **actions** 展开（不单独为 int8/int4
等量化变体建 PSNR 用例）。

要点（与 mflux 源码 / CLI 行为对齐）：
- **FLUX.2 Klein 蒸馏版**：``mflux-generate-flux2`` 在蒸馏权重上要求 ``--guidance 1.0``。
- **FIBO / FIBO-Lite**：``mflux-generate-fibo`` 对 ``--prompt`` 做 ``json.loads``；基准固定为合法 JSON，
  避免走 VLM 路径导致与丹青不可复现对比。
- **FIBO-Edit / FIBO-Edit-RMBG**：``mflux-generate-fibo-edit``；RMBG 使用与 mflux 默认一致的
  ``edit_instruction`` JSON（见 ``FIBO_EDIT_RMBG_JSON``）。
- **Qwen-Image**：注册表仅 ``rewrite``，PSNR 套件只含 ``rewrite``。
- **FLUX.1 Kontext**：``mflux-generate-kontext`` 将 ``--image-path`` 设为必填；无「纯文生」对位，
  故 PSNR 仅覆盖 ``rewrite``；``create`` 见 README 对照表。
- **FLUX.2 + ``vae_scale: 16`` 的 rewrite**：丹青管线若尚未与 16× VAE 网格对齐，``rewrite`` 可能在
  丹青侧 SKIP；用例仍保留以便回归。
- **Z-Image ``z-image-create`` vs mflux**：基准里 ``--guidance`` 与 mflux 一致（如 4.0）；与参考图
  PSNR 差主要来自 **CFG 合成公式**（mflux：``cond + g*(cond - uncond)``；丹青默认：diffusers 式
  ``uncond + g*(cond - uncond)``），不是 guidance 数值抄错。``guidance=0`` 的 ``z-image-rewrite`` 无 CFG，
  可对齐 PSNR。详见 ``BENCHMARK_EXIT_EXEMPT_MISMATCH_VS_MFLUX`` 中注释。

``ALL_SANITY_CASES``：无上游对照时，仅用像素统计拒绝白/黑/近单色平场（见 ``sanity.py``）。
SeedVR2 超分健全性：若 ``models/Upscaler/seedvr2-*-fp16`` 下缺少 ``job_mlx.expected_seedvr2_weight_files``
  所列文件，运行器 **SKIP**（不计 FAIL），与 ``make bench-seedvr2-mflux`` 对 3b 缺权重行为一致。

``python -m tests.benchmark mflux --all`` 的进程退出码：仅统计「非豁免」FAIL；豁免集合为
``BENCHMARK_EXIT_EXEMPT_MISMATCH_VS_MFLUX``（与 mflux 像素级尚未对齐、但仍保留对照输出的用例）。

小分辨率 (256px) 快速对比。

模型目录：读取 ``default_config/workspace.pointer.json`` 的 ``custom_workspace_dir``（见
``resolve_benchmark_data_root()``）；未配置时回退仓库根 ``models/``。

最近一次全量对照（2026-05-18，本机 studio-workspace）：

| 状态 | 用例 |
|------|------|
| PASS | seedvr2-7b-upscale；flux2-klein 全系 rewrite + base 文生图；flux2-klein-9b create；z-image-turbo create/rewrite；z-image-rewrite |
| WARN | flux2-klein-4b-create；z-image-create（豁免退出码）；qwen-image-rewrite（豁免） |
| SKIP 无权重 | seedvr2-3b；flux1-kontext；fibo-lite；fibo-edit |
| PASS | flux1-schnell/dev/krea-dev create（2026-05-19；RMSNorm eps + registry max_seq_len + 禁用 Flux.1 双路 CFG） |
| SKIP 丹青未跑通 | fibo create/rewrite；fibo-edit-rmbg（VAE encoder） |
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# 仓库根（``tests/benchmark/cases.py`` → 上三级）
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

SRC_IMAGE = "tests/benchmark/outputs/rewrite_src.png"

# mflux ``FiboEditUtil`` / CLI 默认 RMBG 指令（与 ``mflux.models.fibo.variants.edit.util`` 一致）
_FIBO_EDIT_RMBG_INSTRUCTION = (
    "Generate a detailed grayscale alpha matte. Map the opaque foreground to white "
    "and the background to black. Produce soft, anti-aliased grayscale gradients at the "
    "edges of the subject to represent fine details and transparency."
)
FIBO_EDIT_RMBG_JSON = json.dumps({"edit_instruction": _FIBO_EDIT_RMBG_INSTRUCTION}, ensure_ascii=False)

FIBO_JSON_TXT2IMG = json.dumps(
    {
        "style_medium": "photograph",
        "what": "a small terracotta vase on linen fabric, soft side light, shallow depth of field",
        "negative": "human, text, watermark",
    },
    ensure_ascii=False,
)
FIBO_JSON_REWRITE = json.dumps(
    {
        "edit_instruction": "Shift the color grade slightly cooler; increase gentle contrast on the subject.",
        "negative": "blur, watermark",
    },
    ensure_ascii=False,
)
FIBO_EDIT_JSON = json.dumps(
    {
        "edit_instruction": "Soften highlights slightly while keeping composition and major colors unchanged.",
        "negative": "add objects, text",
    },
    ensure_ascii=False,
)

# 与注册表 ``versions.*.local_path`` 对齐：图像基础模型在 ``models/Image/``，视频在 ``models/Video/``，
# 超分等在 ``models/Upscaler/``（供 ``--model`` 传入 mflux 本地路径）。
MFLUX_FP16_MODEL_ROOT: dict[str, str] = {
    "flux1-schnell": "models/Image/flux1-schnell-fp16",
    "flux1-dev": "models/Image/flux1-dev-fp16",
    "flux1-krea-dev": "models/Image/flux1-krea-dev-fp16",
    "flux1-kontext": "models/Image/flux1-kontext-fp16",
    "flux2-klein-4b": "models/Image/flux2-klein-4b-fp16",
    "flux2-klein-9b": "models/Image/flux2-klein-9b-fp16",
    "flux2-klein-base-4b": "models/Image/flux2-klein-base-4b-fp16",
    "flux2-klein-base-9b": "models/Image/flux2-klein-base-9b-fp16",
    "z-image": "models/Image/z-image-fp16",
    "z-image-turbo": "models/Image/z-image-turbo-fp16",
    "fibo": "models/Image/fibo-fp16",
    "fibo-lite": "models/Image/fibo-lite-fp16",
    "fibo-edit": "models/Image/fibo-edit-fp16",
    "fibo-edit-rmbg": "models/Image/fibo-edit-rmbg-fp16",
    "qwen-image": "models/Image/qwen-image-fp16",
    "seedvr2-3b": "models/Upscaler/seedvr2-3b-fp16",
    "seedvr2-7b": "models/Upscaler/seedvr2-7b-fp16",
}

# 可选 bundle：未安装时 sanity 音频用例 SKIP（不删用例，便于回归）
ACE_STEP_AUDIO_BUNDLE = "models/Audio/acestep-v15-xl-sft"
HEARTMULA_AUDIO_BUNDLE = "models/Audio/heartmula-oss-3b-happy-new-year"
WAN_VIDEO_BUNDLE = "models/Video/wan-2.2-ti2v-5b-original"


def ace_step_bundle_installed() -> bool:
    """ACE-Step audio weights (turbo DiT + VAE safetensors) under workspace models/."""
    root = resolve_benchmark_data_root() / ACE_STEP_AUDIO_BUNDLE
    dit = root / "acestep-v15-turbo" / "model.safetensors"
    vae = root / "vae" / "diffusion_pytorch_model.safetensors"
    enc = root / "Qwen3-Embedding-0.6B" / "config.json"
    return dit.is_file() and vae.is_file() and enc.is_file()


def heartmula_bundle_installed() -> bool:
    """HeartMuLa Gen + LM + Codec layout under workspace models/ (see ``bundle.py``)."""
    from backend.engine.families.heartmula.bundle import bundle_is_ready

    root = resolve_benchmark_data_root() / HEARTMULA_AUDIO_BUNDLE
    return bundle_is_ready(root)


def mlx_runtime_available() -> bool:
    try:
        import mlx.core  # noqa: F401

        return True
    except ImportError:
        return False


def wan_video_bundle_installed() -> bool:
    """Wan 2.2 TI2V 5B original bundle (T5 + VAE + DiT shards)."""
    root = resolve_benchmark_data_root() / WAN_VIDEO_BUNDLE
    if not root.is_dir():
        return False
    vae_ok = (
        (root / "Wan2.2_VAE.pth").is_file()
        or (root / "Wan2_2_VAE.pth").is_file()
        or (
            any((root / "vae").glob("*.safetensors"))
            if (root / "vae").is_dir()
            else False
        )
    )
    t5_ok = any(root.glob("models_t5*.pth"))
    dit_ok = any(root.rglob("*.safetensors")) or any(root.rglob("*.pth"))
    return bool(vae_ok and t5_ok and dit_ok)


MFLUX_OPTIONAL_FP16_MODELS: frozenset[str] = frozenset({
    "fibo-lite",
    "fibo-edit",
    "flux1-kontext",
    "seedvr2-3b",
})

# 视频权重目录为 ``models/Video/``（见注册表）；``ALL_CASES`` 尚无视频 PSNR 行，故未建 mflux 路径映射表。


def resolve_benchmark_data_root() -> Path:
    """``default_config/workspace.pointer.json`` 的 ``custom_workspace_dir``，否则仓库根。"""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from backend.utils.config_paths import resolve_default_config_root
    from backend.utils.workspace import resolve_workspace_root

    default_cfg = resolve_default_config_root(bootstrap_root=_REPO_ROOT, bundle_root=None)
    try:
        return resolve_workspace_root(_REPO_ROOT, default_config_root=default_cfg)
    except RuntimeError:
        pass
    return _REPO_ROOT


def resolve_fp16_bundle_dir(model_key: str) -> Path:
    """将 ``MFLUX_FP16_MODEL_ROOT`` 相对路径解析到有效数据根（workspace 或仓库）。"""
    rel = MFLUX_FP16_MODEL_ROOT.get(model_key)
    if not rel:
        raise KeyError(f"No fp16 bundle path for model {model_key!r}")
    text = rel.strip()
    if text.startswith("models/"):
        return (resolve_benchmark_data_root() / text).resolve()
    root = resolve_benchmark_data_root()
    return (root / text).resolve()


def fp16_bundle_installed(model_key: str) -> bool:
    return resolve_fp16_bundle_dir(model_key).is_dir()


def iter_mflux_cases() -> list[BenchmarkCase]:
    """仅返回本地 fp16 bundle 已存在的用例（见 ``default_config/workspace.pointer.json``）。"""
    runnable: list[BenchmarkCase] = []
    for case in ALL_CASES:
        base = case.model.split(":", 1)[0].strip()
        if fp16_bundle_installed(base):
            runnable.append(case)
    return runnable


def list_skipped_mflux_cases() -> list[tuple[str, str]]:
    """(case_id, reason) — bundle 目录缺失。"""
    skipped: list[tuple[str, str]] = []
    for case in ALL_CASES:
        base = case.model.split(":", 1)[0].strip()
        if fp16_bundle_installed(base):
            continue
        rel = MFLUX_FP16_MODEL_ROOT.get(base, "?")
        skipped.append((case.id, f"missing bundle {rel}"))
    return skipped

# ``python -m tests.benchmark mflux --all`` 退出码：以下用例仍打印 FAIL/WARN，但不计入失败总数
BENCHMARK_EXIT_EXEMPT_MISMATCH_VS_MFLUX: frozenset[str] = frozenset({
    "qwen-image-rewrite",
    # z-image-create: 基准里 guidance 与 mflux 相同；PSNR 差来自两条前向的合成方式（见模块说明）。
    # 在丹青侧硬套 mflux 公式而未先对齐单分支张量时，PSNR 会更差，故保留 diffusers 式合并 + 豁免。
    "z-image-create",
})


@dataclass
class BenchmarkCase:
    id: str
    model: str
    action: str
    prompt: str = ""
    seed: int = 42
    width: int = 256
    height: int = 256
    steps: int = 4
    guidance: float = 0.0
    description: str = ""
    image_strength: float = 0.6
    source_image: str = ""
    upscale_scale: int = 2
    scheduler: str = "flow_match_euler_discrete"
    negative_prompt: str = ""
    # mflux CLI
    _mflux_cli: str = "mflux-generate"
    _mflux_model_flag: str = ""
    _mflux_extra_args: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.description:
            self.description = f"{self.model} {self.action}"
        if not self._mflux_model_flag:
            self._mflux_model_flag = self._model_path()

    def _model_path(self) -> str:
        base, _, ver = self.model.partition(":")
        if ver and ver not in ("fp16", ""):
            raise ValueError(
                f"Benchmark PSNR cases use fp16 registry weights only; got model={self.model!r}"
            )
        key = base.strip()
        if key in MFLUX_FP16_MODEL_ROOT:
            return str(resolve_fp16_bundle_dir(key))
        raise KeyError(f"No fp16 bundle path registered for benchmark model {self.model!r}")


@dataclass
class SanityCase:
    """无 mflux 参考时的成片健全性用例（仅跑 danqing CLI + ``sanity.check_output_image``）。"""

    id: str
    model: str
    prompt: str = "a simple photograph"
    seed: int = 42
    width: int = 256
    height: int = 256
    steps: int = 8
    guidance: float = 3.5
    negative_prompt: str = ""
    description: str = ""
    action: str = "create"  # create | rewrite | upscale
    source_image: str = ""
    image_strength: float = 0.6
    upscale_scale: int = 2
    # ``danqing-*`` 子进程超时（秒）；首包 FIBO 冷启动可能 >10min
    timeout_sec: Optional[int] = None
    # audio (``danqing-audio-generate``) / video (``danqing-video-generate``)
    media: str = "image"  # image | audio | video
    duration: int = 10
    video_size: str = "480x720"
    video_num_frames: int = 17
    video_fps: int = 16
    is_timing_baseline: bool = False
    lyrics: str = ""
    audio_format: str = "wav"
    ace_step_use_lm: bool = False
    # HeartMuLa (optional; appended to ``danqing-audio-generate`` when set)
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    codec_steps: Optional[int] = None
    codec_guidance: Optional[float] = None

    def __post_init__(self):
        if not self.description:
            self.description = f"{self.model} output sanity"
        if self.action in ("rewrite", "upscale") and not self.source_image:
            self.source_image = SRC_IMAGE

    def as_benchmark_case(self) -> BenchmarkCase:
        return BenchmarkCase(
            id=self.id,
            model=self.model,
            action=self.action,
            prompt=self.prompt,
            seed=self.seed,
            width=self.width,
            height=self.height,
            steps=self.steps,
            guidance=self.guidance,
            description=self.description,
            negative_prompt=self.negative_prompt,
            source_image=self.source_image,
            image_strength=self.image_strength,
            upscale_scale=self.upscale_scale,
            _mflux_model_flag="__SANITY_NO_MFLUX_REF__",
        )


ALL_SANITY_CASES: list[SanityCase] = [
    SanityCase(
        id="fibo-sanity",
        model="fibo",
        prompt="a red apple on a wooden table, studio photograph",
        seed=42,
        steps=8,
        guidance=1.0,
        description="FIBO — 成片健全性（与 mflux PSNR 套件独立，可单独冒烟）",
    ),
    SanityCase(
        id="z-image-rewrite-sanity",
        model="z-image",
        action="rewrite",
        prompt="shift colors toward a cooler blue hour mood",
        seed=42,
        steps=4,
        guidance=3.5,
        description="Z-Image rewrite — 健全性（无 mflux 对照）",
    ),
    SanityCase(
        id="qwen-image-rewrite-sanity",
        model="qwen-image",
        action="rewrite",
        prompt="slightly warmer color grading, keep composition",
        seed=42,
        steps=4,
        guidance=2.0,
        description="Qwen-Image rewrite — 健全性（注册表仅声明 rewrite）",
    ),
    SanityCase(
        id="seedvr2-7b-upscale-sanity",
        model="seedvr2-7b",
        action="upscale",
        prompt="",
        seed=42,
        steps=1,
        guidance=0.0,
        upscale_scale=2,
        description="SeedVR2-7B 2× upscale — 健全性",
    ),
    SanityCase(
        id="seedvr2-3b-upscale-sanity",
        model="seedvr2-3b",
        action="upscale",
        prompt="",
        seed=43,
        steps=1,
        guidance=0.0,
        upscale_scale=2,
        description="SeedVR2-3B 2× upscale — 健全性",
    ),
    SanityCase(
        id="ace-step-xl-sft-sanity",
        model="ace-step-xl-sft",
        media="audio",
        prompt="warm piano ballad with gentle lead vocals",
        lyrics="[verse]\n月光洒在窗前\n静静听风的声音",
        duration=10,
        steps=8,
        guidance=3.0,
        seed=42,
        audio_format="wav",
        ace_step_use_lm=False,
        timeout_sec=300,
        description="ACE-Step MLX 10s audio — reject near-silent output (no mflux ref)",
    ),
    SanityCase(
        id="ace-step-xl-sft-cuda-sanity",
        model="ace-step-xl-sft",
        media="audio",
        prompt="warm piano ballad",
        lyrics="[Instrumental]",
        duration=10,
        steps=8,
        guidance=3.0,
        seed=44,
        audio_format="wav",
        ace_step_use_lm=False,
        timeout_sec=600,
        description="ACE-Step CUDA 10s audio — skip when torch.cuda unavailable",
    ),
    SanityCase(
        id="ace-step-xl-sft-sanity-lm",
        model="ace-step-xl-sft",
        media="audio",
        prompt="warm piano ballad",
        lyrics="[verse]\n月光洒在窗前",
        duration=10,
        steps=8,
        guidance=3.0,
        seed=43,
        audio_format="wav",
        ace_step_use_lm=True,
        timeout_sec=420,
        description="ACE-Step MLX 10s + 5Hz LM expansion — audio sanity",
    ),
    SanityCase(
        id="heartmula-oss-3b-happy-new-year-sanity",
        model="heartmula-oss-3b-happy-new-year",
        media="audio",
        prompt="pop, female vocal, acoustic, melodic",
        lyrics="[verse]\nHello world\n[chorus]\nSing along",
        duration=10,
        guidance=1.5,
        temperature=1.0,
        top_k=50,
        codec_steps=10,
        codec_guidance=1.25,
        seed=42,
        audio_format="wav",
        timeout_sec=900,
        description="HeartMuLa MLX 10s audio — reject near-silent output (no upstream ref)",
    ),
    SanityCase(
        id="wan-2.2-ti2v-5b-sanity",
        model="wan-2.2-ti2v-5b",
        media="video",
        prompt="a cat walking on green grass, soft daylight",
        video_size="480x720",
        video_num_frames=17,
        video_fps=16,
        steps=4,
        guidance=5.0,
        seed=42,
        timeout_sec=2400,
        description="Wan 2.2 TI2V 5B quick video sanity (4 steps, 17 frames)",
    ),
    SanityCase(
        id="wan-2.2-ti2v-5b-baseline",
        model="wan-2.2-ti2v-5b",
        media="video",
        prompt="a beautiful sunset over mountains, cinematic lighting",
        video_size="480x720",
        video_num_frames=81,
        video_fps=16,
        steps=8,
        guidance=5.0,
        seed=42,
        timeout_sec=7200,
        is_timing_baseline=True,
        description="Wan 2.2 TI2V 5B timing baseline (8 steps, 81 frames @ 480x720)",
    ),
]


def cuda_runtime_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def get_sanity_case(case_id: str) -> Optional[SanityCase]:
    for c in ALL_SANITY_CASES:
        if c.id == case_id:
            return c
    return None


def list_sanity_cases() -> list[str]:
    return [c.id for c in ALL_SANITY_CASES]


def _flux2_case(
    model_id: str,
    action: str,
    *,
    prompt: str,
    guidance: float,
    steps: int = 4,
    seed: int = 42,
    image_strength: float = 0.6,
    source: str = "",
) -> BenchmarkCase:
    cid = f"{model_id}-{action}"
    if action == "create":
        return BenchmarkCase(
            id=cid,
            model=model_id,
            action="create",
            prompt=prompt,
            seed=seed,
            steps=steps,
            guidance=guidance,
            _mflux_cli="mflux-generate-flux2",
            description=f"{model_id} 文生图 (fp16)",
        )
    return BenchmarkCase(
        id=cid,
        model=model_id,
        action="rewrite",
        prompt=prompt,
        seed=seed,
        steps=steps,
        guidance=guidance,
        source_image=source or SRC_IMAGE,
        image_strength=image_strength,
        _mflux_cli="mflux-generate-flux2",
        description=f"{model_id} 图生图 (fp16)",
    )


def _flux1_txt2img_case(model_id: str, action: str, *, prompt: str, seed: int = 42) -> BenchmarkCase:
    cid = f"{model_id}-{action}"
    if action == "create":
        return BenchmarkCase(
            id=cid,
            model=model_id,
            action="create",
            prompt=prompt,
            seed=seed,
            steps=4,
            guidance=3.5,
            scheduler="flow_match_euler_discrete",
            _mflux_cli="mflux-generate",
            description=f"{model_id} 文生图 (fp16)",
        )
    return BenchmarkCase(
        id=cid,
        model=model_id,
        action="rewrite",
        prompt=prompt,
        seed=seed,
        steps=4,
        guidance=3.5,
        scheduler="flow_match_euler_discrete",
        source_image=SRC_IMAGE,
        image_strength=0.55,
        _mflux_cli="mflux-generate",
        description=f"{model_id} 图生图 (fp16)",
    )


ALL_CASES: list[BenchmarkCase] = [
    # ----- SeedVR2 upscale -----
    BenchmarkCase(
        id="seedvr2-7b-upscale",
        model="seedvr2-7b",
        action="upscale",
        seed=42,
        steps=1,
        guidance=0.0,
        scheduler="",
        upscale_scale=2,
        _mflux_cli="mflux-upscale-seedvr2",
        source_image=SRC_IMAGE,
        description="SeedVR2-7B 超分 2×",
    ),
    BenchmarkCase(
        id="seedvr2-3b-upscale",
        model="seedvr2-3b",
        action="upscale",
        seed=42,
        steps=1,
        guidance=0.0,
        scheduler="",
        upscale_scale=2,
        _mflux_cli="mflux-upscale-seedvr2",
        source_image=SRC_IMAGE,
        description="SeedVR2-3B 超分 2×",
    ),
    # ----- FLUX.2 Klein fp16（注册表 create + rewrite）-----
    _flux2_case(
        "flux2-klein-4b",
        "create",
        prompt="a ceramic bowl on a wooden table, soft window light",
        guidance=1.0,
    ),
    _flux2_case(
        "flux2-klein-4b",
        "rewrite",
        prompt="add subtle rim light and slightly deeper shadows",
        guidance=1.0,
    ),
    _flux2_case(
        "flux2-klein-9b",
        "create",
        prompt="a futuristic city skyline at night, neon lights",
        guidance=1.0,
        steps=4,
    ),
    _flux2_case(
        "flux2-klein-9b",
        "rewrite",
        prompt="add dramatic storm clouds and lightning",
        guidance=1.0,
    ),
    _flux2_case(
        "flux2-klein-base-4b",
        "create",
        prompt="a single oak tree on a grassy hill, golden hour",
        guidance=1.5,
        steps=4,
    ),
    _flux2_case(
        "flux2-klein-base-4b",
        "rewrite",
        prompt="shift the lighting toward blue hour with cooler tones",
        guidance=1.5,
        steps=4,
    ),
    _flux2_case(
        "flux2-klein-base-9b",
        "create",
        prompt="a misty mountain valley at dawn, wide shot",
        guidance=1.5,
        steps=4,
    ),
    _flux2_case(
        "flux2-klein-base-9b",
        "rewrite",
        prompt="increase atmospheric haze and soften distant peaks",
        guidance=1.5,
        steps=4,
    ),
    # ----- FLUX.1 fp16 -----
    _flux1_txt2img_case("flux1-schnell", "create", prompt="a red bicycle leaning against a brick wall"),
    _flux1_txt2img_case("flux1-schnell", "rewrite", prompt="add fresh rain puddles and wet reflections"),
    _flux1_txt2img_case("flux1-dev", "create", prompt="a steaming cup of tea on a book, cozy indoor light"),
    _flux1_txt2img_case("flux1-dev", "rewrite", prompt="add a gentle warm lamp glow on the scene"),
    _flux1_txt2img_case("flux1-krea-dev", "create", prompt="portrait of a person in natural window light"),
    _flux1_txt2img_case("flux1-krea-dev", "rewrite", prompt="slightly increase skin micro-contrast, keep identity"),
    # Kontext：仅 rewrite（mflux-generate-kontext 要求参考图）
    BenchmarkCase(
        id="flux1-kontext-rewrite",
        model="flux1-kontext",
        action="rewrite",
        prompt="add a soft cinematic color grade, keep composition",
        seed=42,
        steps=8,
        guidance=2.5,
        scheduler="linear",
        source_image=SRC_IMAGE,
        image_strength=0.4,
        _mflux_cli="mflux-generate-kontext",
        description="FLUX.1 Kontext 条件编辑 (fp16)",
    ),
    # ----- Z-Image -----
    BenchmarkCase(
        id="z-image-turbo-create",
        model="z-image-turbo",
        action="create",
        prompt="a serene lake at sunset, mountains, oil painting style",
        seed=42,
        steps=4,
        guidance=0.0,
        scheduler="linear",
        _mflux_cli="mflux-generate-z-image-turbo",
        description="Z-Image Turbo 文生图",
    ),
    BenchmarkCase(
        id="z-image-turbo-rewrite",
        model="z-image-turbo",
        action="rewrite",
        prompt="add a golden sunset glow and dramatic clouds",
        seed=42,
        steps=4,
        guidance=0.0,
        scheduler="linear",
        source_image=SRC_IMAGE,
        image_strength=0.6,
        _mflux_cli="mflux-generate-z-image-turbo",
        description="Z-Image Turbo 图生图",
    ),
    BenchmarkCase(
        id="z-image-create",
        model="z-image",
        action="create",
        prompt="a mountain landscape, oil painting",
        seed=42,
        steps=4,
        guidance=4.0,
        scheduler="flow_match_euler_discrete",
        _mflux_cli="mflux-generate-z-image",
        description="Z-Image 文生图",
    ),
    BenchmarkCase(
        id="z-image-rewrite",
        model="z-image",
        action="rewrite",
        prompt="turn this into a winter snowy scene",
        seed=42,
        steps=4,
        guidance=0.0,
        scheduler="flow_match_euler_discrete",
        source_image=SRC_IMAGE,
        image_strength=0.6,
        _mflux_cli="mflux-generate-z-image",
        description="Z-Image 图生图",
    ),
    # ----- Qwen-Image（仅 rewrite）-----
    BenchmarkCase(
        id="qwen-image-rewrite",
        model="qwen-image",
        action="rewrite",
        prompt="change to a snowy winter scene",
        seed=42,
        steps=4,
        guidance=1.0,
        scheduler="flow_match_euler_discrete",
        source_image=SRC_IMAGE,
        image_strength=0.6,
        _mflux_cli="mflux-generate-qwen",
        description="Qwen-Image 图生图",
    ),
    # ----- FIBO / FIBO-Lite（JSON 提示）-----
    BenchmarkCase(
        id="fibo-create",
        model="fibo",
        action="create",
        prompt=FIBO_JSON_TXT2IMG,
        seed=42,
        steps=8,
        guidance=5.0,
        scheduler="flow_match_euler_discrete",
        _mflux_cli="mflux-generate-fibo",
        description="FIBO 文生图 (JSON)",
    ),
    BenchmarkCase(
        id="fibo-rewrite",
        model="fibo",
        action="rewrite",
        prompt=FIBO_JSON_REWRITE,
        seed=42,
        steps=8,
        guidance=5.0,
        scheduler="flow_match_euler_discrete",
        source_image=SRC_IMAGE,
        image_strength=0.55,
        _mflux_cli="mflux-generate-fibo",
        description="FIBO 图生图 (JSON)",
    ),
    BenchmarkCase(
        id="fibo-lite-create",
        model="fibo-lite",
        action="create",
        prompt=FIBO_JSON_TXT2IMG,
        seed=42,
        steps=8,
        guidance=1.0,
        scheduler="flow_match_euler_discrete",
        _mflux_cli="mflux-generate-fibo",
        description="FIBO-Lite 文生图 (JSON)",
    ),
    BenchmarkCase(
        id="fibo-lite-rewrite",
        model="fibo-lite",
        action="rewrite",
        prompt=FIBO_JSON_REWRITE,
        seed=42,
        steps=8,
        guidance=1.0,
        scheduler="flow_match_euler_discrete",
        source_image=SRC_IMAGE,
        image_strength=0.55,
        _mflux_cli="mflux-generate-fibo",
        description="FIBO-Lite 图生图 (JSON)",
    ),
    BenchmarkCase(
        id="fibo-edit-rewrite",
        model="fibo-edit",
        action="rewrite",
        prompt=FIBO_EDIT_JSON,
        seed=42,
        steps=8,
        guidance=3.5,
        scheduler="flow_match_euler_discrete",
        source_image=SRC_IMAGE,
        image_strength=0.55,
        _mflux_cli="mflux-generate-fibo-edit",
        description="FIBO-Edit 图像编辑 (JSON)",
    ),
    BenchmarkCase(
        id="fibo-edit-rmbg-rewrite",
        model="fibo-edit-rmbg",
        action="rewrite",
        prompt=FIBO_EDIT_RMBG_JSON,
        seed=42,
        steps=8,
        guidance=1.0,
        scheduler="flow_match_euler_discrete",
        source_image=SRC_IMAGE,
        image_strength=0.55,
        _mflux_cli="mflux-generate-fibo-edit",
        description="FIBO-Edit-RMBG 抠像 (JSON)",
    ),
]


def get_case(case_id: str) -> Optional[BenchmarkCase]:
    for c in ALL_CASES:
        if c.id == case_id:
            return c
    return None


def list_cases() -> list[str]:
    return [c.id for c in ALL_CASES]
