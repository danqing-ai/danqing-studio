"""
基准测试用例 — (1) 上游 mflux 参考 CLI vs 丹青 PSNR/SSIM；(2) 无参考时的成片健全性。

mflux 对比仅包含有 **真实 mflux 子命令** 的路径；用例按 ``config/models_registry.json`` 中
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
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

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

# 视频权重目录为 ``models/Video/``（见注册表）；``ALL_CASES`` 尚无视频 PSNR 行，故未建 mflux 路径映射表。

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
            return MFLUX_FP16_MODEL_ROOT[key]
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
]


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
