"""
基准测试用例 — 5 模型 x 全 action。

seedvr2-7b(upscale) / flux2-klein-9b(create+rewrite) / z-image-turbo(create+rewrite)
qwen-image(create+rewrite) / z-image(create+rewrite)

小分辨率 (256px) 快速对比。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

SRC_IMAGE = "tests/benchmark/outputs/rewrite_src.png"


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
        m = self.model.lower()
        if "seedvr2" in m: return "models/Upscaler/seedvr2-7b-fp16"
        if "flux2" in m: return "models/Base/flux2-klein-9b-fp16"
        if "z-image-turbo" in m: return "models/Base/z-image-turbo-fp16"
        if "qwen" in m: return "models/Base/qwen-image-fp16"
        if "z-image" in m: return "models/Base/z-image-fp16"
        return "models/Base/z-image-turbo-fp16"


ALL_CASES: list[BenchmarkCase] = [
    # ============ seedvr2-7b (upscale) ============
    BenchmarkCase(
        id="seedvr2-7b-upscale",
        model="seedvr2-7b", action="upscale",
        seed=42,
        _mflux_cli="mflux-upscale-seedvr2",
        source_image=SRC_IMAGE,
        description="SeedVR2-7B 超分 2x",
    ),
    # ============ flux2-klein-9b (create + rewrite) ============
    BenchmarkCase(
        id="flux2-klein-9b-create",
        model="flux2-klein-9b", action="create",
        prompt="a futuristic city skyline at night, neon lights",
        seed=42, steps=4,
        _mflux_cli="mflux-generate-flux2",
        description="Flux2-Klein-9B 文生图",
    ),
    BenchmarkCase(
        id="flux2-klein-9b-rewrite",
        model="flux2-klein-9b", action="rewrite",
        prompt="add dramatic storm clouds and lightning",
        seed=42, steps=4,
        _mflux_cli="mflux-generate-flux2",
        source_image=SRC_IMAGE,
        description="Flux2-Klein-9B 图生图",
    ),
    # ============ z-image-turbo (create + rewrite) ============
    BenchmarkCase(
        id="z-image-turbo-create",
        model="z-image-turbo", action="create",
        prompt="a serene lake at sunset, mountains, oil painting style",
        seed=42, steps=4,
        _mflux_cli="mflux-generate-z-image-turbo",
        description="Z-Image Turbo 文生图",
    ),
    BenchmarkCase(
        id="z-image-turbo-rewrite",
        model="z-image-turbo", action="rewrite",
        prompt="add a golden sunset glow and dramatic clouds",
        seed=42, steps=4,
        _mflux_cli="mflux-generate-z-image-turbo",
        source_image=SRC_IMAGE,
        description="Z-Image Turbo 图生图",
    ),
    # ============ qwen-image (create + rewrite) ============
    BenchmarkCase(
        id="qwen-image-create",
        model="qwen-image", action="create",
        prompt="a cute cat sitting on a windowsill, soft lighting",
        seed=42, steps=4,
        _mflux_cli="mflux-generate-qwen",
        description="Qwen-Image 文生图",
    ),
    BenchmarkCase(
        id="qwen-image-rewrite",
        model="qwen-image", action="rewrite",
        prompt="change to a snowy winter scene",
        seed=42, steps=4,
        _mflux_cli="mflux-generate-qwen",
        source_image=SRC_IMAGE,
        description="Qwen-Image 图生图",
    ),
    # ============ z-image (create + rewrite) ============
    BenchmarkCase(
        id="z-image-create",
        model="z-image", action="create",
        prompt="a mountain landscape, oil painting",
        seed=42, steps=4,
        _mflux_cli="mflux-generate-z-image",
        description="Z-Image 文生图",
    ),
    BenchmarkCase(
        id="z-image-rewrite",
        model="z-image", action="rewrite",
        prompt="turn this into a winter snowy scene",
        seed=42, steps=4,
        _mflux_cli="mflux-generate-z-image",
        source_image=SRC_IMAGE,
        description="Z-Image 图生图",
    ),
]


def get_case(case_id: str) -> Optional[BenchmarkCase]:
    for c in ALL_CASES:
        if c.id == case_id: return c
    return None


def list_cases() -> list[str]:
    return [c.id for c in ALL_CASES]
