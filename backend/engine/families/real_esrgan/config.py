"""Real-ESRGAN variant metadata (matches mlx-community ``config.json``)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

from backend.engine.families.real_esrgan.arch import RRDBNet, SRVGGNetCompact


@dataclass(frozen=True)
class RealESRGANVariant:
    name: str
    arch: str
    netscale: int
    num_feat: int = 64
    num_block: int = 23
    num_grow_ch: int = 32
    num_conv: int = 16
    act_type: str = "prelu"

    @classmethod
    def from_config(cls, data: dict[str, Any]) -> RealESRGANVariant:
        return cls(
            name=str(data.get("name", "RealESRGAN_x4plus")),
            arch=str(data.get("arch", "RRDBNet")),
            netscale=int(data.get("netscale", 4)),
            num_feat=int(data.get("num_feat", 64)),
            num_block=int(data.get("num_block", 23)),
            num_grow_ch=int(data.get("num_grow_ch", 32)),
            num_conv=int(data.get("num_conv", 16)),
            act_type=str(data.get("act_type", "prelu")),
        )


def load_variant_config(bundle_path: Path) -> RealESRGANVariant:
    cfg_path = bundle_path / "config.json"
    if not cfg_path.is_file():
        raise RuntimeError(
            f"Real-ESRGAN bundle at {bundle_path} is missing config.json "
            "(expected mlx-community converted layout)."
        )
    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return RealESRGANVariant.from_config(data)


def build_arch(variant: RealESRGANVariant):
    if variant.arch == "RRDBNet":
        return RRDBNet(
            3,
            3,
            scale=variant.netscale,
            num_feat=variant.num_feat,
            num_block=variant.num_block,
            num_grow_ch=variant.num_grow_ch,
        )
    if variant.arch == "SRVGGNetCompact":
        return SRVGGNetCompact(
            3,
            3,
            num_feat=variant.num_feat,
            num_conv=variant.num_conv,
            upscale=variant.netscale,
            act_type=variant.act_type,
        )
    raise RuntimeError(f"Unsupported Real-ESRGAN arch {variant.arch!r}")


def _dni_blend(general: dict, wdn: dict, denoise_strength: float) -> dict:
    s = float(denoise_strength)
    return {k: general[k] * s + wdn[k] * (1.0 - s) for k in general}


def load_model_from_bundle(
    bundle_path: Path,
    *,
    denoise_strength: float = 1.0,
):
    variant = load_variant_config(bundle_path)
    weights_path = bundle_path / "model.safetensors"
    if not weights_path.is_file():
        raise RuntimeError(
            f"Real-ESRGAN bundle at {bundle_path} is missing model.safetensors"
        )

    weights = dict(mx.load(str(weights_path)).items())
    wdn_path = bundle_path / "model_wdn.safetensors"
    if wdn_path.is_file() and denoise_strength < 1.0:
        wdn = dict(mx.load(str(wdn_path)).items())
        weights = _dni_blend(weights, wdn, denoise_strength)

    model = build_arch(variant)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    model.eval()
    return model, variant


def expected_weight_files() -> tuple[str, ...]:
    return ("config.json", "model.safetensors")
