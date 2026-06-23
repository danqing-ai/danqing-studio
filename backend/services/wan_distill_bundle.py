"""Assemble Wan 2.2 LightX2V / TurboDiffusion distill DiT into MoE or flat bundle layout."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

# ModelScope: https://modelscope.cn/models/lightx2v/Wan2.2-Distill-Models
_WAN_DISTILL_VARIANTS: dict[str, tuple[str, str]] = {
    "i2v_720p": (
        "wan2.2_i2v_A14b_high_noise_lightx2v_4step_720p_260412.safetensors",
        "wan2.2_i2v_A14b_low_noise_lightx2v_4step_720p_260412.safetensors",
    ),
    "i2v_fp8": (
        "wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors",
        "wan2.2_i2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors",
    ),
    "t2v_fp8": (
        "wan2.2_t2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors",
        "wan2.2_t2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors",
    ),
    "turbo_i2v_720p": (
        "TurboWan2.2-I2V-A14B-high-720P.pth",
        "TurboWan2.2-I2V-A14B-low-720P.pth",
    ),
    "turbo_i2v_720p_quant": (
        "TurboWan2.2-I2V-A14B-high-720P-quant.pth",
        "TurboWan2.2-I2V-A14B-low-720P-quant.pth",
    ),
}

_WAN_TURBO_SINGLE_VARIANTS: dict[str, str] = {
    "turbo_t2v_480p_14b": "TurboWan2.1-T2V-14B-480P.pth",
    "turbo_t2v_480p_14b_quant": "TurboWan2.1-T2V-14B-480P-quant.pth",
    "turbo_t2v_720p_14b": "TurboWan2.1-T2V-14B-720P.pth",
    "turbo_t2v_720p_14b_quant": "TurboWan2.1-T2V-14B-720P-quant.pth",
    "turbo_t2v_480p_1.3b": "TurboWan2.1-T2V-1.3B-480P.pth",
    "turbo_t2v_480p_1.3b_quant": "TurboWan2.1-T2V-1.3B-480P-quant.pth",
}


def _expert_dest_name(src: Path) -> str:
    if src.suffix.lower() == ".pth":
        return "diffusion_pytorch_model.pth"
    return "diffusion_pytorch_model.safetensors"


def _move_expert_shard(src: Path, dest_dir: Path) -> None:
    if not src.is_file():
        raise RuntimeError(f"Wan distill bundle missing {src.name} under {src.parent}.")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / _expert_dest_name(src)
    if dest.exists():
        dest.unlink()
    shutil.move(str(src), str(dest))


def assemble_wan_distill_bundle(bundle_root: Path, variant: str) -> None:
    """Move LightX2V / TurboDiffusion shards into MoE or flat Wan bundle layout."""
    root = Path(bundle_root)
    if not variant:
        raise RuntimeError("wan_distill_variant is required for Wan distill bundle assembly.")

    vae21 = root / "Wan2.1_VAE.pth"
    vae22 = root / "Wan2.2_VAE.pth"
    if vae21.is_file() and not vae22.exists():
        vae22.symlink_to(vae21.name)

    single_name = _WAN_TURBO_SINGLE_VARIANTS.get(str(variant))
    if single_name is not None:
        _move_expert_shard(root / single_name, root)
        source = "turbodiffusion"
        dual = False
    else:
        names = _WAN_DISTILL_VARIANTS.get(str(variant))
        if names is None:
            known = ", ".join(sorted(set(_WAN_DISTILL_VARIANTS) | set(_WAN_TURBO_SINGLE_VARIANTS)))
            raise RuntimeError(f"Unknown wan_distill_variant {variant!r}. Supported: {known}.")
        high_name, low_name = names
        for expert, filename in (("high", high_name), ("low", low_name)):
            dest_dir = root / ("high_noise_model" if expert == "high" else "low_noise_model")
            _move_expert_shard(root / filename, dest_dir)
        source = "turbodiffusion" if str(variant).startswith("turbo_") else "lightx2v_distill"
        dual = True

    index_path = root / "model_index.json"
    payload: dict[str, object] = {}
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "_wan_distill_variant": variant,
            "_danqing_bundle_source": source,
            "dual_model": dual,
        }
    )
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
