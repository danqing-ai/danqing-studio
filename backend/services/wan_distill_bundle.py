"""Assemble Wan 2.2 LightX2V distill DiT shards into MoE bundle layout."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

# Keys → (high_noise_filename, low_noise_filename) at bundle root after download.
# ModelScope mirror: https://modelscope.cn/models/lightx2v/Wan2.2-Distill-Models
_WAN_DISTILL_VARIANTS: dict[str, tuple[str, str]] = {
    # 2026-04-12 — latest I2V 720p BF16 (README: fine detail + texture)
    "i2v_720p": (
        "wan2.2_i2v_A14b_high_noise_lightx2v_4step_720p_260412.safetensors",
        "wan2.2_i2v_A14b_low_noise_lightx2v_4step_720p_260412.safetensors",
    ),
    # Lighter I2V fallback (~15 GB per expert)
    "i2v_fp8": (
        "wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors",
        "wan2.2_i2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors",
    ),
    # T2V — no 720p drop yet on Distill-Models; FP8 is current upstream artifact
    "t2v_fp8": (
        "wan2.2_t2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors",
        "wan2.2_t2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors",
    ),
}


def assemble_wan_distill_bundle(bundle_root: Path, variant: str) -> None:
    """Move LightX2V distill safetensors into ``high_noise_model/`` + ``low_noise_model/``."""
    root = Path(bundle_root)
    vae21 = root / "Wan2.1_VAE.pth"
    vae22 = root / "Wan2.2_VAE.pth"
    if vae21.is_file() and not vae22.exists():
        vae22.symlink_to(vae21.name)
    if not variant:
        raise RuntimeError("wan_distill_variant is required for Wan distill bundle assembly.")
    names = _WAN_DISTILL_VARIANTS.get(str(variant))
    if names is None:
        known = ", ".join(sorted(_WAN_DISTILL_VARIANTS))
        raise RuntimeError(
            f"Unknown wan_distill_variant {variant!r}. Supported: {known}."
        )

    high_name, low_name = names
    for expert, filename in (("high", high_name), ("low", low_name)):
        src = root / filename
        dest_dir = root / ("high_noise_model" if expert == "high" else "low_noise_model")
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "diffusion_pytorch_model.safetensors"

        if dest.is_file():
            # Older installs copied instead of moving; drop the redundant root shard.
            if src.is_file():
                try:
                    if src.resolve() != dest.resolve():
                        src.unlink()
                except OSError:
                    src.unlink()
            continue

        if not src.is_file():
            raise RuntimeError(
                f"Wan distill bundle missing {filename} under {root}. "
                "Check allow_patterns for lightx2v/Wan2.2-Distill-Models (ModelScope)."
            )
        shutil.move(str(src), str(dest))

    index_path = root / "model_index.json"
    payload: dict[str, object] = {}
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "_wan_distill_variant": variant,
            "_danqing_bundle_source": "lightx2v_distill",
            "dual_model": True,
        }
    )
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
