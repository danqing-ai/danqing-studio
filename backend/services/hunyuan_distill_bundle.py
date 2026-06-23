"""Assemble HunyuanVideo-1.5 LightX2V single-file DiT into diffusers ``transformer/`` layout."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

# ModelScope mirror: https://modelscope.cn/models/lightx2v/Hy1.5-Distill-Models
_HUNYUAN_DISTILL_VARIANTS: dict[str, str] = {
    "t2v_480p": "hy1.5_t2v_480p_lightx2v_4step.safetensors",
}


def assemble_hunyuan_distill_bundle(bundle_root: Path, variant: str) -> None:
    """Move LightX2V distill safetensor into ``transformer/diffusion_pytorch_model.safetensors``."""
    root = Path(bundle_root)
    if not variant:
        raise RuntimeError("hunyuan_distill_variant is required for Hunyuan distill bundle assembly.")
    filename = _HUNYUAN_DISTILL_VARIANTS.get(str(variant))
    if filename is None:
        known = ", ".join(sorted(_HUNYUAN_DISTILL_VARIANTS))
        raise RuntimeError(
            f"Unknown hunyuan_distill_variant {variant!r}. Supported: {known}."
        )

    dest_dir = root / "transformer"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "diffusion_pytorch_model.safetensors"
    src = root / filename

    if dest.is_file():
        if src.is_file():
            try:
                if src.resolve() != dest.resolve():
                    src.unlink()
            except OSError:
                src.unlink()
    else:
        if not src.is_file():
            raise RuntimeError(
                f"Hunyuan distill bundle missing {filename} under {root}. "
                "Check allow_patterns for lightx2v/Hy1.5-Distill-Models (ModelScope)."
            )
        shutil.move(str(src), str(dest))

    cfg = dest_dir / "config.json"
    if not cfg.is_file():
        raise RuntimeError(
            f"Hunyuan distill bundle missing transformer/config.json under {root}. "
            "Install hunyuan-video-1.5-shared encoders first."
        )

    index_path = root / "model_index.json"
    payload: dict[str, object] = {}
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "_hunyuan_distill_variant": variant,
            "_danqing_bundle_source": "lightx2v_distill",
        }
    )
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
