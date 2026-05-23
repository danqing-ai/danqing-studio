"""Assemble DanQing diffusers-style Hunyuan bundles from ModelScope native trees."""
from __future__ import annotations

import json
import shutil
from pathlib import Path


def assemble_hunyuan_modelscope_bundle(
    bundle_root: Path,
    variant: str,
) -> None:
    """Hoist ``transformer/<variant>/`` to diffusers ``transformer/`` layout."""
    root = Path(bundle_root)
    if not variant:
        raise RuntimeError("hunyuan_ms_variant is required for ModelScope HunyuanVideo assembly.")

    native = root / "transformer" / variant
    if not native.is_dir():
        raise RuntimeError(
            f"ModelScope HunyuanVideo bundle missing transformer/{variant}/ under {root}. "
            f"Check allow_patterns and repo_id Tencent-Hunyuan/HunyuanVideo-1.5."
        )

    flat = root / "transformer"
    staging = root / "_hunyuan_ms_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    for item in native.iterdir():
        dest = staging / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    if flat.exists():
        shutil.rmtree(flat)
    staging.rename(flat)

    # Drop leftover native variant subdirs if a full repo was downloaded.
    for sub in list(flat.iterdir()):
        if sub.is_dir() and sub.name in {
            "480p_t2v", "480p_i2v", "480p_i2v_step_distilled", "480p_i2v_distilled",
            "480p_t2v_distilled", "720p_t2v", "720p_i2v", "1080p_sr_distilled",
            "720p_sr_distilled", "720p_i2v_distilled_sparse",
        }:
            shutil.rmtree(sub)

    if not (flat / "config.json").is_file():
        raise RuntimeError(
            f"Assembled HunyuanVideo transformer/ missing config.json under {root} "
            f"(variant={variant!r})."
        )

    index_path = root / "model_index.json"
    if not index_path.is_file():
        index_path.write_text(
            json.dumps(
                {
                    "_class_name": "HunyuanVideo15Pipeline",
                    "_hunyuan_ms_variant": variant,
                    "_danqing_bundle_source": "modelscope",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
