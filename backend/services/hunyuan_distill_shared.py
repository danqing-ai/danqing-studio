"""Finalize HunyuanVideo-1.5 LightX2V distill bundles after shared assets are co-located."""
from __future__ import annotations

import json
from pathlib import Path

_TRANSFORMER_CONFIG = "transformer/config.json"


def finalize_hunyuan_distill_bundle(*, distill_root: Path, variant: str) -> None:
    """Require VAE + transformer config in ``distill_root``; ensure ``model_index.json``."""
    root = Path(distill_root)
    if not root.is_dir():
        raise RuntimeError(f"Hunyuan distill bundle root not found: {root}")

    vae = root / "vae"
    cfg = root / _TRANSFORMER_CONFIG
    missing: list[str] = []
    if not vae.is_dir():
        missing.append("vae/")
    if not cfg.is_file():
        missing.append(_TRANSFORMER_CONFIG)
    if missing:
        raise RuntimeError(
            f"Hunyuan distill bundle missing {', '.join(missing)} under {root}. "
            "Re-download; bundle_repos must include Tencent-Hunyuan/HunyuanVideo-1.5 encoders."
        )

    index_path = root / "model_index.json"
    if not index_path.is_file():
        index_path.write_text(
            json.dumps(
                {
                    "_class_name": "HunyuanVideo15Pipeline",
                    "_hunyuan_distill_variant": variant,
                    "_danqing_bundle_source": "lightx2v_distill",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
