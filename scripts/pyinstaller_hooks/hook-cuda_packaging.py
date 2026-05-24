"""Drop MLX when ``DANQING_PYINSTALLER_PROFILE=cuda`` (Linux/Windows CUDA bundles)."""

from __future__ import annotations

import os
import sys

excludedimports: list[str] = []

_profile = os.environ.get("DANQING_PYINSTALLER_PROFILE", "").strip().lower()
if _profile in ("cuda", "full") or (
    sys.platform != "darwin" and _profile != "mlx"
):
    excludedimports = [
        "mlx",
        "mlx.core",
        "mlx.nn",
        "mlx_lm",
        "backend.engine.runtime.mlx",
        "backend.engine.families.heartmula.generation_mlx",
        "backend.engine.families.heartmula.weights_mlx",
        "backend.engine.families.heartmula.mlx",
        "backend.engine.families.seedvr2.video_restore_mlx",
        "backend.engine.families.wan.transformer_mlx",
        "backend.engine.families.wan.vae_mlx",
        "backend.engine.families.wan.text_encoder_mlx",
    ]
