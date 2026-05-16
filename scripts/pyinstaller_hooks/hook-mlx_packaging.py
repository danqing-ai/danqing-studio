"""Drop CUDA / PyTorch when ``DANQING_PYINSTALLER_PROFILE=mlx`` (default on macOS)."""

from __future__ import annotations

import os
import sys

excludedimports: list[str] = []

if os.environ.get("DANQING_PYINSTALLER_PROFILE", "").lower() == "mlx" or (
    sys.platform == "darwin"
    and os.environ.get("DANQING_PYINSTALLER_PROFILE", "").lower() != "full"
):
    excludedimports = [
        "torch",
        "torchvision",
        "torchaudio",
        "cv2",
        "pyarrow",
        "datasets",
        "pandas",
        "matplotlib",
        "scipy",
        "backend.engine.runtime.cuda",
        "backend.engine.common.text_encoders.t5_cuda",
        "backend.engine.common.text_encoders.clip_cuda",
        "backend.engine.common.text_encoders.qwen25vl_cuda",
        "backend.engine.families.z_image.text_encoder_cuda",
        "backend.engine.families.ace_step.transformer_cuda",
        "backend.engine.families.ace_step.vae_cuda",
        "hf_xet",
        "soundfile",
        "backend.engine.pipelines.music_pipeline",
        "backend.engine.families.ace_step",
    ]
