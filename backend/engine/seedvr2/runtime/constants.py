"""SeedVR2 内部推理仅用常量（原 ``cli/defaults`` 中与 Studio 超分相关的子集）。"""
from __future__ import annotations

import os
from pathlib import Path

import platformdirs

DIMENSION_STEP_PIXELS = 16

if os.environ.get("DQ_WEIGHT_DL_CACHE"):
    DQ_WEIGHT_DL_CACHE = Path(os.environ["DQ_WEIGHT_DL_CACHE"]).resolve()
else:
    DQ_WEIGHT_DL_CACHE = Path(platformdirs.user_cache_dir(appname="danqing-studio", appauthor=False))
