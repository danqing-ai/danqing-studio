"""Download cache root for bundle weight fetches (``DQ_WEIGHT_DL_CACHE``)."""
from __future__ import annotations

import os
from pathlib import Path

import platformdirs

if os.environ.get("DQ_WEIGHT_DL_CACHE"):
    DQ_WEIGHT_DL_CACHE = Path(os.environ["DQ_WEIGHT_DL_CACHE"]).resolve()
else:
    DQ_WEIGHT_DL_CACHE = Path(platformdirs.user_cache_dir(appname="danqing-studio", appauthor=False))
