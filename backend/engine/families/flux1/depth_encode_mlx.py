"""Depth Pro structural guide — MLX path (native Depth Pro)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from backend.engine.families.flux1.depth_pro_mlx import estimate_depth_pro_mlx


def estimate_depth_rgb01_mlx(
    pil: Image.Image,
    *,
    width: int,
    height: int,
    depth_bundle_root: Path,
    on_log: Any = None,
) -> np.ndarray:
    """Return float01 RGB depth visualization ``[H,W,3]`` (BFL DepthImageEncoder style)."""
    if not depth_bundle_root.is_dir():
        raise RuntimeError(
            f"depth-pro bundle not found at {depth_bundle_root}; "
            "install depth-pro from Models → Tools"
        )

    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    if pil.size != (width, height):
        pil = pil.resize((width, height), Image.Resampling.LANCZOS)

    depth = estimate_depth_pro_mlx(pil, depth_bundle_root=depth_bundle_root, on_log=on_log)
    depth_img = Image.fromarray(depth, mode="F")
    depth_img = depth_img.resize((width, height), Image.Resampling.BICUBIC)
    d = np.asarray(depth_img, dtype=np.float32)
    d = d - float(d.min())
    mx = float(d.max())
    if mx > 1e-8:
        d = d / mx
    rgb = np.stack([d, d, d], axis=-1).astype(np.float32)
    if on_log:
        on_log("info", f"depth_estimate depth-pro bundle={depth_bundle_root.name} size={width}x{height}")
    return rgb
