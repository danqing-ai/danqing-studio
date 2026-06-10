"""Depth Pro structural guide — CUDA path (torch + transformers)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def estimate_depth_rgb01_cuda(
    pil: Image.Image,
    *,
    width: int,
    height: int,
    depth_bundle_root: Path,
    on_log: Any = None,
) -> np.ndarray:
    """Return float01 RGB depth visualization ``[H,W,3]`` (BFL DepthImageEncoder style)."""
    import torch

    if not depth_bundle_root.is_dir():
        raise RuntimeError(
            f"depth-pro bundle not found at {depth_bundle_root}; "
            "install depth-pro from Models → Tools"
        )

    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    if pil.size != (width, height):
        pil = pil.resize((width, height), Image.Resampling.LANCZOS)

    processor, model, device = _load_depth_pro(depth_bundle_root)
    inputs = processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        depth = outputs.predicted_depth
        if depth.ndim == 4:
            depth = depth[:, 0]
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)

    d = depth[0].float().cpu().numpy()
    d = d - float(d.min())
    mx = float(d.max())
    if mx > 1e-8:
        d = d / mx
    rgb = np.stack([d, d, d], axis=-1).astype(np.float32)
    if on_log:
        on_log("info", f"depth_estimate depth-pro bundle={depth_bundle_root.name} size={width}x{height}")
    return rgb


def _load_depth_pro(bundle_root: Path) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    device = torch.device("cpu")
    processor = AutoImageProcessor.from_pretrained(str(bundle_root), local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(
        str(bundle_root),
        local_files_only=True,
    )
    model.eval()
    model.to(device)
    return processor, model, device
