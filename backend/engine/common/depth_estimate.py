"""Monocular depth maps for Flux structural (depth) control — Depth Pro bundle."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def estimate_depth_rgb01(
    pil: Image.Image,
    *,
    width: int,
    height: int,
    depth_bundle_root: Path,
    on_log: Any = None,
) -> np.ndarray:
    """Return float01 RGB depth visualization ``[H,W,3]`` (BFL DepthImageEncoder style)."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "depth structural guide requires PyTorch for Depth Pro preprocessing; "
            "run: pip install torch — or use flux-canny-controlnet instead"
        ) from exc

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


def resolve_depth_pro_bundle_root(registry: Any, project_root: Path) -> Path:
    from backend.engine.common.pipeline_registry import local_bundle_root as bundle_root_fn
    from backend.engine.common.pipeline_registry import resolve_version_block as version_block_fn

    entry = registry.require("depth-pro")
    version_key = version_block_fn(entry, None)
    root = bundle_root_fn(project_root, entry, version_key)
    if root is None:
        raise RuntimeError("depth-pro registry entry has no local bundle path")
    return root
