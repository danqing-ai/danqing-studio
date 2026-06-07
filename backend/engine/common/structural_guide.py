"""Structural guide (Flux Canny/Depth/Redux) — preprocess, registry mapping, patch-embed load."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

GuideType = Literal["canny", "depth", "redux"]

# Registry controlnet id → companion LoRA (BFL structural-conditioning LoRA path).
CONTROLNET_LORA_MAP: dict[str, str] = {
    "flux-canny-controlnet": "flux1-canny-dev-lora",
    "flux-depth-controlnet": "flux1-depth-dev-lora",
}

_GUIDE_TYPE_HINTS: tuple[tuple[str, GuideType], ...] = (
    ("depth", "depth"),
    ("redux", "redux"),
    ("canny", "canny"),
)


def infer_guide_type(model_id: str) -> GuideType:
    key = (model_id or "").strip().lower()
    for hint, gtype in _GUIDE_TYPE_HINTS:
        if hint in key:
            return gtype
    return "canny"


def is_fill_controlnet(model_id: str) -> bool:
    return "fill" in (model_id or "").strip().lower()


def is_redux_controlnet(model_id: str) -> bool:
    return "redux" in (model_id or "").strip().lower()


def companion_lora_id(controlnet_model_id: str) -> str | None:
    if is_redux_controlnet(controlnet_model_id) or is_fill_controlnet(controlnet_model_id):
        return None
    return CONTROLNET_LORA_MAP.get((controlnet_model_id or "").strip())


def preprocess_structural_rgb(
    pil: Image.Image,
    *,
    guide_type: GuideType,
    width: int,
    height: int,
    registry: Any,
    project_root: Path,
    on_log: Any = None,
) -> np.ndarray:
    """Return float01 RGB ``[H,W,3]`` ready for VAE encode (linear 0..1)."""
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    if pil.size != (width, height):
        pil = pil.resize((width, height), Image.Resampling.LANCZOS)
    rgb = np.asarray(pil, dtype=np.float32) / 255.0

    if guide_type == "canny":
        return _canny_rgb(rgb)
    if guide_type == "depth":
        from backend.engine.common.depth_estimate import (
            estimate_depth_rgb01,
            resolve_depth_pro_bundle_root,
        )

        depth_root = resolve_depth_pro_bundle_root(registry, project_root)
        return estimate_depth_rgb01(
            pil,
            width=width,
            height=height,
            depth_bundle_root=depth_root,
            on_log=on_log,
        )
    raise RuntimeError(f"preprocess_structural_rgb does not handle guide_type={guide_type!r}")


def _canny_rgb(rgb: np.ndarray, low: int = 50, high: int = 200) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "structural guide (canny) requires opencv-python-headless; "
            "pip install opencv-python-headless"
        ) from exc
    gray = cv2.cvtColor((rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    e = edges.astype(np.float32) / 255.0
    return np.stack([e, e, e], axis=-1)


def load_flux1_structural_patch_embed(
    *,
    registry: Any,
    project_root: Path,
    controlnet_model_id: str,
    ctx: Any,
    on_log: Any = None,
) -> tuple[Any, Any]:
    """Load 128-dim packed ``x_embedder`` from a controlnet bundle (Flux Canny/Depth)."""
    from backend.engine.common.pipeline_registry import local_bundle_root as bundle_root_fn
    from backend.engine.common.pipeline_registry import resolve_version_block as version_block_fn
    from backend.engine.families.flux1.weights import remap_flux1_weights

    entry = registry.require(controlnet_model_id)
    version_key = version_block_fn(entry, None)
    bundle_root = bundle_root_fn(project_root, entry, version_key)
    tp = (bundle_root / "transformer") if bundle_root else None
    if tp is None or not tp.exists():
        raise RuntimeError(
            f"structural guide requires installed controlnet bundle {controlnet_model_id!r} "
            f"(missing transformer/ under {bundle_root}); install from Models → ControlNet"
        )

    raw: dict[str, Any] = {}
    for sf in sorted(tp.glob("*.safetensors")):
        raw.update(ctx.load_weights(str(sf)))
    remapped = remap_flux1_weights(raw)
    weight = remapped.get("patch_embed.proj.weight")
    bias = remapped.get("patch_embed.proj.bias")
    if weight is None or bias is None:
        raise RuntimeError(
            f"controlnet bundle {controlnet_model_id!r} missing x_embedder "
            f"(patch_embed.proj.weight/bias)"
        )
    if hasattr(weight, "shape"):
        sh = tuple(weight.shape)
        in_ch = int(sh[-1]) if len(sh) == 4 else int(sh[1]) if len(sh) == 2 else -1
    else:
        in_ch = -1
    if in_ch != 128:
        raise RuntimeError(
            f"controlnet bundle {controlnet_model_id!r} x_embedder input dim {in_ch} "
            f"(expected 128 for Flux structural concat); wrong bundle or corrupt weights"
        )
    if on_log:
        on_log("info", f"structural_guide loaded x_embedder from {controlnet_model_id} in_ch=128")
    return weight, bias
