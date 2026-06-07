"""FLUX.1 Redux — SigLIP vision + redux MLP tokens appended to T5 context (BFL prepare_redux)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def encode_redux_context_tokens(
    pil: Image.Image,
    *,
    redux_bundle_root: Path,
    on_log: Any = None,
) -> np.ndarray:
    """Return ``[1, seq, 4096]`` float32 tokens to concat after T5 ``txt_embeds``."""
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError(
            "redux style guide requires PyTorch for SigLIP preprocessing; "
            "run: pip install torch"
        ) from exc

    from safetensors.torch import load_file as load_sft
    from transformers import SiglipImageProcessor, SiglipVisionModel

    if not redux_bundle_root.is_dir():
        raise RuntimeError(
            f"flux-redux bundle not found at {redux_bundle_root}; "
            "install flux-redux from Models → ControlNet"
        )

    redux_path = _find_redux_weights(redux_bundle_root)
    device = torch.device("cpu")

    sd = load_sft(str(redux_path), device=str(device))
    redux_dim = 1152
    txt_dim = 4096
    redux_up = nn.Linear(redux_dim, txt_dim * 3, bias=False).to(device)
    redux_down = nn.Linear(txt_dim * 3, txt_dim, bias=False).to(device)
    up_key = "redux_up.weight"
    down_key = "redux_down.weight"
    if up_key not in sd or down_key not in sd:
        for k in sd:
            if "redux_up" in k and k.endswith(".weight"):
                up_key = k
            if "redux_down" in k and k.endswith(".weight"):
                down_key = k
    if up_key not in sd or down_key not in sd:
        raise RuntimeError(
            f"flux-redux bundle missing redux_up/redux_down weights under {redux_bundle_root}"
        )
    redux_up.weight.data.copy_(sd[up_key].to(dtype=torch.float32))
    redux_down.weight.data.copy_(sd[down_key].to(dtype=torch.float32))

    siglip_name = "google/siglip-so400m-patch14-384"
    siglip_dir = redux_bundle_root / "siglip"
    if siglip_dir.is_dir():
        siglip_src = str(siglip_dir)
    else:
        siglip_src = siglip_name
    processor = SiglipImageProcessor.from_pretrained(siglip_src)
    siglip = SiglipVisionModel.from_pretrained(siglip_src).to(device).eval()

    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    inputs = processor(images=[pil], return_tensors="pt", do_resize=True, do_convert_rgb=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        hidden = siglip(**inputs).last_hidden_state.to(torch.float32)
        projected = redux_down(torch.nn.functional.silu(redux_up(hidden)))

    out = projected.cpu().numpy().astype(np.float32)
    if on_log:
        on_log(
            "info",
            f"redux_encode tokens shape={tuple(out.shape)} bundle={redux_bundle_root.name}",
        )
    return out


def _find_redux_weights(bundle_root: Path) -> Path:
    candidates = [
        bundle_root / "flux1-redux-dev.safetensors",
        bundle_root / "redux" / "diffusion_pytorch_model.safetensors",
    ]
    for p in candidates:
        if p.is_file():
            return p
    for p in sorted(bundle_root.rglob("*.safetensors")):
        if "redux" in p.name.lower() or "siglip" not in p.name.lower():
            return p
    raise RuntimeError(f"no redux safetensors under {bundle_root}")


def resolve_redux_bundle_root(registry: Any, project_root: Path, controlnet_model_id: str) -> Path:
    from backend.engine.common.pipeline_registry import local_bundle_root as bundle_root_fn
    from backend.engine.common.pipeline_registry import resolve_version_block as version_block_fn

    entry = registry.require(controlnet_model_id)
    version_key = version_block_fn(entry, None)
    root = bundle_root_fn(project_root, entry, version_key)
    if root is None:
        raise RuntimeError(f"controlnet bundle path missing for {controlnet_model_id!r}")
    return root
