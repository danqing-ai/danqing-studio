"""FLUX.1 Redux — SigLIP + redux MLP on CUDA (MLX path: ``redux_encode_mlx``)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def encode_redux_context_tokens_cuda(
    pil: Image.Image,
    *,
    redux_bundle_root: Path,
    on_log: Any = None,
) -> np.ndarray:
    """Return ``[1, seq, 4096]`` float32 tokens to concat after T5 ``txt_embeds``."""
    import torch
    import torch.nn as nn
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

    siglip_dir = redux_bundle_root / "image_encoder"
    fe_dir = redux_bundle_root / "feature_extractor"
    if not (siglip_dir / "config.json").is_file():
        siglip_dir = redux_bundle_root / "siglip"
    if not (siglip_dir / "config.json").is_file():
        raise RuntimeError(
            f"flux-redux bundle missing SigLIP vision config under {redux_bundle_root}/image_encoder"
        )
    processor_src = str(fe_dir) if (fe_dir / "preprocessor_config.json").is_file() else str(siglip_dir)
    processor = SiglipImageProcessor.from_pretrained(processor_src, local_files_only=True)
    siglip = SiglipVisionModel.from_pretrained(str(siglip_dir), local_files_only=True).to(device).eval()

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
