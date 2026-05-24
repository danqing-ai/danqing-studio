"""Optional semantic alignment scoring for benchmark sanity cases."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

MediaType = Literal["image", "video", "audio"]

_MODEL_CACHE: dict[tuple[str, str], tuple[object, object, object]] = {}


def _require_torch():
    try:
        import torch
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError(f"semantic_gate_missing_torch:{e!r}") from e
    return torch


def _load_clip(model_id: str) -> tuple[object, object, object]:
    key = ("clip", model_id)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    try:
        from transformers import CLIPModel, CLIPProcessor
    except Exception as e:
        raise RuntimeError(f"semantic_gate_missing_clip_deps:{e!r}") from e

    torch = _require_torch()
    try:
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id)
    except Exception as e:
        raise RuntimeError(f"semantic_gate_clip_model_load_failed(model={model_id}):{e!r}") from e
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    _MODEL_CACHE[key] = (model, processor, device)
    return model, processor, device


def _load_clap(model_id: str) -> tuple[object, object, object]:
    key = ("clap", model_id)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    try:
        from transformers import AutoProcessor, ClapModel
    except Exception as e:
        raise RuntimeError(f"semantic_gate_missing_clap_deps:{e!r}") from e

    torch = _require_torch()
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = ClapModel.from_pretrained(model_id)
    except Exception as e:
        raise RuntimeError(f"semantic_gate_clap_model_load_failed(model={model_id}):{e!r}") from e
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    _MODEL_CACHE[key] = (model, processor, device)
    return model, processor, device


def _read_image(path: str | Path):
    from PIL import Image

    return Image.open(path).convert("RGB")


def _read_video_center_frame(path: str | Path):
    import imageio.v3 as iio
    from PIL import Image

    meta = iio.immeta(path)
    n = int(meta.get("n_images", 0) or 0)
    if n < 1:
        n = 1
    idx = max(0, n // 2)
    arr = np.asarray(iio.imread(path, index=idx), dtype=np.uint8)
    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]
    return Image.fromarray(arr)


def _read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except Exception as e:
        raise RuntimeError(f"semantic_gate_missing_soundfile:{e!r}") from e
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    return mono, int(sr)


def _cosine_to_score(x: float) -> float:
    return float(np.clip((x + 1.0) * 50.0, 0.0, 100.0))


def score_image_text(path: str | Path, prompt: str, model_id: str = "openai/clip-vit-base-patch32") -> float:
    model, processor, device = _load_clip(model_id)
    torch = _require_torch()
    image = _read_image(path)
    with torch.no_grad():
        inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)
        out = model(**inputs)
        txt = out.text_embeds
        img = out.image_embeds
        txt = txt / (txt.norm(dim=-1, keepdim=True) + 1e-12)
        img = img / (img.norm(dim=-1, keepdim=True) + 1e-12)
        sim = float((txt * img).sum(dim=-1).mean().item())
    return _cosine_to_score(sim)


def score_video_text(path: str | Path, prompt: str, model_id: str = "openai/clip-vit-base-patch32") -> float:
    model, processor, device = _load_clip(model_id)
    torch = _require_torch()
    frame = _read_video_center_frame(path)
    with torch.no_grad():
        inputs = processor(text=[prompt], images=frame, return_tensors="pt", padding=True).to(device)
        out = model(**inputs)
        txt = out.text_embeds
        img = out.image_embeds
        txt = txt / (txt.norm(dim=-1, keepdim=True) + 1e-12)
        img = img / (img.norm(dim=-1, keepdim=True) + 1e-12)
        sim = float((txt * img).sum(dim=-1).mean().item())
    return _cosine_to_score(sim)


def score_audio_text(path: str | Path, prompt: str, model_id: str = "laion/clap-htsat-unfused") -> float:
    model, processor, device = _load_clap(model_id)
    torch = _require_torch()
    audio, sr = _read_audio(path)
    with torch.no_grad():
        inputs = processor(
            text=[prompt],
            audios=[audio],
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        ).to(device)
        out = model(**inputs)
        txt = out.text_embeds
        aud = out.audio_embeds
        txt = txt / (txt.norm(dim=-1, keepdim=True) + 1e-12)
        aud = aud / (aud.norm(dim=-1, keepdim=True) + 1e-12)
        sim = float((txt * aud).sum(dim=-1).mean().item())
    return _cosine_to_score(sim)


def score_semantic_alignment(
    *,
    media: MediaType,
    path: str | Path,
    prompt: str,
    backend: str = "",
    model_id: str = "",
) -> float:
    media = media.strip().lower()
    backend = (backend or "").strip().lower()
    if media in ("image", "video"):
        use_backend = backend or "clip"
        if use_backend != "clip":
            raise RuntimeError(f"semantic_gate_unsupported_backend(media={media},backend={use_backend})")
        clip_id = model_id or "openai/clip-vit-base-patch32"
        if media == "image":
            return score_image_text(path, prompt, model_id=clip_id)
        return score_video_text(path, prompt, model_id=clip_id)
    if media == "audio":
        use_backend = backend or "clap"
        if use_backend != "clap":
            raise RuntimeError(f"semantic_gate_unsupported_backend(media={media},backend={use_backend})")
        clap_id = model_id or "laion/clap-htsat-unfused"
        return score_audio_text(path, prompt, model_id=clap_id)
    raise RuntimeError(f"semantic_gate_unknown_media:{media}")
