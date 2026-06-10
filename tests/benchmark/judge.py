"""L2 PickScore judge for image eval benchmark."""
from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .judge_assets import JUDGE_MODEL_ID_HF, JUDGE_MODEL_ID_MS, resolve_judge_model_path

JUDGE_MODEL_ID = JUDGE_MODEL_ID_MS
JUDGE_FLOOR = 0.20
GOLDEN_RELATIVE = 0.85

_MODEL: Any | None = None
_PROCESSOR: Any | None = None
_LOADED_FROM: str = ""


@dataclass
class JudgeResult:
    ok: bool
    score: float
    min_required: float
    reason: str = ""
    model_id: str = JUDGE_MODEL_ID


def reset_judge_cache() -> None:
    global _MODEL, _PROCESSOR, _LOADED_FROM
    if _MODEL is not None:
        try:
            import torch

            _MODEL.cpu()
        except Exception:
            pass
    _MODEL = None
    _PROCESSOR = None
    _LOADED_FROM = ""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _load_pickscore() -> tuple[Any, Any]:
    global _MODEL, _PROCESSOR, _LOADED_FROM
    model_ref = resolve_judge_model_path()
    if _MODEL is not None and _PROCESSOR is not None and _LOADED_FROM == model_ref:
        return _MODEL, _PROCESSOR

    try:
        from transformers import AutoModel, AutoProcessor
    except Exception as exc:
        raise RuntimeError(f"judge_missing_deps:{exc!r}") from exc

    try:
        processor = AutoProcessor.from_pretrained(model_ref)
        model = AutoModel.from_pretrained(model_ref)
    except Exception as exc:
        raise RuntimeError(f"judge_load_failed(model={model_ref}):{exc!r}") from exc

    model.eval()
    _PROCESSOR = processor
    _MODEL = model
    _LOADED_FROM = model_ref
    return model, processor


def _feature_tensor(features: Any):
    import torch

    if torch.is_tensor(features):
        return features
    pooler = getattr(features, "pooler_output", None)
    if pooler is not None:
        return pooler
    hidden = getattr(features, "last_hidden_state", None)
    if hidden is not None:
        return hidden[:, 0]
    raise RuntimeError(f"judge_unexpected_feature_type:{type(features)!r}")


def score_pickscore(prompt: str, path: str | Path) -> float:
    import torch
    from PIL import Image

    model, processor = _load_pickscore()
    text = (prompt or "").strip()
    if not text:
        raise RuntimeError("judge_empty_prompt")

    image = Image.open(path).convert("RGB")
    device = torch.device("cpu")
    model.to(device)

    with torch.no_grad():
        inputs = processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        image_features = _feature_tensor(model.get_image_features(pixel_values=inputs["pixel_values"]))
        text_features = _feature_tensor(
            model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        score = float((text_features @ image_features.T).squeeze().item())
    return score


def required_score(*, golden: float | None, floor: float | None = None) -> float:
    base = JUDGE_FLOOR if floor is None else float(floor)
    if golden is None:
        return base
    return max(base, float(golden) * GOLDEN_RELATIVE)


def judge_image(
    prompt: str,
    path: str | Path,
    *,
    golden: float | None = None,
    judge_floor: float | None = None,
) -> JudgeResult:
    score = score_pickscore(prompt, path)
    min_required = required_score(golden=golden, floor=judge_floor)
    loaded = _LOADED_FROM or resolve_judge_model_path()
    if score < min_required:
        return JudgeResult(
            ok=False,
            score=score,
            min_required=min_required,
            reason=f"low_pickscore(score={score:.3f},min={min_required:.3f})",
            model_id=loaded,
        )
    return JudgeResult(ok=True, score=score, min_required=min_required, model_id=loaded)
