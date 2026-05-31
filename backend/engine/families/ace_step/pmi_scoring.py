"""Optional LM log-prob quality scoring (lightweight PMI-style heuristic)."""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PmiAssessment:
    enabled: bool
    score: Optional[float]
    mean_logprob: Optional[float]
    note: str

    def quality_bonus(self) -> float:
        if not self.enabled or self.score is None:
            return 0.0
        if self.score >= 0.75:
            return 4.0
        if self.score >= 0.55:
            return 1.5
        if self.score < 0.35:
            return -6.0
        return 0.0


def pmi_scoring_enabled() -> bool:
    return os.environ.get("ACESTEP_PMI_SCORING", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def score_lm_output_cuda(
    model: Any,
    tokenizer: Any,
    prompt: str,
    output_text: str,
    *,
    device: Any,
) -> PmiAssessment:
    """Teacher-forced mean token logprob for generated suffix (CUDA only)."""
    import torch

    if not output_text.strip():
        return PmiAssessment(False, None, None, "empty_output")

    from backend.engine.families.ace_step.constrained_generate import resolve_hf_tokenizer

    hf_tok = resolve_hf_tokenizer(tokenizer)
    full = prompt + output_text
    prompt_ids = hf_tok(prompt, return_tensors="pt").input_ids.to(device)
    full_ids = hf_tok(full, return_tensors="pt").input_ids.to(device)
    prompt_len = int(prompt_ids.shape[1])
    if full_ids.shape[1] <= prompt_len:
        return PmiAssessment(True, None, None, "no_generated_tokens")

    with torch.inference_mode():
        logits = model(full_ids).logits[:, prompt_len - 1 : -1, :]
        targets = full_ids[:, prompt_len:]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        mean_lp = float(gathered.mean().item())

    # Map mean logprob (~[-4, 0]) to 0..1
    score = max(0.0, min(1.0, (mean_lp + 4.0) / 4.0))
    if math.isnan(score):
        return PmiAssessment(True, None, mean_lp, "nan_score")
    return PmiAssessment(True, score, mean_lp, "ok")


def score_lm_output_mlx(
    model: Any,
    tokenizer: Any,
    prompt: str,
    output_text: str,
    *,
    eval_fn: Any | None = None,
) -> PmiAssessment:
    import mlx.core as mx
    import numpy as np

    if not output_text.strip():
        return PmiAssessment(False, None, None, "empty_output")

    from backend.engine.families.ace_step.constrained_generate import resolve_hf_tokenizer

    hf_tok = resolve_hf_tokenizer(tokenizer)
    prompt_ids = hf_tok(prompt, return_tensors="np")["input_ids"][0]
    full_ids = hf_tok(prompt + output_text, return_tensors="np")["input_ids"][0]
    prompt_len = int(len(prompt_ids))
    if len(full_ids) <= prompt_len:
        return PmiAssessment(True, None, None, "no_generated_tokens")

    ids = mx.array(full_ids.astype(np.int32))[None, :]
    logits = model(ids)
    if eval_fn is not None:
        eval_fn(logits)
    logits_f32 = logits[0, prompt_len - 1 : -1, :].astype(mx.float32)
    if eval_fn is not None:
        eval_fn(logits_f32)
    logits_np = np.asarray(logits_f32, dtype=np.float64)
    targets = full_ids[prompt_len:]
    max_l = logits_np.max(axis=-1, keepdims=True)
    logsum = np.log(np.exp(logits_np - max_l).sum(axis=-1)) + max_l.squeeze(-1)
    gathered = logits_np[np.arange(len(targets)), targets] - logsum
    mean_lp = float(gathered.mean())
    score = max(0.0, min(1.0, (mean_lp + 4.0) / 4.0))
    if math.isnan(score):
        return PmiAssessment(True, None, mean_lp, "nan_score")
    return PmiAssessment(True, score, mean_lp, "ok")
