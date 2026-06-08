"""LM log-prob quality scoring — CUDA path."""
from __future__ import annotations

import math
from typing import Any

from backend.engine.families.ace_step.lm.constrained_generate import resolve_hf_tokenizer
from backend.engine.families.ace_step.quality.pmi_scoring import PmiAssessment


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

    score = max(0.0, min(1.0, (mean_lp + 4.0) / 4.0))
    if math.isnan(score):
        return PmiAssessment(True, None, mean_lp, "nan_score")
    return PmiAssessment(True, score, mean_lp, "ok")
