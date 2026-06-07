"""LM log-prob quality scoring — MLX path."""
from __future__ import annotations

import math
from typing import Any

from backend.engine.families.ace_step.constrained_generate import resolve_hf_tokenizer
from backend.engine.families.ace_step.pmi_scoring import PmiAssessment


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
