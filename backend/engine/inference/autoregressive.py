"""AutoregressiveInference — Token-by-token LM 自回归解码 (L2).

ACE-Step 5Hz LM 非 CFG 路径经 ``constrained_generate_mlx`` 接入；CFG 仍走双 cache 专用循环。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from backend.engine.inference._runtime import raise_if_cancelled


@dataclass
class AutoregressiveBundle:
    """LM 自回归解码专用 bundle — 与 flow/block 音频 bundle 正交。"""

    prompt_tokens: list[int]
    max_new_tokens: int
    prefill_fn: Callable
    logits_fn: Callable
    sample_fn: Callable
    cancel_token: Any | None = None
    process_logits_fn: Callable | None = None
    eval_fn: Callable | None = None
    eos_token_ids: set[int] | None = None


class AutoregressiveInference:
    """Token-by-token 自回归解码策略（ACE-Step 5Hz LM 非 CFG 路径）。"""

    def run(self, bundle: AutoregressiveBundle) -> dict[str, Any]:
        eval_fn = bundle.eval_fn or (lambda *_: None)
        eos_ids = bundle.eos_token_ids or set()

        state = bundle.prefill_fn(bundle.prompt_tokens)
        if isinstance(state, tuple):
            logits, state = state
        else:
            logits = state
            state = None

        all_tokens: list[int] = list(bundle.prompt_tokens)
        new_tokens: list[int] = []

        for step_idx in range(bundle.max_new_tokens):
            raise_if_cancelled(bundle.cancel_token)

            if bundle.process_logits_fn is not None:
                logits = bundle.process_logits_fn(logits, all_tokens, step_idx)

            token = bundle.sample_fn(logits)
            token_id = int(token) if not isinstance(token, int) else token
            eval_fn(token)

            new_tokens.append(token_id)
            all_tokens.append(token_id)

            if token_id in eos_ids:
                break

            logits = bundle.logits_fn(token_id, state)
            eval_fn(logits)

        return {
            "tokens": new_tokens,
            "num_tokens": len(new_tokens),
            "all_tokens": all_tokens,
        }
