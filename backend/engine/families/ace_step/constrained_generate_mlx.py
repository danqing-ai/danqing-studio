"""Constrained 5Hz LM generation — MLX path (no PyTorch)."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from backend.engine.families.ace_step.constrained_generate import (
    ConstrainedGenerationConfig,
    _combine_cfg_logits,
    compute_max_new_tokens,
    configure_constrained_processor,
    resolve_hf_tokenizer,
)
from backend.engine.families.ace_step.constrained_lm import MetadataConstrainedLogitsProcessor


def mlx_logits_to_numpy(logits_mx: Any, *, eval_fn: Any | None = None) -> np.ndarray:
    """Convert MLX logits (often bfloat16) to float32 numpy for constrained processors."""
    import mlx.core as mx

    if eval_fn is None:
        eval_fn = mx.eval
    f32 = logits_mx.astype(mx.float32)
    eval_fn(f32)
    return np.asarray(f32, dtype=np.float32)


def _log_softmax_np(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits if temperature <= 0 else logits / max(temperature, 1e-5)
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return log_probs.astype(np.float32)


def _apply_rep_penalty_np(
    scores: np.ndarray,
    token_ids: list[int],
    penalty: float,
) -> np.ndarray:
    if penalty == 1.0 or not token_ids:
        return scores
    out = scores.copy()
    idx = np.asarray(token_ids, dtype=np.int64)
    selected = out[:, idx]
    modified = np.where(selected > 0, selected / penalty, selected * penalty)
    out[:, idx] = modified
    return out


def _apply_top_k_np(logits: np.ndarray, top_k: Optional[int]) -> np.ndarray:
    if top_k is None or top_k <= 0:
        return logits
    k = min(top_k, logits.shape[-1])
    values = np.partition(logits, -k, axis=-1)[..., -k:]
    min_val = np.min(values, axis=-1, keepdims=True)
    return np.where(logits < min_val, float("-inf"), logits)


def _apply_top_p_np(logits: np.ndarray, top_p: Optional[float]) -> np.ndarray:
    if top_p is None or top_p >= 1.0 or top_p <= 0.0:
        return logits
    sorted_idx = np.argsort(-logits, axis=-1)
    sorted_logits = np.take_along_axis(logits, sorted_idx, axis=-1)
    shifted = sorted_logits - np.max(sorted_logits, axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    cumulative = np.cumsum(probs, axis=-1)
    mask = cumulative > top_p
    mask[..., 0] = False
    sorted_logits = np.where(mask, float("-inf"), sorted_logits)
    out = np.full_like(logits, float("-inf"))
    np.put_along_axis(out, sorted_idx, sorted_logits, axis=-1)
    return out


def generate_constrained_mlx(
    model: Any,
    tokenizer: Any,
    prompt: str,
    processor: MetadataConstrainedLogitsProcessor,
    cfg: ConstrainedGenerationConfig,
    *,
    eval_fn: Any | None = None,
) -> str:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    if eval_fn is None:
        eval_fn = mx.eval

    configure_constrained_processor(processor, cfg)
    use_cfg = cfg.cfg_scale > 1.0 and bool((cfg.uncond_prompt or "").strip())

    if use_cfg:
        return _generate_constrained_mlx_cfg(
            model,
            tokenizer,
            prompt,
            cfg.uncond_prompt,
            processor,
            cfg,
            eval_fn=eval_fn,
        )

    inputs = resolve_hf_tokenizer(tokenizer)(prompt, return_tensors="np")
    prompt_ids = list(inputs["input_ids"][0].astype(np.int32))
    prompt_mx = mx.array(inputs["input_ids"][0].astype(np.int32))

    max_new = cfg.max_new_tokens or compute_max_new_tokens(
        target_duration=cfg.target_duration,
        generation_phase=cfg.generation_phase,
    )
    eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id
    pad_id = tokenizer.pad_token_id or eos_id
    sampler = make_sampler(
        temp=cfg.temperature if cfg.temperature > 0 else 0.0,
        top_p=cfg.top_p if cfg.top_p is not None and 0.0 < cfg.top_p < 1.0 else 1.0,
        top_k=cfg.top_k if cfg.top_k is not None and cfg.top_k > 0 else 0,
    )

    cache = make_prompt_cache(model)
    remaining = prompt_mx
    while len(remaining) > 1:
        chunk = remaining[:-1]
        logits = model(chunk[None], cache=cache)
        eval_fn(logits)
        remaining = remaining[-1:]

    logits = model(remaining[None], cache=cache)
    eval_fn(logits)

    all_ids = list(prompt_ids)
    new_tokens: list[int] = []
    rep_penalty = float(cfg.repetition_penalty)

    for _ in range(max_new):
        step_logits_mx = logits[:, -1, :]
        step_logits = mlx_logits_to_numpy(step_logits_mx, eval_fn=eval_fn)
        ids_np = np.asarray([all_ids], dtype=np.int64)
        if processor is not None and cfg.use_constrained_decoding:
            step_logits = processor(ids_np, step_logits)
        step_logits = _apply_rep_penalty_np(step_logits, all_ids, rep_penalty)
        step_logits = _apply_top_k_np(step_logits, cfg.top_k)
        step_logits = _apply_top_p_np(step_logits, cfg.top_p)
        logprobs = _log_softmax_np(step_logits, cfg.temperature)
        token_arr = sampler(mx.array(logprobs))
        eval_fn(token_arr)
        token_id = int(token_arr.item())
        if processor is not None and cfg.use_constrained_decoding:
            processor.update_state(token_id)
        new_tokens.append(token_id)
        all_ids.append(token_id)
        if token_id in {eos_id, pad_id}:
            break
        next_input = mx.array([[token_id]])
        logits = model(next_input, cache=cache)
        eval_fn(logits)

    return tokenizer.decode(new_tokens, skip_special_tokens=False)


def _prefill_mlx_prompt(model: Any, prompt_mx: Any, cache: Any, eval_fn: Any) -> Any:
    remaining = prompt_mx
    while len(remaining) > 1:
        logits = model(remaining[:-1][None], cache=cache)
        eval_fn(logits)
        remaining = remaining[-1:]
    logits = model(remaining[None], cache=cache)
    eval_fn(logits)
    return logits


def _generate_constrained_mlx_cfg(
    model: Any,
    tokenizer: Any,
    cond_prompt: str,
    uncond_prompt: str,
    processor: MetadataConstrainedLogitsProcessor,
    cfg: ConstrainedGenerationConfig,
    *,
    eval_fn: Any,
) -> str:
    """Codes-phase CFG with dual MLX caches (interleaved cond/uncond decode)."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    hf_tok = resolve_hf_tokenizer(tokenizer)
    cond_inputs = hf_tok(cond_prompt, return_tensors="np")
    uncond_inputs = hf_tok(uncond_prompt, return_tensors="np")
    cond_ids = list(cond_inputs["input_ids"][0].astype(np.int32))
    uncond_ids = list(uncond_inputs["input_ids"][0].astype(np.int32))
    cond_mx = mx.array(cond_inputs["input_ids"][0].astype(np.int32))
    uncond_mx = mx.array(uncond_inputs["input_ids"][0].astype(np.int32))

    max_new = cfg.max_new_tokens or compute_max_new_tokens(
        target_duration=cfg.target_duration,
        generation_phase=cfg.generation_phase,
    )
    eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id
    pad_id = tokenizer.pad_token_id or eos_id
    sampler = make_sampler(
        temp=cfg.temperature if cfg.temperature > 0 else 0.0,
        top_p=cfg.top_p if cfg.top_p is not None and 0.0 < cfg.top_p < 1.0 else 1.0,
        top_k=cfg.top_k if cfg.top_k is not None and cfg.top_k > 0 else 0,
    )
    rep_penalty = float(cfg.repetition_penalty)

    cond_cache = make_prompt_cache(model)
    uncond_cache = make_prompt_cache(model)
    cond_logits = _prefill_mlx_prompt(model, cond_mx, cond_cache, eval_fn)
    uncond_logits = _prefill_mlx_prompt(model, uncond_mx, uncond_cache, eval_fn)

    cond_all = list(cond_ids)
    new_tokens: list[int] = []

    for _ in range(max_new):
        cond_step = mlx_logits_to_numpy(cond_logits[:, -1, :], eval_fn=eval_fn)
        uncond_step = mlx_logits_to_numpy(uncond_logits[:, -1, :], eval_fn=eval_fn)
        combined = _combine_cfg_logits(cond_step, uncond_step, cfg.cfg_scale)
        ids_np = np.asarray([cond_all], dtype=np.int64)
        if processor is not None and cfg.use_constrained_decoding:
            combined = processor(ids_np, combined)
        combined = _apply_rep_penalty_np(combined, cond_all, rep_penalty)
        combined = _apply_top_k_np(combined, cfg.top_k)
        combined = _apply_top_p_np(combined, cfg.top_p)
        logprobs = _log_softmax_np(combined, cfg.temperature)
        token_arr = sampler(mx.array(logprobs))
        eval_fn(token_arr)
        token_id = int(token_arr.item())
        if processor is not None and cfg.use_constrained_decoding:
            processor.update_state(token_id)
        new_tokens.append(token_id)
        cond_all.append(token_id)
        if token_id in {eos_id, pad_id}:
            break
        next_input = mx.array([[token_id]])
        cond_logits = model(next_input, cache=cond_cache)
        uncond_logits = model(next_input, cache=uncond_cache)
        eval_fn(cond_logits, uncond_logits)

    return tokenizer.decode(new_tokens, skip_special_tokens=False)
