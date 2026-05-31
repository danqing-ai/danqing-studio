"""Constrained 5Hz LM generation helpers (MLX + CUDA)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from backend.engine.families.ace_step.constrained_lm import MetadataConstrainedLogitsProcessor
from backend.engine.families.ace_step.lm_constants import DURATION_MAX, DURATION_MIN

logger = logging.getLogger(__name__)


def resolve_hf_tokenizer(tokenizer: Any) -> Any:
    """Return the underlying HF tokenizer (mlx-lm ``TokenizerWrapper`` is not callable)."""
    return getattr(tokenizer, "_tokenizer", tokenizer)


def mlx_logits_to_torch(logits_mx: Any, *, eval_fn: Any | None = None) -> Any:
    """Convert MLX logits (often bfloat16) to torch float32 for constrained processors."""
    import mlx.core as mx
    import torch

    if eval_fn is None:
        eval_fn = mx.eval
    f32 = logits_mx.astype(mx.float32)
    eval_fn(f32)
    return torch.from_numpy(np.asarray(f32))


@dataclass(frozen=True)
class ConstrainedGenerationConfig:
    temperature: float = 0.85
    top_k: Optional[int] = None
    top_p: Optional[float] = 0.9
    repetition_penalty: float = 1.0
    target_duration: Optional[float] = None
    generation_phase: str = "understand"
    user_metadata: Optional[dict[str, Any]] = None
    stop_at_reasoning: bool = False
    skip_genres: bool = False
    skip_caption: bool = False
    skip_language: bool = False
    use_constrained_decoding: bool = True
    debug: bool = False
    max_new_tokens: Optional[int] = None
    cfg_scale: float = 1.0
    uncond_prompt: str = ""


def create_constrained_processor(
    tokenizer: Any,
    *,
    max_duration: int = DURATION_MAX,
) -> MetadataConstrainedLogitsProcessor:
    return MetadataConstrainedLogitsProcessor(
        tokenizer=tokenizer,
        enabled=True,
        debug=False,
        max_duration=max_duration,
    )


def compute_max_new_tokens(
    *,
    target_duration: Optional[float],
    generation_phase: str,
    fallback_max: int = 1024,
    max_model_len: int = 4096,
) -> int:
    if target_duration is not None and target_duration > 0:
        effective = max(DURATION_MIN, min(DURATION_MAX, float(target_duration)))
        target_codes = int(effective * 5)
        if generation_phase == "codes":
            return min(max_model_len - 64, target_codes + 10)
        return min(max_model_len - 64, target_codes + 500)
    cap = DURATION_MAX * 5 + 500
    return min(fallback_max, cap, max_model_len - 64)


def configure_constrained_processor(
    processor: MetadataConstrainedLogitsProcessor,
    cfg: ConstrainedGenerationConfig,
) -> None:
    processor.reset()
    processor.enabled = cfg.use_constrained_decoding
    processor.debug = cfg.debug
    processor.metadata_temperature = None
    processor.codes_temperature = None
    processor.set_target_duration(cfg.target_duration)
    processor.set_user_metadata(cfg.user_metadata)
    processor.set_stop_at_reasoning(cfg.stop_at_reasoning)
    processor.set_skip_genres(cfg.skip_genres)
    processor.set_skip_caption(cfg.skip_caption)
    processor.set_skip_language(cfg.skip_language)
    processor.set_generation_phase(cfg.generation_phase)


def _apply_top_k(logits: Any, top_k: Optional[int]) -> Any:
    import torch

    if top_k is None or top_k <= 0:
        return logits
    values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
    min_val = values[:, -1].unsqueeze(-1)
    return torch.where(logits < min_val, torch.full_like(logits, float("-inf")), logits)


def _apply_top_p(logits: Any, top_p: Optional[float]) -> Any:
    import torch

    if top_p is None or top_p >= 1.0 or top_p <= 0.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    mask = cumulative > top_p
    mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    out = torch.full_like(logits, float("-inf"))
    return out.scatter(1, sorted_idx, sorted_logits)


def _sample_token(logits: Any, temperature: float) -> int:
    import torch

    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())
    scaled = logits / max(temperature, 1e-5)
    probs = torch.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def _combine_cfg_logits(cond_logits: Any, uncond_logits: Any, cfg_scale: float) -> Any:
    return uncond_logits + cfg_scale * (cond_logits - uncond_logits)


def generate_constrained_cuda(
    model: Any,
    tokenizer: Any,
    prompt: str,
    processor: MetadataConstrainedLogitsProcessor,
    cfg: ConstrainedGenerationConfig,
) -> str:
    import torch
    from transformers import LogitsProcessorList
    from transformers.generation.logits_processor import RepetitionPenaltyLogitsProcessor

    configure_constrained_processor(processor, cfg)
    use_cfg = cfg.cfg_scale > 1.0 and bool((cfg.uncond_prompt or "").strip())

    if use_cfg:
        return _generate_constrained_cuda_cfg(
            model, tokenizer, prompt, cfg.uncond_prompt, processor, cfg
        )

    inputs = resolve_hf_tokenizer(tokenizer)(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    max_new = cfg.max_new_tokens or compute_max_new_tokens(
        target_duration=cfg.target_duration,
        generation_phase=cfg.generation_phase,
    )
    eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    logits_processors = LogitsProcessorList()
    if cfg.repetition_penalty != 1.0:
        logits_processors.append(
            RepetitionPenaltyLogitsProcessor(penalty=cfg.repetition_penalty)
        )

    generated = input_ids.clone()
    attn = attention_mask.clone() if attention_mask is not None else torch.ones_like(input_ids)
    past = None
    use_cache = hasattr(model, "generation_config") and getattr(
        model.generation_config, "use_cache", True
    )

    with torch.inference_mode():
        for _ in range(max_new):
            outputs = model(
                input_ids=generated if past is None else generated[:, -1:],
                attention_mask=attn,
                past_key_values=past,
                use_cache=use_cache,
            )
            next_logits = outputs.logits[:, -1, :]
            if processor is not None and cfg.use_constrained_decoding:
                next_logits = processor(generated, next_logits)
            for lp in logits_processors:
                next_logits = lp(generated, next_logits)
            next_logits = _apply_top_k(next_logits, cfg.top_k)
            next_logits = _apply_top_p(next_logits, cfg.top_p)
            token_id = _sample_token(next_logits, cfg.temperature)
            if processor is not None and cfg.use_constrained_decoding:
                processor.update_state(token_id)
            next_t = torch.tensor([[token_id]], device=device, dtype=generated.dtype)
            generated = torch.cat([generated, next_t], dim=1)
            attn = torch.cat(
                [attn, torch.ones((attn.shape[0], 1), device=device, dtype=attn.dtype)],
                dim=1,
            )
            if use_cache and hasattr(outputs, "past_key_values"):
                past = outputs.past_key_values
            if token_id in {eos_id, pad_id}:
                break

    new_ids = generated[0, input_ids.shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=False)


def _generate_constrained_cuda_cfg(
    model: Any,
    tokenizer: Any,
    cond_prompt: str,
    uncond_prompt: str,
    processor: MetadataConstrainedLogitsProcessor,
    cfg: ConstrainedGenerationConfig,
) -> str:
    """Codes-phase CFG: batched cond/uncond forward (upstream ``lm_cfg_scale``)."""
    import torch

    device = next(model.parameters()).device
    pad_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        batch = resolve_hf_tokenizer(tokenizer)(
            [cond_prompt, uncond_prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
    finally:
        tokenizer.padding_side = pad_side

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    max_new = cfg.max_new_tokens or compute_max_new_tokens(
        target_duration=cfg.target_duration,
        generation_phase=cfg.generation_phase,
    )
    eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    generated = input_ids.clone()
    attn = (
        attention_mask.clone()
        if attention_mask is not None
        else torch.ones_like(input_ids)
    )
    past = None
    use_cache = hasattr(model, "generation_config") and getattr(
        model.generation_config, "use_cache", True
    )
    rep_penalty = float(cfg.repetition_penalty)

    with torch.inference_mode():
        for _ in range(max_new):
            step_in = generated if past is None else generated[:, -1:]
            outputs = model(
                input_ids=step_in,
                attention_mask=attn,
                past_key_values=past,
                use_cache=use_cache,
            )
            step_logits = outputs.logits[:, -1, :]
            cond_logits = step_logits[0:1]
            uncond_logits = step_logits[1:2]
            combined = _combine_cfg_logits(cond_logits, uncond_logits, cfg.cfg_scale)
            if processor is not None and cfg.use_constrained_decoding:
                combined = processor(generated[0:1], combined)
            if rep_penalty != 1.0 and generated.shape[1] > 0:
                idx = generated[0]
                selected = combined[:, idx]
                modified = torch.where(
                    selected > 0,
                    selected / rep_penalty,
                    selected * rep_penalty,
                )
                combined[:, idx] = modified
            combined = _apply_top_k(combined, cfg.top_k)
            combined = _apply_top_p(combined, cfg.top_p)
            token_id = _sample_token(combined, cfg.temperature)
            if processor is not None and cfg.use_constrained_decoding:
                processor.update_state(token_id)
            next_col = torch.tensor([[token_id], [token_id]], device=device, dtype=generated.dtype)
            generated = torch.cat([generated, next_col], dim=1)
            attn = torch.cat(
                [attn, torch.ones((2, 1), device=device, dtype=attn.dtype)],
                dim=1,
            )
            if use_cache and hasattr(outputs, "past_key_values"):
                past = outputs.past_key_values
            if token_id in {eos_id, pad_id}:
                break

    prompt_len = input_ids.shape[1]
    new_ids = generated[0, prompt_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=False)


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
    import torch
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
        step_logits = mlx_logits_to_torch(step_logits_mx, eval_fn=eval_fn)
        ids_torch = torch.tensor([all_ids], dtype=torch.long)
        if processor is not None and cfg.use_constrained_decoding:
            step_logits = processor(ids_torch, step_logits)
        if rep_penalty != 1.0 and all_ids:
            idx = torch.tensor(all_ids, dtype=torch.long)
            selected = step_logits[:, idx]
            modified = torch.where(
                selected > 0,
                selected / rep_penalty,
                selected * rep_penalty,
            )
            step_logits[:, idx] = modified
        step_logits = _apply_top_k(step_logits, cfg.top_k)
        step_logits = _apply_top_p(step_logits, cfg.top_p)
        logprobs = torch.log_softmax(
            step_logits / max(cfg.temperature, 1e-5) if cfg.temperature > 0 else step_logits,
            dim=-1,
        )
        token_arr = sampler(mx.array(logprobs.numpy()))
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
    import torch
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
    uncond_all = list(uncond_ids)
    new_tokens: list[int] = []

    for _ in range(max_new):
        cond_step = mlx_logits_to_torch(cond_logits[:, -1, :], eval_fn=eval_fn)
        uncond_step = mlx_logits_to_torch(uncond_logits[:, -1, :], eval_fn=eval_fn)
        combined = _combine_cfg_logits(cond_step, uncond_step, cfg.cfg_scale)
        ids_torch = torch.tensor([cond_all], dtype=torch.long)
        if processor is not None and cfg.use_constrained_decoding:
            combined = processor(ids_torch, combined)
        if rep_penalty != 1.0 and cond_all:
            idx = torch.tensor(cond_all, dtype=torch.long)
            selected = combined[:, idx]
            modified = torch.where(
                selected > 0,
                selected / rep_penalty,
                selected * rep_penalty,
            )
            combined[:, idx] = modified
        combined = _apply_top_k(combined, cfg.top_k)
        combined = _apply_top_p(combined, cfg.top_p)
        logprobs = torch.log_softmax(
            combined / max(cfg.temperature, 1e-5) if cfg.temperature > 0 else combined,
            dim=-1,
        )
        token_arr = sampler(mx.array(logprobs.numpy()))
        eval_fn(token_arr)
        token_id = int(token_arr.item())
        if processor is not None and cfg.use_constrained_decoding:
            processor.update_state(token_id)
        new_tokens.append(token_id)
        cond_all.append(token_id)
        uncond_all.append(token_id)
        if token_id in {eos_id, pad_id}:
            break
        next_input = mx.array([[token_id]])
        cond_logits = model(next_input, cache=cond_cache)
        uncond_logits = model(next_input, cache=uncond_cache)
        eval_fn(cond_logits, uncond_logits)

    return tokenizer.decode(new_tokens, skip_special_tokens=False)
