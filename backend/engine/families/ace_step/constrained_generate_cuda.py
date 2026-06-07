"""Constrained 5Hz LM generation — CUDA / PyTorch path."""
from __future__ import annotations

from typing import Any, Optional

from backend.engine.families.ace_step.constrained_generate import (
    ConstrainedGenerationConfig,
    _combine_cfg_logits,
    compute_max_new_tokens,
    configure_constrained_processor,
    resolve_hf_tokenizer,
)
from backend.engine.families.ace_step.constrained_lm import MetadataConstrainedLogitsProcessor


def _apply_processor_cuda(
    processor: MetadataConstrainedLogitsProcessor,
    input_ids: Any,
    scores: Any,
) -> Any:
    import torch

    scores_np = scores.detach().cpu().float().numpy()
    input_ids_np = input_ids.detach().cpu().long().numpy()
    out_np = processor(input_ids_np, scores_np)
    return torch.from_numpy(out_np).to(device=scores.device, dtype=scores.dtype)


def _apply_top_k_torch(logits: Any, top_k: Optional[int]) -> Any:
    import torch

    if top_k is None or top_k <= 0:
        return logits
    values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
    min_val = values[:, -1].unsqueeze(-1)
    return torch.where(logits < min_val, torch.full_like(logits, float("-inf")), logits)


def _apply_top_p_torch(logits: Any, top_p: Optional[float]) -> Any:
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


def _sample_token_torch(logits: Any, temperature: float) -> int:
    import torch

    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())
    scaled = logits / max(temperature, 1e-5)
    probs = torch.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


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
                next_logits = _apply_processor_cuda(processor, generated, next_logits)
            for lp in logits_processors:
                next_logits = lp(generated, next_logits)
            next_logits = _apply_top_k_torch(next_logits, cfg.top_k)
            next_logits = _apply_top_p_torch(next_logits, cfg.top_p)
            token_id = _sample_token_torch(next_logits, cfg.temperature)
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
                combined = _apply_processor_cuda(processor, generated[0:1], combined)
            if rep_penalty != 1.0 and generated.shape[1] > 0:
                idx = generated[0]
                selected = combined[:, idx]
                modified = torch.where(
                    selected > 0,
                    selected / rep_penalty,
                    selected * rep_penalty,
                )
                combined[:, idx] = modified
            combined = _apply_top_k_torch(combined, cfg.top_k)
            combined = _apply_top_p_torch(combined, cfg.top_p)
            token_id = _sample_token_torch(combined, cfg.temperature)
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
