"""Constrained 5Hz LM generation — shared config/helpers (no MLX / PyTorch)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from backend.engine.families.ace_step.constrained_lm import MetadataConstrainedLogitsProcessor
from backend.engine.families.ace_step.lm_constants import DURATION_MAX, DURATION_MIN

logger = logging.getLogger(__name__)


def resolve_hf_tokenizer(tokenizer: Any) -> Any:
    """Return the underlying HF tokenizer (mlx-lm ``TokenizerWrapper`` is not callable)."""
    inner = getattr(tokenizer, "_tokenizer", None)
    if inner is None:
        inner = getattr(tokenizer, "tokenizer", None)
    return inner if inner is not None else tokenizer


def validate_constrained_processor(processor: MetadataConstrainedLogitsProcessor) -> None:
    """Fail loud when ACE-Step LM audio-code vocabulary was not discovered."""
    n_codes = len(processor.audio_code_token_ids)
    if n_codes == 0 or processor.non_audio_code_mask is None:
        raise RuntimeError(
            "ACE-Step 5Hz LM constrained decoder found no <|audio_code_N|> tokens in the "
            f"tokenizer vocabulary (vocab_size={processor.vocab_size}). "
            "Unconstrained code generation produces rhythmic garbage audio. "
            "Restart the API server after upgrades, or reinstall acestep-5Hz-lm in the model bundle."
        )


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
    processor = MetadataConstrainedLogitsProcessor(
        tokenizer=tokenizer,
        enabled=True,
        debug=False,
        max_duration=max_duration,
    )
    validate_constrained_processor(processor)
    logger.info(
        "ACE-Step LM constrained vocab ready: %d audio code tokens (vocab_size=%d)",
        len(processor.audio_code_token_ids),
        processor.vocab_size,
    )
    return processor


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
    if cfg.use_constrained_decoding and cfg.generation_phase == "codes":
        validate_constrained_processor(processor)


def _combine_cfg_logits(cond_logits: Any, uncond_logits: Any, cfg_scale: float) -> Any:
    return uncond_logits + cfg_scale * (cond_logits - uncond_logits)
