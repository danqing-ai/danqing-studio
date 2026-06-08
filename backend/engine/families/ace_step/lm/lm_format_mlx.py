"""
ACE-Step 5Hz LM — MLX caption/lyrics expansion (``acestep-5Hz-lm-*`` in model bundle).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from backend.engine.runtime.mlx_runtime import load_weights_dict, run_eval
from backend.engine.families.ace_step.lm.constrained_generate import (
    ConstrainedGenerationConfig,
    create_constrained_processor,
    validate_constrained_processor,
)
from backend.engine.families.ace_step.lm.constrained_generate_mlx import generate_constrained_mlx
from backend.engine.families.ace_step.lm.lm_format import (
    DEFAULT_LM_REWRITE_INSTRUCTION,
    LmFormatResult,
    build_lm_format_result,
    compact_vocal_lyrics_structure,
    format_sample_understand_config,
    is_instrumental_lyrics,
    normalize_lyrics_body,
    parse_lm_understand_output,
    resolve_lm_dir,
    run_planner_codes_phase,
)

logger = logging.getLogger(__name__)


def _load_safetensors(load_fn: Any | None, path: Path) -> dict[str, Any]:
    return load_weights_dict(load_fn, str(path))


def load_ace_step_lm_mlx(
    lm_dir: Path,
    *,
    ctx: Any | None = None,
    quantize_bits: Optional[int] = None,
) -> tuple[Any, Any]:
    """Load Qwen3 5Hz LM with mlx-lm (checkpoint uses flat keys, not ``model.*``)."""
    from mlx_lm.models.qwen3 import Model, ModelArgs
    from mlx_lm.utils import load_tokenizer

    lm_path = Path(lm_dir)
    with open(lm_path / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["model_type"] = "qwen3"

    model = Model(ModelArgs.from_dict(cfg))
    raw = _load_safetensors(getattr(ctx, "load_weights", None), lm_path / "model.safetensors")
    weights = {
        (f"model.{key}" if not key.startswith("model.") else key): value
        for key, value in raw.items()
    }
    model.load_weights(list(weights.items()), strict=False)
    if quantize_bits in (4, 8):
        import mlx.core as mx
        import mlx.nn as nn

        logger.info("ACE-Step LM: applying %d-bit MLX quantization", quantize_bits)
        nn.quantize(model, bits=int(quantize_bits))
        run_eval(getattr(ctx, "eval", None), model.parameters())
    else:
        run_eval(getattr(ctx, "eval", None), model.parameters())
    tokenizer = load_tokenizer(str(lm_path))
    return model, tokenizer


class AceStepLmFormatterMlx:
    """Expand caption/lyrics via 5Hz LM on MLX (format_sample / Custom path only)."""

    def __init__(
        self,
        lm_dir: Path,
        *,
        ctx: Any | None = None,
        quantize_bits: Optional[int] = None,
        max_duration: int = 600,
    ):
        self._lm_dir = Path(lm_dir)
        self._ctx = ctx
        self._quantize_bits = quantize_bits
        self._max_duration = max_duration
        self._model: Any = None
        self._tokenizer: Any = None
        self._processor: Any = None
        self.last_pmi: Any = None

    @classmethod
    def from_bundle(
        cls,
        bundle_root: Path,
        *,
        ctx: Any | None = None,
        lm_dir: Optional[Path] = None,
        quantize_bits: Optional[int] = None,
        max_duration: int = 600,
    ) -> Optional["AceStepLmFormatterMlx"]:
        resolved = resolve_lm_dir(bundle_root, preferred=lm_dir)
        if resolved is None:
            return None
        return cls(
            resolved,
            ctx=ctx,
            quantize_bits=quantize_bits,
            max_duration=max_duration,
        )

    def load(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading ACE-Step 5Hz LM (MLX) from %s", self._lm_dir)
        self._model, self._tokenizer = load_ace_step_lm_mlx(
            self._lm_dir,
            ctx=self._ctx,
            quantize_bits=self._quantize_bits,
        )
        self._processor = create_constrained_processor(
            self._tokenizer,
            max_duration=self._max_duration,
        )

    def _chat_prompt(self, instruction: str, user_content: str) -> str:
        return self._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": f"# Instruction\n{instruction}\n\n"},
                {"role": "user", "content": user_content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _ensure_processor(self) -> None:
        """Rebuild processor if a stale cached formatter survived a hot reload."""
        if self._tokenizer is None:
            self.load()
            return
        if self._processor is None:
            self._processor = create_constrained_processor(
                self._tokenizer,
                max_duration=self._max_duration,
            )
            return
        try:
            validate_constrained_processor(self._processor)
        except RuntimeError:
            logger.warning(
                "ACE-Step LM constrained processor missing audio-code vocab; rebuilding"
            )
            self._processor = create_constrained_processor(
                self._tokenizer,
                max_duration=self._max_duration,
            )

    def _generate(self, prompt: str, cfg: ConstrainedGenerationConfig) -> str:
        if self._model is None:
            self.load()
        self._ensure_processor()
        output_text = generate_constrained_mlx(
            self._model,
            self._tokenizer,
            prompt,
            self._processor,
            cfg,
            eval_fn=getattr(self._ctx, "eval", None),
        )
        self.last_pmi = self._maybe_score_pmi(prompt, output_text)
        return output_text

    def _maybe_score_pmi(self, prompt: str, output_text: str) -> Any:
        from backend.engine.families.ace_step.quality.pmi_scoring import pmi_scoring_enabled
        from backend.engine.families.ace_step.quality.pmi_scoring_mlx import score_lm_output_mlx

        if not pmi_scoring_enabled():
            return None
        return score_lm_output_mlx(
            self._model,
            self._tokenizer,
            prompt,
            output_text,
            eval_fn=getattr(self._ctx, "eval", None),
        )

    def format_sample(
        self,
        *,
        caption: str,
        lyrics: str,
        duration: Optional[float] = None,
        bpm: Optional[int] = None,
        keyscale: str = "",
        timesignature: str = "",
        language: str = "",
    ) -> LmFormatResult:
        if self._model is None:
            self.load()

        caption_in = (caption or "").strip() or "NO USER INPUT"
        lyrics_in = compact_vocal_lyrics_structure((lyrics or "").strip()) or "[Instrumental]"
        user_has_vocals = not is_instrumental_lyrics(lyrics_in)
        prompt = self._chat_prompt(
            DEFAULT_LM_REWRITE_INSTRUCTION,
            f"# Caption\n{caption_in}\n\n# Lyric\n{lyrics_in}",
        )
        user_meta = None
        if language:
            user_meta = {"language": language}
        if duration is not None:
            user_meta = dict(user_meta or {})
            user_meta["duration"] = int(round(duration))
        if bpm is not None:
            user_meta = dict(user_meta or {})
            user_meta["bpm"] = int(bpm)
        if keyscale:
            user_meta = dict(user_meta or {})
            user_meta["keyscale"] = keyscale
        if timesignature:
            user_meta = dict(user_meta or {})
            user_meta["timesignature"] = timesignature

        output_text = self._generate(
            prompt,
            format_sample_understand_config(
                duration=duration,
                user_metadata=user_meta,
                metadata_only=user_has_vocals,
            ),
        )
        meta, _parsed_lyrics = parse_lm_understand_output(
            output_text,
            instrumental=not user_has_vocals,
        )
        result = build_lm_format_result(
            meta,
            caption_in=caption_in,
            lyrics_in=lyrics_in,
            duration=duration,
            bpm=bpm,
            keyscale=keyscale,
            timesignature=timesignature,
            language=language,
            want_vocals=user_has_vocals,
            preserve_user_lyrics=user_has_vocals,
        )
        try:
            result = run_planner_codes_phase(
                generate_fn=self._generate,
                tokenizer=self._tokenizer,
                result=result,
                duration_hint=duration,
            )
        except Exception as exc:
            raise RuntimeError(
                f"ACE-Step LM planner codes phase failed (MLX format_sample): {exc}"
            ) from exc
        logger.info(
            "ACE-Step LM (MLX) format: caption=%r -> %r (bpm=%s, codes=%d)",
            caption_in[:60],
            result.caption[:80],
            result.bpm,
            len(result.audio_code_indices),
        )
        return result
