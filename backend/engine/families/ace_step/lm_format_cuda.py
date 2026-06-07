"""
ACE-Step 5Hz LM — caption/metadata expansion (CUDA / PyTorch path).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from backend.engine.families.ace_step.constrained_generate import (
    ConstrainedGenerationConfig,
    create_constrained_processor,
    validate_constrained_processor,
)
from backend.engine.families.ace_step.constrained_generate_cuda import generate_constrained_cuda
from backend.engine.families.ace_step.lm_format import (
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


def build_audio_codes_tensor(
    indices: tuple[int, ...] | list[int],
    *,
    device: Any,
    max_codes: Optional[int] = None,
) -> Any:
    """Build upstream DiT ``audio_codes`` indices tensor ``[1, T, 1]``."""
    import torch

    use = list(indices)
    if max_codes is not None and max_codes > 0:
        use = use[:max_codes]
    if not use:
        raise ValueError("audio_code indices must be non-empty")
    tensor = torch.tensor(use, dtype=torch.long, device=device)
    return tensor.reshape(1, -1, 1)


class AceStepLmFormatterCuda:
    """Expand caption/lyrics via 5Hz LM on CUDA (format_sample / Custom path only)."""

    def __init__(
        self,
        lm_dir: Path,
        device: Any = None,
        *,
        max_duration: int = 600,
    ):
        import torch

        self._lm_dir = Path(lm_dir)
        self._device = device or torch.device("cpu")
        self._max_duration = max_duration
        self._model: Any = None
        self._tokenizer: Any = None
        self._processor: Any = None
        self.last_pmi: Any = None

    @classmethod
    def from_bundle(
        cls,
        bundle_root: Path,
        device: Any = None,
        *,
        lm_dir: Optional[Path] = None,
        max_duration: int = 600,
    ) -> Optional["AceStepLmFormatterCuda"]:
        resolved = resolve_lm_dir(bundle_root, preferred=lm_dir)
        if resolved is None:
            return None
        return cls(
            resolved,
            device=device,
            max_duration=max_duration,
        )

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._model is not None:
            return

        logger.info("Loading ACE-Step 5Hz LM (CUDA) from %s on %s", self._lm_dir, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(str(self._lm_dir))
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self._lm_dir),
            torch_dtype=torch.float32,
        )
        self._model.eval()
        self._model.to(self._device)
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
        output_text = generate_constrained_cuda(
            self._model,
            self._tokenizer,
            prompt,
            self._processor,
            cfg,
        )
        self.last_pmi = self._maybe_score_pmi(prompt, output_text)
        return output_text

    def _maybe_score_pmi(self, prompt: str, output_text: str) -> Any:
        from backend.engine.families.ace_step.pmi_scoring import pmi_scoring_enabled
        from backend.engine.families.ace_step.pmi_scoring_cuda import score_lm_output_cuda

        if not pmi_scoring_enabled():
            return None
        return score_lm_output_cuda(
            self._model,
            self._tokenizer,
            prompt,
            output_text,
            device=self._device,
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
                f"ACE-Step LM planner codes phase failed (CUDA format_sample): {exc}"
            ) from exc
        logger.info(
            "ACE-Step LM format (CUDA): caption=%r -> %r (bpm=%s, codes=%d)",
            caption_in[:60],
            result.caption[:80],
            result.bpm,
            len(result.audio_code_indices),
        )
        return result
