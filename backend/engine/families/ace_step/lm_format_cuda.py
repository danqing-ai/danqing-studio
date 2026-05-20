"""
ACE-Step 5Hz LM — caption/metadata expansion (CUDA / PyTorch path).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from backend.engine.families.ace_step.lm_format import (
    DEFAULT_LM_REWRITE_INSTRUCTION,
    LmFormatResult,
    build_lm_format_result,
    parse_lm_output,
    resolve_lm_dir,
)

logger = logging.getLogger(__name__)


class AceStepLmFormatterCuda:
    """Expand caption/lyrics via 5Hz LM on CUDA (PyTorch)."""

    def __init__(self, lm_dir: Path, device: Any = None):
        import torch

        self._lm_dir = Path(lm_dir)
        self._device = device or torch.device("cpu")
        self._model: Any = None
        self._tokenizer: Any = None

    @classmethod
    def from_bundle(
        cls, bundle_root: Path, device: Any = None
    ) -> Optional["AceStepLmFormatterCuda"]:
        lm_dir = resolve_lm_dir(bundle_root)
        if lm_dir is None:
            return None
        return cls(lm_dir, device=device)

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

    def _build_prompt(self, caption: str, lyrics: str) -> str:
        user_content = f"# Caption\n{caption}\n\n# Lyric\n{lyrics}"
        return self._tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": f"# Instruction\n{DEFAULT_LM_REWRITE_INSTRUCTION}\n\n",
                },
                {"role": "user", "content": user_content},
            ],
            tokenize=False,
            add_generation_prompt=True,
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
        import torch

        if self._model is None:
            self.load()

        caption_in = (caption or "").strip() or "NO USER INPUT"
        lyrics_in = (lyrics or "").strip() or "[Instrumental]"
        prompt = self._build_prompt(caption_in, lyrics_in)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.85,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            )

        gen_ids = out_ids[0, inputs["input_ids"].shape[1] :]
        output_text = self._tokenizer.decode(gen_ids, skip_special_tokens=False)
        meta = parse_lm_output(output_text)
        result = build_lm_format_result(
            meta,
            caption_in=caption_in,
            lyrics_in=lyrics_in,
            duration=duration,
            bpm=bpm,
            keyscale=keyscale,
            timesignature=timesignature,
            language=language,
        )
        logger.info(
            "ACE-Step LM format (CUDA): caption=%r -> %r (bpm=%s duration=%s)",
            caption_in[:60],
            result.caption[:80],
            result.bpm,
            result.duration,
        )
        return result
