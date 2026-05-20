"""
ACE-Step 5Hz LM — MLX caption/lyrics expansion (``acestep-5Hz-lm-*`` in model bundle).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx

from backend.engine.families.ace_step.lm_format import (
    DEFAULT_LM_REWRITE_INSTRUCTION,
    LmFormatResult,
    build_lm_format_result,
    parse_lm_output,
    resolve_lm_dir,
)

logger = logging.getLogger(__name__)


def load_ace_step_lm_mlx(lm_dir: Path) -> tuple[Any, Any]:
    """Load Qwen3 5Hz LM with mlx-lm (checkpoint uses flat keys, not ``model.*``)."""
    from mlx_lm.models.qwen3 import Model, ModelArgs
    from mlx_lm.utils import load_tokenizer

    lm_path = Path(lm_dir)
    with open(lm_path / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["model_type"] = "qwen3"

    model = Model(ModelArgs.from_dict(cfg))
    raw = dict(mx.load(str(lm_path / "model.safetensors")))
    weights = {
        (f"model.{key}" if not key.startswith("model.") else key): value
        for key, value in raw.items()
    }
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    tokenizer = load_tokenizer(str(lm_path))
    return model, tokenizer


class AceStepLmFormatterMlx:
    """Expand caption/lyrics via 5Hz LM on MLX."""

    def __init__(self, lm_dir: Path):
        self._lm_dir = Path(lm_dir)
        self._model: Any = None
        self._tokenizer: Any = None

    @classmethod
    def from_bundle(cls, bundle_root: Path) -> Optional["AceStepLmFormatterMlx"]:
        lm_dir = resolve_lm_dir(bundle_root)
        if lm_dir is None:
            return None
        return cls(lm_dir)

    def load(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading ACE-Step 5Hz LM (MLX) from %s", self._lm_dir)
        self._model, self._tokenizer = load_ace_step_lm_mlx(self._lm_dir)

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
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        if self._model is None:
            self.load()

        caption_in = (caption or "").strip() or "NO USER INPUT"
        lyrics_in = (lyrics or "").strip() or "[Instrumental]"
        prompt = self._build_prompt(caption_in, lyrics_in)
        sampler = make_sampler(temp=0.85, top_p=0.9)
        output_text = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=1024,
            sampler=sampler,
            verbose=False,
        )
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
            "ACE-Step LM (MLX) format: caption=%r -> %r (bpm=%s)",
            caption_in[:60],
            result.caption[:80],
            result.bpm,
        )
        return result
