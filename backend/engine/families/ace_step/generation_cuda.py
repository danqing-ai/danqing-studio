"""
ACE-Step text-to-music — PyTorch / CUDA path.

Uses upstream ``AceStepConditionGenerationModel.generate_audio`` + diffusers VAE decode.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from backend.engine.families.ace_step.generation import (
    DEFAULT_DIT_INSTRUCTION,
    SAMPLE_RATE,
    SFT_GEN_PROMPT,
    diffusion_retry_seed,
    duration_to_latent_frames,
    estimate_hum_ratio,
    estimate_mains_correlation,
    capture_inference_lyrics,
    ensure_vocal_caption_hint,
    resolve_vocal_language,
    vocal_language_mismatch_warning,
    warn_weak_vocal_lyrics,
    warn_lyrics_token_truncation,
    LYRIC_TOKEN_MAX_LENGTH,
    format_lyrics,
    format_metadata,
    latents_collapsed_to_silence,
    normalize_waveform,
    resolve_dit_bundle,
    resolve_silence_latent_npy_path,
    resolve_silence_latent_path,
    snap_latent_frames_for_inference,
)
from backend.engine.families.ace_step.vae import AceStepVAE

logger = logging.getLogger(__name__)


def _write_silence_latent_npy(pt_path: Path) -> Path:
    """Convert ``silence_latent.pt`` to ``.npy`` for MLX (CUDA load path only)."""
    import torch

    npy_path = resolve_silence_latent_npy_path(pt_path)
    if npy_path.is_file():
        return npy_path
    sl = torch.load(pt_path, weights_only=True)
    if sl.dim() == 3:
        sl = sl.transpose(1, 2)
    elif sl.dim() == 2:
        sl = sl.transpose(0, 1).unsqueeze(0)
    np.save(npy_path, sl.detach().cpu().float().numpy())
    return npy_path


class AceStepCudaGenerator:
    """CUDA PyTorch ACE-Step generator."""

    def __init__(self, ctx: Any, bundle_root: Path):
        self._ctx = ctx
        self._bundle_root = Path(bundle_root)
        self._device = "cuda"
        self._condition_model: Any = None
        self._vae: AceStepVAE | None = None
        self._text_tokenizer: Any = None
        self._text_encoder: Any = None
        self._silence_latent: Any = None
        self._model_config: Any = None
        self._lm_formatter: Any = None
        self.last_latent_frames: int = 0
        self.last_hum_ratio: float = 0.0
        self.last_mains_acf: float = 0.0
        self.last_latent_cos: float = 0.0
        self.last_latent_diff_mean: float = 0.0
        self.last_decode_mode: str = "cuda_vae"
        self.last_lm_expanded: bool = False
        self.last_lyrics_capture: Any = None

    @property
    def model_config(self) -> Any:
        if self._model_config is None:
            raise RuntimeError("ACE-Step model not loaded")
        return self._model_config

    def load(self) -> None:
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("ACE-Step CUDA backend requires torch.cuda")

        bundle = self._bundle_root
        dit_bundle = resolve_dit_bundle(bundle)
        if not (dit_bundle / "config.json").is_file():
            raise RuntimeError(f"ACE-Step config.json missing under {dit_bundle}")

        self._model_config = AutoConfig.from_pretrained(
            str(dit_bundle),
            trust_remote_code=True,
        )
        logger.info("Loading ACE-Step PyTorch model (CUDA) from %s", dit_bundle)
        self._condition_model = AutoModel.from_pretrained(
            str(dit_bundle),
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        self._condition_model.eval()
        self._condition_model.to(self._device)

        enc_dir = bundle / "Qwen3-Embedding-0.6B"
        if not enc_dir.is_dir():
            raise RuntimeError(f"Qwen3-Embedding-0.6B not found at {enc_dir}")
        self._text_tokenizer = AutoTokenizer.from_pretrained(str(enc_dir))
        self._text_encoder = AutoModel.from_pretrained(str(enc_dir))
        self._text_encoder.eval()
        self._text_encoder.to(self._device)

        sp = resolve_silence_latent_path(bundle, dit_bundle)
        _write_silence_latent_npy(sp)
        sl = torch.load(sp, weights_only=True)
        if sl.dim() == 3:
            sl = sl.transpose(1, 2)
        elif sl.dim() == 2:
            sl = sl.transpose(0, 1).unsqueeze(0)
        self._silence_latent = sl.to(self._device, dtype=torch.float32)

        self._vae = AceStepVAE(self._ctx, vae_dir=str(bundle / "vae"))
        self._vae._vae.to(self._device).float()

    @staticmethod
    def _lm_enabled() -> bool:
        import os

        return os.environ.get("ACESTEP_USE_LM", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )

    def _ensure_lm_formatter(self) -> Any:
        if self._lm_formatter is not None:
            return self._lm_formatter
        from backend.engine.families.ace_step.lm_format_cuda import AceStepLmFormatterCuda

        import torch

        fmt = AceStepLmFormatterCuda.from_bundle(
            self._bundle_root, device=torch.device(self._device),
        )
        if fmt is None:
            return None
        fmt.load()
        self._lm_formatter = fmt
        return fmt

    def _silence_latent_slice(self, length: int):
        import torch

        available = self._silence_latent.shape[1]
        if length <= available:
            return self._silence_latent[0, :length, :]
        repeats = (length + available - 1) // available
        tiled = self._silence_latent[0].repeat(repeats, 1)
        return tiled[:length, :]

    def _infer_refer_latent(self):
        import torch

        refer_latent = self._silence_latent[:, :750, :]
        refer_order = torch.tensor([0], device=self._device, dtype=torch.long)
        return refer_latent, refer_order

    @staticmethod
    def _is_turbo(config: Any) -> bool:
        return bool(getattr(config, "is_turbo", False))

    def generate_waveform(
        self,
        *,
        prompt: str,
        lyrics: str = "[Instrumental]",
        vocal_language: str = "en",
        duration: float = 30.0,
        steps: int = 50,
        guidance: float = 3.0,
        seed: int = 0,
        bpm: Optional[int] = None,
        key_scale: str = "",
        time_signature: str = "",
        shift: float = 3.0,
    ) -> np.ndarray:
        import torch

        del steps, guidance

        if self._condition_model is None or self._vae is None:
            raise RuntimeError("ACE-Step CUDA generator not loaded; call load() first")

        duration = max(10.0, min(600.0, float(duration)))
        requested_samples = int(round(duration * SAMPLE_RATE))
        latent_frames = duration_to_latent_frames(duration)
        max_latent_length = snap_latent_frames_for_inference(latent_frames)
        self.last_latent_frames = max_latent_length

        silence_tiled = self._silence_latent_slice(max_latent_length)
        caption = (prompt or "").strip()
        lyrics_use = (lyrics or "").strip() or "[Instrumental]"
        lyrics_input = lyrics_use
        bpm_use = bpm
        key_use = key_scale or ""
        ts_use = time_signature or ""
        lang_use = resolve_vocal_language(lyrics_use, vocal_language)
        self.last_lm_expanded = False
        mismatch = vocal_language_mismatch_warning(lyrics_use, lang_use)
        if mismatch:
            logger.warning("ACE-Step: %s", mismatch)

        if self._lm_enabled():
            lm_fmt = self._ensure_lm_formatter()
            if lm_fmt is not None:
                try:
                    expanded = lm_fmt.format_sample(
                        caption=caption or "instrumental music",
                        lyrics=lyrics_use,
                        duration=duration,
                        bpm=bpm_use,
                        keyscale=key_use,
                        timesignature=ts_use,
                        language=lang_use,
                    )
                    caption = expanded.caption
                    lyrics_use = expanded.lyrics
                    if expanded.bpm is not None:
                        bpm_use = expanded.bpm
                    key_use = expanded.keyscale or key_use
                    ts_use = expanded.timesignature or ts_use
                    lang_use = expanded.language or lang_use
                    self.last_lm_expanded = True
                except Exception as exc:
                    raise RuntimeError(
                        f"ACE-Step 5Hz LM expansion failed: {exc}"
                    ) from exc

        caption = ensure_vocal_caption_hint(caption, lyrics_use, lang_use)
        self.last_lyrics_capture = capture_inference_lyrics(
            lyrics_input=lyrics_input,
            lyrics_effective=lyrics_use,
            caption_effective=caption,
            lm_expanded=self.last_lm_expanded,
        )
        vocal_warn = warn_weak_vocal_lyrics(lyrics_use)
        if vocal_warn:
            logger.warning("ACE-Step: %s", vocal_warn)

        instruction = DEFAULT_DIT_INSTRUCTION
        if not instruction.endswith(":"):
            instruction = instruction + ":"
        metadata = format_metadata(
            duration=duration,
            bpm=bpm_use,
            key_scale=key_use,
            time_signature=ts_use,
        )
        text_prompt = SFT_GEN_PROMPT.format(
            instruction=instruction,
            caption=caption,
            metadata=metadata,
        )
        lyrics_text = format_lyrics(lyrics_use, lang_use)

        text_tok = self._text_tokenizer(
            text_prompt, padding="longest", truncation=True, max_length=256, return_tensors="pt",
        ).to(self._device)
        lyric_len_tok = int(
            self._text_tokenizer(
                lyrics_text, truncation=False, return_tensors="pt"
            ).input_ids.shape[1]
        )
        trunc_warn = warn_lyrics_token_truncation(lyric_len_tok)
        if trunc_warn:
            logger.warning("ACE-Step: %s", trunc_warn)
        lyric_tok = self._text_tokenizer(
            lyrics_text,
            padding="longest",
            truncation=True,
            max_length=LYRIC_TOKEN_MAX_LENGTH,
            return_tensors="pt",
        ).to(self._device)

        with torch.inference_mode():
            text_hidden = self._text_encoder(input_ids=text_tok.input_ids).last_hidden_state.float()
            lyric_hidden = self._text_encoder.embed_tokens(lyric_tok.input_ids).float()

        refer_packed, refer_order = self._infer_refer_latent()
        src_latents = silence_tiled.unsqueeze(0).clone()
        latent_dim = int(src_latents.shape[-1])
        chunk_masks = torch.ones(1, max_latent_length, latent_dim, device=self._device, dtype=torch.float32)
        is_covers = torch.zeros(1, device=self._device, dtype=torch.long)
        attention_mask = torch.ones(1, max_latent_length, device=self._device, dtype=torch.float32)

        if self._is_turbo(self._model_config) and shift >= 2.5:
            shift = 1.0

        src_np = src_latents.detach().cpu().float().numpy()
        latents = None
        latent_cos = 1.0
        latent_diff = 0.0
        max_attempts = 6 if self.last_lm_expanded else 3
        run_seed = int(seed)

        for attempt in range(max_attempts):
            torch.manual_seed(run_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(run_seed)
            with torch.inference_mode():
                out = self._condition_model.generate_audio(
                    text_hidden_states=text_hidden,
                    text_attention_mask=text_tok.attention_mask.float(),
                    lyric_hidden_states=lyric_hidden,
                    lyric_attention_mask=lyric_tok.attention_mask.float(),
                    refer_audio_acoustic_hidden_states_packed=refer_packed,
                    refer_audio_order_mask=refer_order,
                    src_latents=src_latents,
                    chunk_masks=chunk_masks,
                    is_covers=is_covers,
                    silence_latent=self._silence_latent,
                    attention_mask=attention_mask,
                    seed=run_seed,
                    infer_method="ode",
                    shift=shift,
                )
            latents_t = out["target_latents"]
            latents = latents_t.detach().cpu().float().numpy()
            collapsed, latent_cos, latent_diff = latents_collapsed_to_silence(latents, src_np)
            if not collapsed:
                break
            if attempt < max_attempts - 1:
                run_seed = diffusion_retry_seed(int(seed), attempt + 1)
                logger.warning(
                    "ACE-Step CUDA latents near silence (cos=%.4f, diff=%.4f); retry seed %d",
                    latent_cos,
                    latent_diff,
                    run_seed,
                )

        self.last_latent_cos = latent_cos
        self.last_latent_diff_mean = latent_diff

        with torch.inference_mode():
            audio = self._vae.decode(torch.from_numpy(latents).to(self._device))
        wf = audio.detach().cpu().float().numpy()
        if wf.ndim == 3:
            wf = wf[0]
        if wf.ndim == 2 and wf.shape[0] <= 8 and wf.shape[0] < wf.shape[1]:
            wf = wf.T
        wf = normalize_waveform(wf.astype(np.float32))
        if wf.shape[0] > requested_samples:
            wf = wf[:requested_samples]

        self.last_hum_ratio = estimate_hum_ratio(wf)
        self.last_mains_acf = estimate_mains_correlation(wf)
        return wf
