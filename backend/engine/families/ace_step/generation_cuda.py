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
    COVER_DIT_INSTRUCTION,
    SAMPLE_RATE,
    SFT_GEN_PROMPT,
    diffusion_retry_seed,
    duration_to_latent_frames,
    estimate_hum_ratio,
    estimate_mains_correlation,
    capture_inference_lyrics,
    ensure_vocal_caption_hint,
    finalize_lyrics_for_inference,
    resolve_vocal_language,
    vocal_language_mismatch_warning,
    warn_weak_vocal_lyrics,
    warn_lyrics_token_truncation,
    LYRIC_TOKEN_MAX_LENGTH,
    format_lyrics,
    format_metadata,
    latents_collapsed_to_silence,
    latent_cosine_similarity,
    latent_diff_mean,
    LATENT_HOP_SAMPLES,
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
        self.last_quality: Any = None
        self.last_audio_code_indices: tuple[int, ...] = ()
        self.last_pmi: Any = None
        self._lm_formatter: Any = None
        self._lm_formatter_key: tuple[Any, ...] | None = None

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

    def _ensure_lm_formatter(
        self,
        *,
        simple_mode: bool = False,
        quantize_bits: Optional[int] = None,
        lm_dir: Optional[Path] = None,
    ) -> Any:
        key = (simple_mode, quantize_bits, str(lm_dir) if lm_dir else "")
        if self._lm_formatter is not None and self._lm_formatter_key == key:
            return self._lm_formatter
        from backend.engine.families.ace_step.lm_format_cuda import AceStepLmFormatterCuda
        from backend.engine.families.ace_step.resource_policy import resolve_lm_dir_for_policy, resolve_resource_policy

        import torch

        policy = resolve_resource_policy(backend="cuda")
        resolved_dir = lm_dir or resolve_lm_dir_for_policy(self._bundle_root, policy)
        fmt = AceStepLmFormatterCuda.from_bundle(
            self._bundle_root,
            device=torch.device(self._device),
            lm_dir=resolved_dir,
            simple_mode=simple_mode,
        )
        if fmt is None:
            return None
        fmt.load()
        self._lm_formatter = fmt
        self._lm_formatter_key = key
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
        simple_mode: bool = False,
        instrumental: bool = False,
        lm_enabled: bool = True,
        lm_quantize_bits: Optional[int] = None,
    ) -> np.ndarray:
        import torch

        del steps, guidance
        from backend.engine.families.ace_step.quality_score import assess_generation_quality

        if self._condition_model is None or self._vae is None:
            raise RuntimeError("ACE-Step CUDA generator not loaded; call load() first")

        duration = max(10.0, min(600.0, float(duration)))
        requested_samples = int(round(duration * SAMPLE_RATE))
        latent_frames = duration_to_latent_frames(duration)
        max_latent_length = snap_latent_frames_for_inference(latent_frames)
        self.last_latent_frames = max_latent_length

        silence_tiled = self._silence_latent_slice(max_latent_length)
        caption = (prompt or "").strip()
        lyrics_input = (lyrics or "").strip()
        if simple_mode and lm_enabled:
            lyrics_use = lyrics_input if lyrics_input else (
                "[Instrumental]" if instrumental else ""
            )
        else:
            lyrics_use = lyrics_input or "[Instrumental]"
        bpm_use = bpm
        key_use = key_scale or ""
        ts_use = time_signature or ""
        lang_use = resolve_vocal_language(lyrics_use or lyrics_input, vocal_language)
        from backend.engine.families.ace_step.lm_format import is_instrumental_lyrics

        self.last_lm_expanded = False
        self.last_audio_code_indices = ()
        self.last_pmi = None
        mismatch = vocal_language_mismatch_warning(lyrics_use, lang_use)
        if mismatch:
            logger.warning("ACE-Step: %s", mismatch)

        if lm_enabled and self._lm_enabled():
            lm_fmt = self._ensure_lm_formatter(
                simple_mode=simple_mode,
                quantize_bits=lm_quantize_bits,
            )
            if lm_fmt is not None:
                try:
                    if simple_mode:
                        expanded = lm_fmt.create_sample(
                            query=caption or "instrumental music",
                            instrumental=instrumental or is_instrumental_lyrics(lyrics_input),
                            vocal_language=lang_use,
                            duration=duration,
                        )
                    else:
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
                    if expanded.duration is not None:
                        duration = float(expanded.duration)
                        max_latent_length = snap_latent_frames_for_inference(
                            duration_to_latent_frames(duration)
                        )
                        self.last_latent_frames = max_latent_length
                        silence_tiled = self._silence_latent_slice(max_latent_length)
                    key_use = expanded.keyscale or key_use
                    ts_use = expanded.timesignature or ts_use
                    lang_use = expanded.language or lang_use
                    self.last_lm_expanded = True
                    self.last_audio_code_indices = tuple(expanded.audio_code_indices)
                    self.last_pmi = getattr(lm_fmt, "last_pmi", None)
                    if expanded.audio_code_indices:
                        logger.info(
                            "ACE-Step: LM produced %d audio codes for DiT hints",
                            len(expanded.audio_code_indices),
                        )
                except Exception as exc:
                    raise RuntimeError(
                        f"ACE-Step 5Hz LM expansion failed: {exc}"
                    ) from exc

        lyrics_use = finalize_lyrics_for_inference(
            lyrics_use,
            instrumental=instrumental,
            lm_expanded=self.last_lm_expanded,
        )
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
        audio_codes_t = None
        if self.last_audio_code_indices:
            pool = int(getattr(self._model_config, "pool_window_size", 5))
            max_codes = max(1, max_latent_length // pool)
            from backend.engine.families.ace_step.lm_format import build_audio_codes_tensor

            audio_codes_t = build_audio_codes_tensor(
                self.last_audio_code_indices,
                device=self._device,
                max_codes=max_codes,
            )
            is_covers = torch.ones(1, device=self._device, dtype=torch.long)

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
                gen_kwargs: dict[str, Any] = dict(
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
                if audio_codes_t is not None:
                    gen_kwargs["audio_codes"] = audio_codes_t
                out = self._condition_model.generate_audio(**gen_kwargs)
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
        collapsed, _, _ = latents_collapsed_to_silence(latents, src_np)
        self.last_quality = assess_generation_quality(
            hum_ratio=self.last_hum_ratio,
            mains_acf=self.last_mains_acf,
            latent_cos=latent_cos,
            latent_diff=latent_diff,
            lm_expanded=self.last_lm_expanded,
            near_silence=collapsed,
            pmi_bonus=self.last_pmi.quality_bonus() if self.last_pmi is not None else 0.0,
        )
        return wf

    def _encode_reference_latents(self, reference_waveform: np.ndarray) -> Any:
        import torch

        wf = np.asarray(reference_waveform, dtype=np.float32)
        if wf.ndim == 1:
            wf = wf[:, None]
        if wf.shape[0] <= 8 and wf.shape[0] < wf.shape[1]:
            wf = wf.T
        tensor = torch.from_numpy(wf).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            latents = self._vae.encode(tensor)
        if latents.ndim == 2:
            latents = latents.unsqueeze(0)
        return latents.float()

    def generate_cover_waveform(
        self,
        *,
        reference_waveform: np.ndarray,
        prompt: str = "",
        lyrics: str = "[Instrumental]",
        vocal_language: str = "en",
        duration: Optional[float] = None,
        seed: int = 0,
        bpm: Optional[int] = None,
        key_scale: str = "",
        time_signature: str = "",
        shift: float = 3.0,
        audio_cover_strength: float = 1.0,
        **kwargs: Any,
    ) -> np.ndarray:
        import torch

        del kwargs
        from backend.engine.families.ace_step.quality_score import assess_generation_quality

        if self._condition_model is None or self._vae is None:
            raise RuntimeError("ACE-Step CUDA generator not loaded; call load() first")

        ref_latents = self._encode_reference_latents(reference_waveform)
        ref_len = int(ref_latents.shape[1])
        if duration is not None:
            max_latent_length = snap_latent_frames_for_inference(
                duration_to_latent_frames(float(duration))
            )
        else:
            max_latent_length = snap_latent_frames_for_inference(ref_len)
        max_latent_length = max(max_latent_length, ref_len)
        self.last_latent_frames = max_latent_length

        if ref_len < max_latent_length:
            pad = self._silence_latent_slice(max_latent_length - ref_len)
            src_latents = torch.cat(
                [ref_latents[0], pad.to(self._device)],
                dim=0,
            ).unsqueeze(0)
        else:
            src_latents = ref_latents[:, :max_latent_length, :].clone()

        refer_len = min(750, int(src_latents.shape[1]))
        refer_packed = src_latents[:, :refer_len, :]
        refer_order = torch.tensor([0], device=self._device, dtype=torch.long)

        duration_sec = max_latent_length * LATENT_HOP_SAMPLES / SAMPLE_RATE
        requested_samples = int(round(duration_sec * SAMPLE_RATE))
        caption = (prompt or "").strip() or "cover generation"
        lyrics_use = (lyrics or "").strip() or "[Instrumental]"
        lang_use = resolve_vocal_language(lyrics_use, vocal_language)
        self.last_lm_expanded = False
        self.last_lyrics_capture = capture_inference_lyrics(
            lyrics_input=lyrics_use,
            lyrics_effective=lyrics_use,
            caption_effective=caption,
            lm_expanded=False,
        )

        instruction = COVER_DIT_INSTRUCTION
        if not instruction.endswith(":"):
            instruction = instruction + ":"
        metadata = format_metadata(
            duration=duration_sec,
            bpm=bpm,
            key_scale=key_scale or "",
            time_signature=time_signature or "",
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

        latent_dim = int(src_latents.shape[-1])
        chunk_masks = torch.ones(
            1, max_latent_length, latent_dim, device=self._device, dtype=torch.float32
        )
        is_covers = torch.ones(1, device=self._device, dtype=torch.long)
        attention_mask = torch.ones(
            1, max_latent_length, device=self._device, dtype=torch.float32
        )

        if self._is_turbo(self._model_config) and shift >= 2.5:
            shift = 1.0

        src_np = src_latents.detach().cpu().float().numpy()
        strength = float(max(0.0, min(1.0, audio_cover_strength)))
        run_seed = int(seed)
        latents = None
        latent_cos = 1.0
        latent_diff = 0.0

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
                audio_cover_strength=strength,
            )
        latents_t = out["target_latents"]
        latents = latents_t.detach().cpu().float().numpy()
        latent_cos = latent_cosine_similarity(latents, src_np)
        latent_diff = latent_diff_mean(latents, src_np)
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
        collapsed, _, _ = latents_collapsed_to_silence(latents, src_np)
        self.last_quality = assess_generation_quality(
            hum_ratio=self.last_hum_ratio,
            mains_acf=self.last_mains_acf,
            latent_cos=latent_cos,
            latent_diff=latent_diff,
            lm_expanded=False,
            near_silence=collapsed,
        )
        return wf
