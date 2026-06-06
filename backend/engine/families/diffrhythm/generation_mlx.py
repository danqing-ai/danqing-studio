"""
DiffRhythm 2 text-to-music — pure MLX generation path.

CFM block flow matching + BigVGAN decode on MLX; MuQ/G2P conditioning in ``condition_mlx``.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import numpy as np

from backend.engine.families.diffrhythm.condition_mlx import (
    MuQStyleEncoderMLX,
    lyrics_to_mx_array,
    parse_lyrics_to_token_ids,
)
from backend.engine.families.diffrhythm.generation import (
    LATENT_FRAME_RATE,
    SAMPLE_RATE,
    assert_diffrhythm2_bundle,
    duration_to_latent_frames,
    estimate_hum_ratio,
    estimate_mains_correlation,
    normalize_waveform,
    resolve_dit_bundle,
)
from backend.engine.families.diffrhythm.transformer_mlx import (
    DiffRhythm2CFMMLX,
    DiffRhythm2DiTMLX,
    load_cfm_weights,
)
from backend.engine.families.diffrhythm.vae_mlx import (
    DiffRhythm2DecoderMLX,
    make_fake_stereo,
)
from backend.engine.families.diffrhythm.weights_mlx import load_diffrhythm_safetensors_for_mlx
from backend.engine.config.model_configs import DiffRhythmConfig

logger = logging.getLogger(__name__)


class DiffRhythmMlxGenerator:
    """MLX DiffRhythm 2 generator (text style + lyrics → waveform)."""

    def __init__(self, ctx: Any, bundle_root: Path):
        self._ctx = ctx
        self._bundle_root = Path(bundle_root)
        self._cfm: DiffRhythm2CFMMLX | None = None
        self._decoder: DiffRhythm2DecoderMLX | None = None
        self._style_encoder: MuQStyleEncoderMLX | None = None
        self._model_config: dict[str, Any] | None = None
        self._config = DiffRhythmConfig()
        self.last_latent_frames: int = 0
        self.last_hum_ratio: float = 0.0
        self.last_mains_acf: float = 0.0
        self.last_latent_cos: float = 0.0
        self.last_latent_diff_mean: float = 0.0
        self.last_decode_mode: str = "bigvgan_mlx"
        self.last_quality: Any = None

    @property
    def model_config(self) -> Any:
        if self._model_config is None:
            raise RuntimeError("DiffRhythm 2 model not loaded")
        return self._model_config

    def load(self) -> None:
        bundle = assert_diffrhythm2_bundle(self._bundle_root)
        dit_bundle = resolve_dit_bundle(bundle)
        cfg_path = dit_bundle / "config.json"
        with open(cfg_path, encoding="utf-8") as f:
            self._model_config = json.load(f)

        cfg = self._model_config
        block_size = int(cfg.get("block_size", self._config.block_size))
        dit = DiffRhythm2DiTMLX(
            dim=int(cfg.get("dim", self._config.dim)),
            depth=int(cfg.get("depth", self._config.depth)),
            heads=int(cfg.get("heads", self._config.heads)),
            ff_mult=int(cfg.get("ff_mult", self._config.ff_mult)),
            mel_dim=int(cfg.get("mel_dim", self._config.mel_dim)),
            text_num_embeds=int(cfg.get("text_num_embeds", self._config.text_num_embeds)),
            block_size=block_size,
            repa_depth=int(cfg.get("repa_depth", self._config.repa_depth)),
            repa_dims=list(cfg.get("repa_dims", self._config.repa_dims)),
        )
        self._cfm = DiffRhythm2CFMMLX(
            dit,
            num_channels=int(cfg.get("mel_dim", self._config.mel_dim)),
            block_size=block_size,
        )

        weights = load_diffrhythm_safetensors_for_mlx(
            str(dit_bundle / "model.safetensors"),
            array_fn=self._ctx.array,
        )
        load_cfm_weights(self._cfm, weights)
        self._ctx.eval(self._cfm.parameters())

        mulan_cache = bundle / "mulan"
        self._style_encoder = MuQStyleEncoderMLX(
            mulan_cache,
            self._config.mulan_repo_id,
        )
        self._style_encoder.load()

        self._decoder = DiffRhythm2DecoderMLX(self._ctx, vae_dir=str(bundle))
        logger.info("DiffRhythm 2 MLX stack loaded from %s", bundle)

    def generate_waveform(
        self,
        *,
        prompt: str,
        lyrics: str,
        vocal_language: str = "en",
        duration: float = 30.0,
        steps: int = 16,
        guidance: float = 2.0,
        seed: int = 0,
        bpm: Optional[int] = None,
        key_scale: str = "",
        time_signature: str = "",
        shift: float = 1.0,
        **kwargs: Any,
    ) -> np.ndarray:
        del bpm, key_scale, time_signature, shift, kwargs

        if self._cfm is None or self._decoder is None or self._style_encoder is None:
            raise RuntimeError("DiffRhythm 2 MLX generator not loaded; call load() first")

        max_secs = float(self._config.max_duration_seconds)
        duration = max(5.0, min(max_secs, float(duration)))
        requested_samples = int(round(duration * SAMPLE_RATE))
        block_size = int(self._model_config.get("block_size", self._config.block_size))
        latent_frames = duration_to_latent_frames(duration, block_size=block_size)
        latent_duration = int(round(duration * LATENT_FRAME_RATE))
        self.last_latent_frames = latent_frames

        token_ids = parse_lyrics_to_token_ids(lyrics, vocal_language=vocal_language)
        text_mx = lyrics_to_mx_array(token_ids, array_fn=self._ctx.array)
        text_mx = mx.expand_dims(text_mx, 0)

        style_mx = self._style_encoder.encode_text(prompt, array_fn=self._ctx.array)
        if style_mx.ndim == 1:
            style_mx = mx.expand_dims(style_mx, 0)

        t0 = time.monotonic()
        latents = self._cfm.sample_block_cache(
            text=text_mx,
            duration=latent_duration,
            style_prompt=style_mx,
            steps=int(steps),
            cfg_strength=float(guidance),
            seed=int(seed),
        )
        self._ctx.eval(latents)
        gen_s = time.monotonic() - t0
        logger.info(
            "DiffRhythm 2 CFM done: %.1fs, latent_frames=%d, steps=%d, cfg=%.2f",
            gen_s,
            int(latents.shape[1]),
            steps,
            guidance,
        )

        audio_mx = self._decoder.decode_audio(latents)
        self._ctx.eval(audio_mx)
        wf = np.array(audio_mx, dtype=np.float32)
        if wf.ndim == 3:
            wf = wf[0]
        mono = wf[:, 0] if wf.ndim == 2 else wf.reshape(-1)
        if self._config.fake_stereo:
            stereo = make_fake_stereo(mono[None, :], SAMPLE_RATE)
            wf = stereo.T
        else:
            wf = mono[:, None]

        wf = normalize_waveform(wf.astype(np.float32))
        if wf.shape[0] > requested_samples:
            wf = wf[:requested_samples]

        self.last_hum_ratio = estimate_hum_ratio(wf)
        self.last_mains_acf = estimate_mains_correlation(wf)

        from backend.engine.families.diffrhythm.quality_score import assess_generation_quality

        self.last_quality = assess_generation_quality(
            hum_ratio=self.last_hum_ratio,
            mains_acf=self.last_mains_acf,
        )
        return wf
