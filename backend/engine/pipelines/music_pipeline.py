"""
MusicPipeline — audio request → text encoding → DiT diffusion → VAE decode → asset save.

Synchronous ``run()`` called by ``DanQingAudioEngine`` via ``asyncio.to_thread``.
"""
from __future__ import annotations

import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from backend.core.contracts import (
    AudioGenerationRequest, EngineResult, ExecutionContext,
    LogEvent, parse_model_version, parse_size,
)
from backend.engine.common.cache import ModelCache
from backend.engine.common.pipeline_registry import (
    local_bundle_root as _local_bundle_root_fn,
    registry_scalar_default as _registry_scalar_default_fn,
    resolve_project_path as _resolve_project_path_fn,
    resolve_version_block as _resolve_version_block_fn,
)
from backend.engine._transformer_registry import (
    get_audio_transformer_class as _get_audio_transformer_class,
    get_audio_weight_remap as _get_audio_weight_remap,
)
from backend.engine.config.model_configs import get_config_class
from backend.engine.runtime._base import RuntimeContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text prompt templates (mirrors acestep/constants.py)
# ---------------------------------------------------------------------------

SFT_GEN_PROMPT = """# Instruction
{instruction}

# Caption
{caption}

# Metas
{metadata}<|endoftext|>"""

DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"

# ---------------------------------------------------------------------------
# Timestep scheduling (from dit_generate.py)
# ---------------------------------------------------------------------------

VALID_SHIFTS = [1.0, 2.0, 3.0]

SHIFT_TIMESTEPS: Dict[float, List[float]] = {
    1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
    2.0: [1.0, 0.9333333333333333, 0.8571428571428571, 0.7692307692307693,
          0.6666666666666666, 0.5454545454545454, 0.4, 0.2222222222222222],
    3.0: [1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75,
          0.6428571428571429, 0.5, 0.3],
}

_AUDIO_DENOISE_PROGRESS_SHARE = 0.80
_AUDIO_POST_PROGRESS_SHARE = 0.20


def _get_timestep_schedule(
    shift: float = 3.0,
    timesteps: Optional[list] = None,
    infer_steps: Optional[int] = None,
) -> List[float]:
    if timesteps is not None:
        ts_list = [t for t in timesteps if t != 0]
        if len(ts_list) < 1:
            logger.warning("timesteps empty after removing zeros; using default shift=%s", shift)
        else:
            return ts_list

    if infer_steps is not None and infer_steps > 0:
        raw = [1.0 - i / infer_steps for i in range(infer_steps)]
        if shift != 1.0:
            raw = [shift * t / (1.0 + (shift - 1.0) * t) for t in raw]
        return raw

    original_shift = shift
    shift = min(VALID_SHIFTS, key=lambda x: abs(x - shift))
    if original_shift != shift:
        logger.warning("shift=%.2f rounded to nearest valid shift=%.1f", original_shift, shift)
    return SHIFT_TIMESTEPS[shift]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class MusicPipeline:
    """Audio generation pipeline — MLX / CUDA dual-backend via RuntimeContext."""

    def __init__(
        self,
        ctx: RuntimeContext,
        model_registry: Any,
        asset_store: Any,
        model_cache: ModelCache | None = None,
        project_root: Path | None = None,
    ):
        self.ctx = ctx
        self._registry = model_registry
        self._asset_store = asset_store
        self._cache = model_cache
        self._project_root = project_root or Path.cwd()
        self._dit: Any = None
        self._vae: Any = None
        self._text_tokenizer: Any = None
        self._text_encoder: Any = None
        self._silence_latent: Any = None

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, local_path: str) -> Path:
        return _resolve_project_path_fn(self._project_root, local_path)

    @staticmethod
    def _registry_scalar_default(entry, key: str, fallback):
        return _registry_scalar_default_fn(entry, key, fallback)

    def _resolve_version_block(self, entry, version_key: str | None) -> dict | None:
        return _resolve_version_block_fn(entry, version_key)

    def _local_bundle_root(self, entry, version_key: str | None) -> Path | None:
        return _local_bundle_root_fn(self._project_root, entry, version_key)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(
        self,
        request: AudioGenerationRequest,
        exec_ctx: ExecutionContext,
    ) -> EngineResult:
        t0 = time.monotonic()
        model_id, version_key = parse_model_version(request.model)
        entry = self._registry.get(model_id)
        if entry is None:
            raise RuntimeError(f"Model {model_id!r} not found in registry")

        bundle_root = self._local_bundle_root(entry, version_key)
        if bundle_root is None or not bundle_root.is_dir():
            raise RuntimeError(f"Model bundle not found for {request.model!r}")

        cfg_cls = get_config_class(entry.family)
        config = cfg_cls()

        exec_ctx.on_log(LogEvent(level="info", message=f"Loading ACE-Step model: {model_id}"))

        # 1) Load DiT
        self._load_dit(entry.family, config, bundle_root)

        # 2) Load VAE
        self._load_vae(config, bundle_root)

        # 3) Load text encoder
        self._load_text_encoder(bundle_root)

        # 4) Load silence latent
        self._load_silence_latent(bundle_root)

        # 5) Prepare conditioning
        cond_result = self._prepare_conditioning(request, config)
        encoder_hidden_states = cond_result["encoder_hidden_states"]
        context_latents = cond_result["context_latents"]

        # 6) Diffusion
        steps = request.steps or config.default_infer_steps
        guidance = request.guidance or 3.0
        seed = request.seed if request.seed is not None and request.seed >= 0 else random.randint(0, 2**31 - 1)
        n = max(request.n, 1)

        exec_ctx.on_log(LogEvent(level="info", message=f"Generating {n} audio(s), steps={steps}, seed={seed}"))

        results: List[Path] = []
        for i in range(n):
            batch_seed = seed + i
            exec_ctx.on_progress(step=i + 1, total=n, label=f"Generating audio {i + 1}/{n}")

            latents = self._diffusion_loop(
                encoder_hidden_states=encoder_hidden_states,
                context_latents=context_latents,
                steps=steps,
                guidance_scale=guidance,
                seed=batch_seed,
                shift=getattr(config, "default_shift", 3.0),
                is_turbo=config.is_turbo,
                progress_cb=lambda p: exec_ctx.on_progress(
                    step=i + 1, total=n, progress=p, label=f"Denoising {i + 1}/{n}",
                ),
            )

            # 7) VAE decode
            exec_ctx.on_log(LogEvent(level="info", message="Decoding audio..."))
            waveform = self._vae_decode(latents)

            # 8) Save
            audio_path = self._save_audio(waveform, request, model_id, batch_seed)
            results.append(audio_path)

        exec_ctx.on_progress(step=n, total=n, progress=1.0, label="Complete")

        elapsed = time.monotonic() - t0
        asset_ids = self._persist_assets(results, request, model_id, elapsed)

        return EngineResult(
            primary_asset_id=asset_ids[0] if asset_ids else "",
            asset_ids=asset_ids,
            metadata={
                "model": model_id,
                "seed": seed,
                "steps": steps,
                "guidance": guidance,
                "duration_seconds": None,
            },
        )

    # ------------------------------------------------------------------
    # DiT loading
    # ------------------------------------------------------------------

    def _load_dit(self, family: str, config: Any, bundle_root: Path):
        cls = _get_audio_transformer_class(family)
        backend = getattr(self.ctx, "backend", "mlx")
        dit_kwargs = {
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "rms_norm_eps": config.rms_norm_eps,
            "attention_bias": config.attention_bias,
            "in_channels": config.in_channels,
            "audio_acoustic_hidden_dim": config.audio_acoustic_hidden_dim,
            "patch_size": config.patch_size,
            "sliding_window": config.sliding_window,
            "layer_types": list(config.layer_types) if config.layer_types else None,
            "rope_theta": config.rope_theta,
            "max_position_embeddings": config.max_position_embeddings,
        }
        self._dit = cls(self.ctx, **dit_kwargs)

        remap_fn = _get_audio_weight_remap(family)
        if remap_fn is not None:
            raw_weights = self._load_safetensor_weights(bundle_root)
            mapped = remap_fn(raw_weights)
            self._dit.load_weights(mapped, strict=False, ctx=self.ctx)

    # ------------------------------------------------------------------
    # VAE loading
    # ------------------------------------------------------------------

    def _load_vae(self, config: Any, bundle_root: Path):
        from backend.engine.families.ace_step.vae import AceStepVAE
        vae_dir = str(bundle_root / "vae")
        self._vae = AceStepVAE(
            self.ctx,
            vae_dir=vae_dir,
            encoder_hidden_size=config.vae_encoder_hidden_size,
            downsampling_ratios=list(config.vae_downsampling_ratios),
            channel_multiples=list(config.vae_channel_multiples),
            decoder_channels=config.vae_decoder_channels,
            decoder_input_channels=config.vae_decoder_input_channels,
            audio_channels=config.audio_channels,
        )

    # ------------------------------------------------------------------
    # Text encoder
    # ------------------------------------------------------------------

    def _load_text_encoder(self, bundle_root: Path):
        enc_dir = bundle_root / "Qwen3-Embedding-0.6B"
        if not enc_dir.is_dir():
            raise RuntimeError(
                f"Qwen3-Embedding-0.6B not found at {enc_dir}. "
                "Download it from Hugging Face: Qwen/Qwen3-Embedding-0.6B"
            )
        from transformers import AutoModel, AutoTokenizer
        self._text_tokenizer = AutoTokenizer.from_pretrained(str(enc_dir))
        self._text_encoder = AutoModel.from_pretrained(str(enc_dir))
        self._text_encoder.eval()

    def _load_silence_latent(self, bundle_root: Path):
        sp = bundle_root / "silence_latent.pt"
        if not sp.exists():
            logger.warning("silence_latent.pt not found; using zeros")
            return
        import torch
        t = torch.load(sp, weights_only=True)
        if t.dim() == 2:
            t = t.transpose(0, 1)
        self._silence_latent = t

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------

    def _prepare_conditioning(
        self, request: AudioGenerationRequest, config: Any,
    ) -> Dict[str, Any]:
        import torch

        caption = request.prompt or ""
        lyrics = request.lyrics or "[Instrumental]"
        language = request.vocal_language or "en"
        instruction = DEFAULT_DIT_INSTRUCTION
        duration = request.duration or 30
        bpm = request.bpm
        key_scale = request.key_scale or ""
        time_sig = request.time_signature or ""

        # Format metas string
        metas_parts = []
        if bpm:
            metas_parts.append(f"bpm: {bpm}")
        if key_scale:
            metas_parts.append(f"key: {key_scale}")
        if time_sig:
            metas_parts.append(f"time_signature: {time_sig}")
        metas_parts.append(f"duration: {duration}")
        metas_str = "\n".join(metas_parts)

        # Format caption prompt
        caption_prompt = SFT_GEN_PROMPT.format(
            instruction=instruction + ":", caption=caption, metadata=metas_str,
        )

        # Format lyrics
        lyric_prompt = f"# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>"

        # Tokenize caption
        cap_inputs = self._text_tokenizer(
            caption_prompt, padding="max_length", truncation=True,
            max_length=256, return_tensors="pt",
        )
        # Tokenize lyrics
        lyr_inputs = self._text_tokenizer(
            lyric_prompt, padding="max_length", truncation=True,
            max_length=2048, return_tensors="pt",
        )

        # Encode via Qwen3-Embedding
        with torch.inference_mode():
            cap_out = self._text_encoder(cap_inputs.input_ids)
            lyr_out = self._text_encoder(lyr_inputs.input_ids)

        cap_emb = cap_out.last_hidden_state if hasattr(cap_out, "last_hidden_state") else cap_out[0]
        lyr_emb = lyr_out.last_hidden_state if hasattr(lyr_out, "last_hidden_state") else lyr_out[0]

        # Concatenate caption + lyric embeddings along sequence dim
        encoder_hidden_states = torch.cat([cap_emb, lyr_emb], dim=1)

        # Context latents — silence-based for now
        # Estimate target latent length from duration
        # VAE hop_length = product of downsampling_ratios = 2*4*4*8*8 = 2048
        # sample_rate = 48000
        # latent_frames ≈ duration * sample_rate / hop_length = duration * 48000 / 2048 ≈ duration * 23.4
        sample_rate = 48000
        downsample_ratios = getattr(config, "vae_downsampling_ratios", (2, 4, 4, 8, 8))
        hop_len = 1
        for r in downsample_ratios:
            hop_len *= r
        latent_frames = max(1, int(duration * sample_rate / hop_len))
        latent_dim = config.audio_acoustic_hidden_dim  # 64

        # Prepare silence latent
        if hasattr(self, "_silence_latent") and self._silence_latent is not None:
            sl = self._silence_latent
            if sl.dim() == 3:
                sl = sl[0]
            if sl.shape[1] < latent_frames:
                pad = torch.zeros(sl.shape[0], latent_frames - sl.shape[1])
                sl = torch.cat([sl, pad], dim=1)
            elif sl.shape[1] > latent_frames:
                sl = sl[:, :latent_frames]
            context_latents = sl.T.unsqueeze(0).repeat(1, 1, 1)
        else:
            ctx_channels = config.in_channels - latent_dim  # 192 - 64 = 128
            context_latents = torch.zeros(1, latent_frames, ctx_channels)

        return {
            "encoder_hidden_states": encoder_hidden_states.float(),
            "context_latents": context_latents.float(),
        }

    # ------------------------------------------------------------------
    # Diffusion loop
    # ------------------------------------------------------------------

    def _diffusion_loop(
        self,
        encoder_hidden_states: Any,
        context_latents: Any,
        steps: int,
        guidance_scale: float,
        seed: int,
        shift: float,
        is_turbo: bool,
        progress_cb: Callable[[float], None] | None = None,
    ) -> Any:
        """ODE/SDE diffusion sampling with APG guidance."""
        import torch

        backend = getattr(self.ctx, "backend", "mlx")
        latent_dim = 64  # VAE latent dim
        bsz = 1
        T = context_latents.shape[1]

        # Convert PyTorch tensors to MLX if needed
        if backend == "mlx":
            import mlx.core as mx
            if isinstance(encoder_hidden_states, torch.Tensor):
                encoder_hidden_states = mx.array(encoder_hidden_states.numpy())
            if isinstance(context_latents, torch.Tensor):
                context_latents = mx.array(context_latents.numpy())

        # Timestep schedule
        t_schedule = _get_timestep_schedule(shift, infer_steps=steps)
        num_steps = len(t_schedule)

        # Noise
        if backend == "mlx":
            import mlx.core as mx
            noise = mx.random.normal((bsz, T, latent_dim), key=mx.random.key(seed))
        else:
            g = torch.Generator(device="cpu")
            g.manual_seed(seed)
            noise = torch.randn(bsz, T, latent_dim, generator=g)

        # CFG — for non-turbo models with guidance > 1
        do_cfg = not is_turbo and guidance_scale > 1.0

        if do_cfg:
            if backend == "mlx":
                import mlx.core as mx
                null_cond = mx.zeros_like(encoder_hidden_states[:1, :1, :])
                null_expanded = mx.broadcast_to(null_cond, encoder_hidden_states.shape)
                encoder_hidden_states = mx.concatenate([encoder_hidden_states, null_expanded], axis=0)
                context_latents = mx.concatenate([context_latents, context_latents], axis=0)
                noise = mx.concatenate([noise, noise], axis=0)
            else:
                null_cond = torch.zeros_like(encoder_hidden_states[:1, :1, :])
                null_expanded = null_cond.expand(encoder_hidden_states.shape)
                encoder_hidden_states = torch.cat([encoder_hidden_states, null_expanded], dim=0)
                context_latents = torch.cat([context_latents, context_latents], dim=0)
                noise = torch.cat([noise, noise], dim=0)

        if backend == "mlx":
            latents = self._diffusion_loop_mlx(
                noise, encoder_hidden_states, context_latents,
                t_schedule, num_steps, bsz, do_cfg, guidance_scale, progress_cb,
            )
        else:
            latents = self._diffusion_loop_torch(
                noise, encoder_hidden_states, context_latents,
                t_schedule, num_steps, bsz, do_cfg, guidance_scale, progress_cb,
            )

        return latents

    def _diffusion_loop_mlx(
        self, noise, enc_hs, ctx_lat,
        t_schedule, num_steps, bsz, do_cfg, guidance_scale, progress_cb,
    ):
        import mlx.core as mx
        from backend.engine.families.ace_step.transformer_mlx import _CrossAttentionCache

        xt = noise
        cache = _CrossAttentionCache() if not do_cfg else None
        momentum_state: Optional[Dict] = {} if do_cfg else None

        for step_idx in range(num_steps):
            current_t = t_schedule[step_idx]
            x_in = mx.concatenate([xt, xt], axis=0) if do_cfg else xt
            t_curr = mx.full((x_in.shape[0],), current_t)

            vt, cache = self._dit(
                hidden_states=x_in,
                timestep=t_curr,
                timestep_r=t_curr,
                encoder_hidden_states=enc_hs,
                context_latents=ctx_lat,
                cache=cache,
                use_cache=not do_cfg,
            )
            mx.eval(vt)

            if do_cfg:
                pred_cond = vt[:bsz]
                pred_uncond = vt[bsz:]
                vt = _apg_forward_mlx(pred_cond, pred_uncond, guidance_scale, momentum_state)

            if step_idx == num_steps - 1:
                t_unsq = mx.full((bsz, 1, 1), current_t)
                xt = xt - vt * t_unsq
                mx.eval(xt)
            else:
                next_t = t_schedule[step_idx + 1]
                dt = current_t - next_t
                dt_arr = mx.full((bsz, 1, 1), dt)
                xt = xt - vt * dt_arr
                mx.eval(xt)

            if progress_cb:
                progress_cb((step_idx + 1) / num_steps * _AUDIO_DENOISE_PROGRESS_SHARE)

        return np.array(xt)

    def _diffusion_loop_torch(
        self, noise, enc_hs, ctx_lat,
        t_schedule, num_steps, bsz, do_cfg, guidance_scale, progress_cb,
    ):
        import torch
        import torch.nn.functional as F

        xt = noise
        momentum_state: Optional[Dict] = {} if do_cfg else None
        device = next(self._dit.parameters()).device

        enc_hs = enc_hs.to(device=device) if hasattr(enc_hs, "to") else enc_hs
        ctx_lat = ctx_lat.to(device=device) if hasattr(ctx_lat, "to") else ctx_lat
        xt = xt.to(device=device) if hasattr(xt, "to") else xt

        for step_idx in range(num_steps):
            current_t = t_schedule[step_idx]
            x_in = torch.cat([xt, xt], dim=0) if do_cfg else xt
            t_curr = torch.full((x_in.shape[0],), current_t, device=device)

            with torch.inference_mode():
                vt, _ = self._dit(
                    hidden_states=x_in,
                    timestep=t_curr,
                    timestep_r=t_curr,
                    encoder_hidden_states=enc_hs,
                    context_latents=ctx_lat,
                )

            if do_cfg:
                pred_cond = vt[:bsz]
                pred_uncond = vt[bsz:]
                vt = _apg_forward_torch(pred_cond, pred_uncond, guidance_scale, momentum_state)

            if step_idx == num_steps - 1:
                xt = xt - vt * current_t
            else:
                next_t = t_schedule[step_idx + 1]
                dt = current_t - next_t
                xt = xt - vt * dt

            if progress_cb:
                progress_cb((step_idx + 1) / num_steps * _AUDIO_DENOISE_PROGRESS_SHARE)

        return xt.detach().cpu()

    # ------------------------------------------------------------------
    # VAE decode
    # ------------------------------------------------------------------

    def _vae_decode(self, latents: Any) -> Any:
        backend = getattr(self.ctx, "backend", "mlx")
        if backend == "mlx":
            import mlx.core as mx
            z = mx.array(latents) if not isinstance(latents, mx.array) else latents
            audio = self._vae.decode(z)
            return np.array(audio)
        else:
            import torch
            audio = self._vae.decode(latents)
            if isinstance(audio, torch.Tensor):
                return audio.detach().cpu().numpy()
            return audio

    # ------------------------------------------------------------------
    # Audio save
    # ------------------------------------------------------------------

    def _save_audio(
        self, waveform: Any, request: AudioGenerationRequest, model_id: str, seed: int,
    ) -> Path:
        import soundfile as sf

        fmt = request.audio_format or "mp3"
        out_dir = self._project_root / "outputs" / "audio"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"ace_step_{model_id.replace('/', '_')}_{ts}_{seed}.wav"
        out_path = out_dir / fname

        wf = np.array(waveform) if not isinstance(waveform, np.ndarray) else waveform
        if wf.ndim == 3:
            wf = wf[0]
        # Ensure shape is [samples, channels]
        if wf.ndim == 1:
            wf = wf[:, None]

        sf.write(str(out_path), wf, 48000)
        return out_path

    # ------------------------------------------------------------------
    # Asset persistence
    # ------------------------------------------------------------------

    def _persist_assets(
        self, paths: List[Path], request: AudioGenerationRequest, model_id: str, elapsed: float,
    ) -> List[str]:
        ids = []
        for p in paths:
            aid = self._asset_store.create_from_file(
                str(p),
                kind="audio",
                metadata={
                    "model": model_id,
                    "prompt": request.prompt,
                    "duration_seconds": None,
                    "format": request.audio_format or "mp3",
                },
            )
            ids.append(aid)
        return ids

    def _load_safetensor_weights(self, bundle_root: Path) -> List[Tuple[str, Any]]:
        """Load all safetensor weights from a bundle directory."""
        from safetensors import safe_open
        import os

        index_path = bundle_root / "model.safetensors.index.json"
        weights: Dict[str, Any] = {}
        if index_path.exists():
            import json
            with open(index_path) as f:
                idx = json.load(f)
            weight_map = idx.get("weight_map", {})
            shard_files = set(weight_map.values())
            backend = getattr(self.ctx, "backend", "mlx")
            for shard_name in sorted(shard_files):
                shard_path = bundle_root / shard_name
                with safe_open(str(shard_path), framework="pt") as sf:
                    for key in sf.keys():
                        t = sf.get_tensor(key)
                        if backend == "mlx":
                            import mlx.core as mx
                            weights[key] = mx.array(t.numpy())
                        else:
                            weights[key] = t
        else:
            shards = sorted(bundle_root.glob("model-*.safetensors"))
            if not shards:
                raise RuntimeError(f"No safetensor files found in {bundle_root}")
            backend = getattr(self.ctx, "backend", "mlx")
            for shard_path in shards:
                with safe_open(str(shard_path), framework="pt") as sf:
                    for key in sf.keys():
                        t = sf.get_tensor(key)
                        if backend == "mlx":
                            import mlx.core as mx
                            weights[key] = mx.array(t.numpy())
                        else:
                            weights[key] = t

        return list(weights.items())


# ---------------------------------------------------------------------------
# APG guidance helpers
# ---------------------------------------------------------------------------

def _apg_forward_mlx(
    pred_cond, pred_uncond, guidance_scale: float,
    momentum_state: Optional[Dict] = None,
    norm_threshold: float = 2.5,
):
    import mlx.core as mx

    proj_axis = 1
    diff = pred_cond - pred_uncond
    if momentum_state is not None:
        diff = diff + momentum_state.get("running", 0)
        momentum_state["running"] = diff

    if norm_threshold > 0:
        diff_norm = mx.sqrt((diff * diff).sum(axis=proj_axis, keepdims=True))
        scale_factor = mx.minimum(
            mx.ones_like(diff_norm), norm_threshold / (diff_norm + 1e-8),
        )
        diff = diff * scale_factor

    v1 = pred_cond / (mx.sqrt((pred_cond * pred_cond).sum(axis=proj_axis, keepdims=True)) + 1e-8)
    parallel = (diff * v1).sum(axis=proj_axis, keepdims=True) * v1
    orthogonal = diff - parallel
    return pred_cond + (guidance_scale - 1) * orthogonal


def _apg_forward_torch(
    pred_cond, pred_uncond, guidance_scale: float,
    momentum_state: Optional[Dict] = None,
    norm_threshold: float = 2.5,
):
    import torch

    proj_axis = 1
    diff = pred_cond - pred_uncond
    if momentum_state is not None:
        diff = diff + momentum_state.get("running", 0)
        momentum_state["running"] = diff

    if norm_threshold > 0:
        diff_norm = torch.sqrt((diff * diff).sum(dim=proj_axis, keepdim=True))
        scale_factor = torch.minimum(
            torch.ones_like(diff_norm), norm_threshold / (diff_norm + 1e-8),
        )
        diff = diff * scale_factor

    v1 = pred_cond / (torch.sqrt((pred_cond * pred_cond).sum(dim=proj_axis, keepdim=True)) + 1e-8)
    parallel = (diff * v1).sum(dim=proj_axis, keepdim=True) * v1
    orthogonal = diff - parallel
    return pred_cond + (guidance_scale - 1) * orthogonal
