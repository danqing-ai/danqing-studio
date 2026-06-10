"""LTX 2.3 two-stage T2V/I2V generation — pure in-repo MLX orchestration."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np
from PIL import Image

from backend.engine.runtime.mlx_runtime import run_eval
from backend.engine.config.model_configs import LTXConfig
from backend.engine.families.ltx.pipeline_math import (
    DISTILLED_SIGMAS,
    STAGE_2_SIGMAS,
    AudioPatchifier,
    LatentState,
    VideoConditionByLatentIndex,
    VideoLatentPatchifier,
    apply_denoise_mask,
    compute_audio_positions,
    compute_audio_token_count,
    compute_video_latent_shape,
    compute_video_positions,
    create_noised_state,
    ltx2_dynamic_schedule,
)
from backend.engine.families.ltx.text_encoder_mlx import LTX23GemmaEncoder
from backend.engine.families.ltx.transformer_mlx import LTX23X0Model, load_ltx23_x0_model
from backend.engine.families.ltx.vae import decode_ltx23_av_to_mp4
from backend.engine.families.ltx.vae_mlx import load_ltx23_latent_upsampler, load_ltx23_video_encoder
from backend.engine.runtime._base import RuntimeContext

_DEFAULT_CFG = 3.0
_DEFAULT_AUDIO_CFG = 7.0


@dataclass
class _DenoiseOutput:
    video_latent: mx.array
    audio_latent: mx.array


def _materialize(ctx: RuntimeContext, *arrays: mx.array) -> None:
    run_eval(getattr(ctx, "eval", None), *arrays)


def _clear_cache(ctx: RuntimeContext) -> None:
    if hasattr(ctx, "clear_cache"):
        ctx.clear_cache()


def _euler_step(x: mx.array, x0: mx.array, sigma: float, sigma_next: float) -> mx.array:
    if sigma == 0:
        return x0
    d = (x - x0) / sigma
    return x + (sigma_next - sigma) * d


def _is_uniform_mask(mask: mx.array) -> bool:
    return bool(mx.all(mask == 1.0).item())


def _per_token_timesteps(ctx: RuntimeContext, sigma: float, denoise_mask: mx.array) -> mx.array:
    return (denoise_mask * sigma).squeeze(-1)


def _denoise_loop(
    ctx: RuntimeContext,
    model: LTX23X0Model,
    video_state: LatentState,
    audio_state: LatentState,
    video_text_embeds: mx.array,
    audio_text_embeds: mx.array,
    sigmas: list[float],
) -> _DenoiseOutput:
    video_x = video_state.latent
    audio_x = audio_state.latent
    video_uniform = _is_uniform_mask(video_state.denoise_mask)
    audio_uniform = _is_uniform_mask(audio_state.denoise_mask)

    for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:]):
        sigma_arr = ctx.array([sigma], dtype=ctx.bfloat16())
        b = int(video_x.shape[0])
        call_kwargs: dict[str, Any] = dict(
            video_latent=video_x,
            audio_latent=audio_x,
            sigma=ctx.broadcast_to(sigma_arr, (b,)),
            video_text_embeds=video_text_embeds,
            audio_text_embeds=audio_text_embeds,
            video_positions=video_state.positions,
            audio_positions=audio_state.positions,
        )
        if not video_uniform:
            call_kwargs["video_timesteps"] = _per_token_timesteps(ctx, sigma, video_state.denoise_mask)
        if not audio_uniform:
            call_kwargs["audio_timesteps"] = _per_token_timesteps(ctx, sigma, audio_state.denoise_mask)

        video_x0, audio_x0 = model(**call_kwargs)
        video_x0 = apply_denoise_mask(ctx, video_x0, video_state.clean_latent, video_state.denoise_mask)
        audio_x0 = apply_denoise_mask(ctx, audio_x0, audio_state.clean_latent, audio_state.denoise_mask)

        video_x = _euler_step(video_x, video_x0, sigma, sigma_next)
        audio_x = _euler_step(audio_x, audio_x0, sigma, sigma_next)
        _materialize(ctx, video_x, audio_x)

    return _DenoiseOutput(video_latent=video_x, audio_latent=audio_x)


def _guided_denoise_loop(
    ctx: RuntimeContext,
    model: LTX23X0Model,
    video_state: LatentState,
    audio_state: LatentState,
    video_text_embeds: mx.array,
    audio_text_embeds: mx.array,
    neg_video_embeds: mx.array,
    neg_audio_embeds: mx.array,
    sigmas: list[float],
    *,
    video_cfg: float,
    audio_cfg: float,
) -> _DenoiseOutput:
    video_x = video_state.latent
    audio_x = audio_state.latent
    video_uniform = _is_uniform_mask(video_state.denoise_mask)
    audio_uniform = _is_uniform_mask(audio_state.denoise_mask)

    for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:]):
        sigma_arr = ctx.array([sigma], dtype=ctx.bfloat16())
        b = int(video_x.shape[0])

        def _predict(v_embeds: mx.array, a_embeds: mx.array) -> tuple[mx.array, mx.array]:
            call_kwargs: dict[str, Any] = dict(
                video_latent=video_x,
                audio_latent=audio_x,
                sigma=ctx.broadcast_to(sigma_arr, (b,)),
                video_text_embeds=v_embeds,
                audio_text_embeds=a_embeds,
                video_positions=video_state.positions,
                audio_positions=audio_state.positions,
            )
            if not video_uniform:
                call_kwargs["video_timesteps"] = _per_token_timesteps(ctx, sigma, video_state.denoise_mask)
            if not audio_uniform:
                call_kwargs["audio_timesteps"] = _per_token_timesteps(ctx, sigma, audio_state.denoise_mask)
            v_x0, a_x0 = model(**call_kwargs)
            v_x0 = apply_denoise_mask(ctx, v_x0, video_state.clean_latent, video_state.denoise_mask)
            a_x0 = apply_denoise_mask(ctx, a_x0, audio_state.clean_latent, audio_state.denoise_mask)
            return v_x0, a_x0

        pos_v, pos_a = _predict(video_text_embeds, audio_text_embeds)
        neg_v, neg_a = _predict(neg_video_embeds, neg_audio_embeds)
        video_x0 = neg_v + video_cfg * (pos_v - neg_v)
        audio_x0 = neg_a + audio_cfg * (pos_a - neg_a)

        video_x = _euler_step(video_x, video_x0, sigma, sigma_next)
        audio_x = _euler_step(audio_x, audio_x0, sigma, sigma_next)
        _materialize(ctx, video_x, audio_x)

    return _DenoiseOutput(video_latent=video_x, audio_latent=audio_x)


def _resize_and_center_crop(image: Image.Image, height: int, width: int) -> Image.Image:
    src_w, src_h = image.size
    scale = max(height / src_h, width / src_w)
    new_h = math.ceil(src_h * scale)
    new_w = math.ceil(src_w * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    crop_left = (new_w - width) // 2
    crop_top = (new_h - height) // 2
    return image.crop((crop_left, crop_top, crop_left + width, crop_top + height))


def _load_image_tensor(image_path: str, height: int, width: int, ctx: RuntimeContext) -> mx.array:
    image = _resize_and_center_crop(Image.open(image_path).convert("RGB"), height, width)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    tensor = ctx.array(arr).transpose(2, 0, 1)[None, ...]
    return tensor.astype(ctx.bfloat16())


def _i2v_conditionings(
    ctx: RuntimeContext,
    image_path: str,
    *,
    enc_h: int,
    enc_w: int,
    bundle_root: Path,
    load_fn: Any | None,
) -> list[VideoConditionByLatentIndex]:
    pixels = _load_image_tensor(image_path, enc_h, enc_w, ctx)
    pixels = ctx.expand_dims(pixels, axis=2)
    encoder = load_ltx23_video_encoder(bundle_root, load_fn=load_fn)
    latent = encoder.encode(pixels)
    return [VideoConditionByLatentIndex(latent=latent, frame_idx=0, strength=1.0)]


class LTX23MlxGenerator:
    """In-repo two-stage LTX 2.3 T2V/I2V generator (MLX only)."""

    def __init__(
        self,
        ctx: RuntimeContext,
        bundle_root: Path,
        config: LTXConfig | None = None,
        *,
        entry: Any | None = None,
        version_key: str | None = None,
    ):
        self.ctx = ctx
        self.bundle_root = Path(bundle_root)
        self.config = config or LTXConfig()
        self._registry_entry = entry
        self._version_key = version_key
        self._encoder: LTX23GemmaEncoder | None = None
        self._dit: LTX23X0Model | None = None
        self._video_encoder = None
        self._upsampler = None
        self._video_patchifier = VideoLatentPatchifier()
        self._audio_patchifier = AudioPatchifier()

    def load(self) -> None:
        """No-op — components load lazily in ``generate_and_save``."""

    def _log(self, on_log: Callable[[str, str], None] | None, level: str, message: str) -> None:
        if on_log:
            on_log(level, message)

    def _weight_stem(self, *, step_distill: bool, stage2_dev_refine: bool) -> str:
        if step_distill:
            return "transformer-distilled"
        if stage2_dev_refine:
            return "transformer-distilled"
        return "transformer-dev"

    def _load_dit(self, *, step_distill: bool, stage2_dev_refine: bool = False) -> LTX23X0Model:
        stem = self._weight_stem(step_distill=step_distill, stage2_dev_refine=stage2_dev_refine)
        self._log(None, "info", f"Loading LTX 2.3 transformer: {stem}")
        self._dit = load_ltx23_x0_model(
            self.ctx,
            self.bundle_root,
            self.config,
            weight_stem=stem,
            entry=self._registry_entry,
            version_key=self._version_key,
        )
        return self._dit

    def _encode_prompts(
        self,
        prompt: str,
        *,
        step_distill: bool,
    ) -> tuple[tuple[mx.array, mx.array], tuple[mx.array, mx.array] | None]:
        if self._encoder is None:
            self._encoder = LTX23GemmaEncoder(self.ctx, self.bundle_root, self.config)
        if step_distill:
            pos = self._encoder.encode(prompt)
            return pos, None
        pos, neg = self._encoder.encode_with_negative(prompt)
        return pos, neg

    def _generate_two_stage(
        self,
        *,
        prompt: str,
        width: int,
        height: int,
        num_frames: int,
        fps: float,
        seed: int,
        stage1_steps: int,
        stage2_steps: int,
        guidance: float,
        step_distill: bool,
        image_path: str | None,
        on_log: Callable[[str, str], None] | None,
    ) -> tuple[mx.array, mx.array]:
        ctx = self.ctx
        load_fn = getattr(ctx, "load_weights", None)
        low_memory = bool(getattr(self.config, "ltx_low_memory", True))

        (video_embeds, audio_embeds), neg = self._encode_prompts(prompt, step_distill=step_distill)
        _materialize(ctx, video_embeds, audio_embeds)
        if neg is not None:
            neg_video, neg_audio = neg
            _materialize(ctx, neg_video, neg_audio)
        else:
            neg_video = neg_audio = None

        if low_memory and self._encoder is not None:
            self._encoder.free()
            self._encoder = None
            _clear_cache(ctx)

        half_h, half_w = height // 2, width // 2
        f_lat, h_half, w_half = compute_video_latent_shape(num_frames, half_h, half_w)
        video_shape = (1, f_lat * h_half * w_half, 128)
        audio_t = compute_audio_token_count(num_frames, frame_rate=fps)
        audio_shape = (1, audio_t, 128)

        video_positions_1 = compute_video_positions(ctx, f_lat, h_half, w_half, frame_rate=fps)
        audio_positions = compute_audio_positions(ctx, audio_t)

        conditionings_1: list[VideoConditionByLatentIndex] = []
        if image_path:
            conditionings_1 = _i2v_conditionings(
                ctx,
                image_path,
                enc_h=h_half * 32,
                enc_w=w_half * 32,
                bundle_root=self.bundle_root,
                load_fn=load_fn,
            )

        video_state = create_noised_state(
            ctx,
            video_shape,
            conditionings=conditionings_1,
            spatial_dims=(f_lat, h_half, w_half),
            positions=video_positions_1,
            seed=seed,
            sigma=1.0,
        )
        audio_state = create_noised_state(
            ctx,
            audio_shape,
            conditionings=[],
            spatial_dims=(f_lat, h_half, w_half),
            positions=audio_positions,
            seed=seed + 1,
            sigma=1.0,
        )

        if step_distill:
            sigmas_1 = DISTILLED_SIGMAS[: stage1_steps + 1]
        else:
            sigmas_1 = ltx2_dynamic_schedule(stage1_steps, f_lat * h_half * w_half)

        model = self._load_dit(step_distill=step_distill)
        self._log(on_log, "info", f"LTX 2.3 stage 1 denoise steps={len(sigmas_1) - 1} at {half_w}x{half_h}")

        if step_distill or neg_video is None:
            output_1 = _denoise_loop(
                ctx, model, video_state, audio_state, video_embeds, audio_embeds, sigmas_1,
            )
        else:
            output_1 = _guided_denoise_loop(
                ctx,
                model,
                video_state,
                audio_state,
                video_embeds,
                audio_text_embeds=audio_embeds,
                neg_video_embeds=neg_video,
                neg_audio_embeds=neg_audio,
                sigmas=sigmas_1,
                video_cfg=float(guidance if guidance > 0 else _DEFAULT_CFG),
                audio_cfg=_DEFAULT_AUDIO_CFG,
            )

        if low_memory:
            _clear_cache(ctx)

        gen_tokens_1 = output_1.video_latent[:, : f_lat * h_half * w_half, :]
        video_half = self._video_patchifier.unpatchify(gen_tokens_1, (f_lat, h_half, w_half), ctx)

        if self._video_encoder is None:
            self._video_encoder = load_ltx23_video_encoder(self.bundle_root, load_fn=load_fn)
        if self._upsampler is None:
            self._upsampler = load_ltx23_latent_upsampler(self.bundle_root, load_fn=load_fn)

        video_mlx = video_half.transpose(0, 2, 3, 4, 1)
        video_denorm = self._video_encoder.denormalize_latent(video_mlx)
        video_denorm = video_denorm.transpose(0, 4, 1, 2, 3)
        video_upscaled = self._upsampler(video_denorm)
        video_up_mlx = video_upscaled.transpose(0, 2, 3, 4, 1)
        video_upscaled = self._video_encoder.normalize_latent(video_up_mlx)
        video_upscaled = video_upscaled.transpose(0, 4, 1, 2, 3)
        _materialize(ctx, video_upscaled)

        h_full = h_half * 2
        w_full = w_half * 2

        conditionings_2: list[VideoConditionByLatentIndex] = []
        if image_path:
            conditionings_2 = _i2v_conditionings(
                ctx,
                image_path,
                enc_h=h_full * 32,
                enc_w=w_full * 32,
                bundle_root=self.bundle_root,
                load_fn=load_fn,
            )

        if low_memory:
            self._video_encoder = None
            self._upsampler = None
            self._dit = None
            _clear_cache(ctx)

        if not step_distill:
            model = self._load_dit(step_distill=False, stage2_dev_refine=True)

        video_tokens, _ = self._video_patchifier.patchify(video_upscaled, ctx)
        sigmas_2 = STAGE_2_SIGMAS[: stage2_steps + 1]
        start_sigma = sigmas_2[0]

        video_positions_2 = compute_video_positions(ctx, f_lat, h_full, w_full, frame_rate=fps)
        video_state_2 = create_noised_state(
            ctx,
            video_tokens.shape,
            conditionings=conditionings_2,
            spatial_dims=(f_lat, h_full, w_full),
            positions=video_positions_2,
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=video_tokens,
        )
        audio_state_2 = create_noised_state(
            ctx,
            output_1.audio_latent.shape,
            conditionings=[],
            spatial_dims=(f_lat, h_full, w_full),
            positions=audio_positions,
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=output_1.audio_latent,
        )

        if model is None:
            model = self._load_dit(step_distill=step_distill)

        self._log(on_log, "info", f"LTX 2.3 stage 2 denoise steps={len(sigmas_2) - 1} at {w_full}x{h_full}")
        output_2 = _denoise_loop(
            ctx, model, video_state_2, audio_state_2, video_embeds, audio_embeds, sigmas_2,
        )

        if low_memory:
            self._dit = None
            _clear_cache(ctx)

        gen_tokens_2 = output_2.video_latent[:, : f_lat * h_full * w_full, :]
        video_latent = self._video_patchifier.unpatchify(gen_tokens_2, (f_lat, h_full, w_full), ctx)
        audio_latent = self._audio_patchifier.unpatchify(output_2.audio_latent)
        return video_latent, audio_latent

    def generate_and_save(
        self,
        *,
        prompt: str,
        output_path: str,
        width: int,
        height: int,
        num_frames: int,
        fps: float,
        seed: int,
        steps: int,
        guidance: float,
        step_distill: bool,
        image_path: str | None,
        on_log: Callable[[str, str], None] | None,
    ) -> str:
        if getattr(self.ctx, "backend", None) != "mlx":
            raise RuntimeError(
                f"LTX 2.3 requires MLX runtime (got {getattr(self.ctx, 'backend', None)!r})"
            )

        stage2_steps = int(getattr(self.config, "ltx_stage2_steps", 3) or 3)
        stage1_steps = max(1, int(steps))
        mode = "distilled" if step_distill else "dev"
        i2v = "i2v" if image_path else "t2v"
        self._log(
            on_log,
            "info",
            " ".join(
                [
                    f"LTX 2.3 MLX pipeline={mode}",
                    f"mode={i2v}",
                    f"size={width}x{height}",
                    f"frames={num_frames}",
                    f"fps={fps}",
                    f"seed={seed}",
                    f"stage1_steps={stage1_steps}",
                    f"stage2_steps={stage2_steps}",
                    f"bundle={self.bundle_root.name}",
                ]
            ),
        )

        video_latent, audio_latent = self._generate_two_stage(
            prompt=prompt,
            width=int(width),
            height=int(height),
            num_frames=int(num_frames),
            fps=float(fps),
            seed=int(seed),
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            guidance=float(guidance),
            step_distill=bool(step_distill),
            image_path=image_path,
            on_log=on_log,
        )

        _materialize(self.ctx, video_latent, audio_latent)
        self._log(on_log, "info", f"LTX 2.3 decode+mux → {output_path}")

        def _decode_log(msg: str) -> None:
            self._log(on_log, "info", msg)

        return decode_ltx23_av_to_mp4(
            self.ctx,
            video_latent,
            audio_latent,
            output_path,
            self.bundle_root,
            frame_rate=float(fps),
            on_log=_decode_log,
        )
