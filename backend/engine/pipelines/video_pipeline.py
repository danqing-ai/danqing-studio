"""
VideoPipeline — 视频请求 → 模型推理 → 资产落盘。

MLX 操作（文本编码 + 模型加载 + 去噪 + VAE 解码）在单线程执行器中执行。
进度回调和结果处理在事件循环线程中运行。

完全后端无关，与 ImagePipeline 保持一致的架构模式。
"""
from __future__ import annotations

import random
import subprocess

import numpy as np
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from backend.core.contracts import (
    ExecutionContext, VideoGenerationRequest, VideoEditRequest,
    LogEvent, ProgressEvent, parse_model_version, parse_size,
)
from backend.engine.common.cache import ModelCache
from backend.engine.common.pipeline_registry import (
    local_bundle_root as _local_bundle_root_fn,
    registry_scalar_default as _registry_scalar_default_fn,
    resolve_project_path as _resolve_project_path_fn,
    resolve_version_block as _resolve_version_block_fn,
)
from backend.engine.common.schedulers import get_scheduler
from backend.engine.common.text_encoders import T5Encoder
from backend.engine._transformer_registry import (
    encode_video_prompt as _encode_video_prompt_fn,
    get_video_transformer_class as _get_video_transformer_class,
    get_video_weight_remap as _get_video_weight_remap,
)
from backend.engine.config.model_configs import get_config_class


def _video_post_denoise_clear_cache(family: str) -> bool:
    """Families whose DiT peak memory should be released before VAE decode."""
    return family in ("hunyuan", "cogvideox")


def _timestep_embed_schedule_from_scheduler(scheduler: Any) -> list[float] | None:
    """Match :class:`ImagePipeline`: ``FlowMatchEulerScheduler.set_timesteps`` returns step *indices*;

    continuous noise-level values for time-MLP live on ``scheduler.timesteps``.
    """
    sched_ts = getattr(scheduler, "timesteps", None)
    if sched_ts is None:
        return None
    arr = np.asarray(sched_ts, dtype=np.float64).reshape(-1)
    return [float(x) for x in arr.tolist()]

from backend.engine.families.ltx.weights import restore_diffusers_names_from_mlx_forge_ltx
from backend.engine.pipelines.image_pipeline import (
    _image_pipeline_emit_complete,
    _image_pipeline_emit_denoise_progress,
    _image_pipeline_emit_post_progress,
    _t5_encoder_bundle_paths,
)
from backend.engine.pipelines.video_bundle_layout import (
    looks_like_mlx_forge_ltx_transformer_keys,
    ltx_flat_vae_decoder_file,
    max_remapped_ltx_block_index,
    resolve_video_transformer_weight_sources,
)
from backend.engine.runtime._base import RuntimeContext


class VideoPipeline:
    """视频生成管线 — 后端无关，与 ImagePipeline 对等。

    同步 run() 方法，由 DanQingVideoEngine 通过 asyncio.to_thread 在线程池中调用。
    """

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
        self._t5: T5Encoder | None = None

    # ------------------------------------------------------------------
    # Path / registry helpers (mirrors ImagePipeline)
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

    def _inject_hunyuan_text_encoder_paths(self, entry, config) -> None:
        """Resolve registry-declared native TE roots (ModelScope Qwen + ByT5)."""
        VideoPipeline.apply_hunyuan_text_encoder_paths(
            entry, config, self._project_root,
        )

    @staticmethod
    def apply_hunyuan_text_encoder_paths(entry, config, project_root: Path) -> None:
        for param_key in ("text_encoder_qwen_local", "text_encoder_byt5_local"):
            val = _registry_scalar_default_fn(entry, param_key, None)
            if val is not None and str(val).strip():
                resolved = _resolve_project_path_fn(project_root, str(val).strip())
                setattr(config, param_key, str(resolved))
        rel = _registry_scalar_default_fn(entry, "text_encoder_release_after_encode", None)
        if rel is not None:
            setattr(config, "text_encoder_release_after_encode", bool(rel))

    def _resolved_original_video_bundle_root(self, entry) -> Path | None:
        """Registry ``versions.original.local_path`` for the same model (T5 fallback for MLX-only trees)."""
        raw = getattr(entry, "raw", {}) or {}
        versions = raw.get("versions") or {}
        ob = versions.get("original")
        if not isinstance(ob, dict):
            return None
        lp = (ob.get("local_path") or "").strip()
        if not lp:
            return None
        p = self._resolve_path(lp)
        return p if p.is_dir() else None

    def _effective_t5_bundle_root(self, entry, bundle_root: Path | None) -> Path | None:
        """Prefer current version bundle; if T5 dirs are missing (typical MLX-forge flat HF), use ``original``."""
        if bundle_root is None or not bundle_root.is_dir():
            return None
        family = getattr(entry, "family", "")
        try:
            if family == "wan":
                self._wan_t5_encoder_bundle_paths(bundle_root)
            else:
                _t5_encoder_bundle_paths(bundle_root)
            return bundle_root
        except RuntimeError as err:
            alt = self._resolved_original_video_bundle_root(entry)
            if alt is not None:
                if family == "wan":
                    self._wan_t5_encoder_bundle_paths(alt)
                else:
                    _t5_encoder_bundle_paths(alt)
                return alt
            raise RuntimeError(
                f"T5 text encoder assets not found under {bundle_root}, "
                f"and no installed ``original`` registry version for ``{entry.id}``."
                + (
                    " Wan bundles require ``models_t5*.pth`` and ``google/umt5-xxl``."
                    if family == "wan"
                    else " Install a full model bundle with ``text_encoder`` + ``tokenizer``."
                )
            ) from err

    @staticmethod
    def _wan_t5_encoder_bundle_paths(bundle_root: Path) -> tuple[str, str]:
        """Wan original bundle: ``models_t5*.pth`` + ``google/umt5-xxl`` tokenizer."""
        from backend.engine.families.wan.text_encoder_mlx import resolve_wan_umt5_pth

        assets = resolve_wan_umt5_pth(bundle_root)
        if assets is None:
            raise RuntimeError(
                f"Wan T5 assets not found under {bundle_root}. Expected "
                f"``models_t5_umt5-xxl-enc-bf16.pth`` (or ``models_t5*.pth``) and "
                f"``google/umt5-xxl`` tokenizer."
            )
        pth_path, tok_dir = assets
        return str(pth_path), str(tok_dir)

    def _resolve_guidance_default(self, entry) -> float:
        g = self._registry_scalar_default(entry, "guidance", None)
        if g is not None:
            return float(g)
        gs = self._registry_scalar_default(entry, "guide_scale", None)
        if gs is not None:
            return float(gs)
        return 0.0

    def _resolve_num_frames(self, request: VideoGenerationRequest | VideoEditRequest, entry) -> int:
        if request.num_frames is not None:
            return int(request.num_frames)
        reg = self._registry_scalar_default(entry, "num_frames", None)
        if reg is not None:
            return int(reg)
        return 81

    def _resolve_fps(self, request: VideoGenerationRequest | VideoEditRequest, entry) -> int:
        if request.fps is not None:
            return int(request.fps)
        reg = self._registry_scalar_default(entry, "fps", None)
        if reg is not None:
            return int(reg)
        return 16

    def _validate_wan_umt5_embeddings(
        self,
        family: str,
        txt_embeds: Any | None,
        on_log: Callable[[str, str], None] | None,
    ) -> None:
        if not on_log or family != "wan" or txt_embeds is None:
            return
        self.ctx.eval(txt_embeds)
        peak = float(self.ctx.sqrt(self.ctx.max(self.ctx.square(txt_embeds))))
        if peak < 1e-3:
            raise RuntimeError(
                "Wan UMT5 embeddings are near zero; text encoder weights may not be loaded"
            )
        on_log("info", f"Wan UMT5 text embeddings ready (peak={peak:.3f})")

    def _validate_generate_geometry(
        self, family: str, config: Any, w: int, h: int, num_frames: int,
    ) -> None:
        if w <= 0 or h <= 0:
            raise RuntimeError(f"invalid video size {w}x{h}")
        vae_sf = int(getattr(config, "vae_scale", 8) or 8)
        if family == "cogvideox":
            tvs = int(getattr(config, "temporal_vae_scale", 4) or 4)
            if (num_frames - 1) % tvs != 0:
                raise RuntimeError(
                    f"CogVideoX requires (num_frames - 1) % {tvs} == 0; got {num_frames} "
                    f"(valid examples: 49, 45, 41 for temporal VAE scale {tvs})"
                )
        if family == "wan":
            tvs = int(getattr(config, "temporal_vae_scale", 4) or 4)
            if (num_frames - 1) % tvs != 0:
                raise RuntimeError(
                    f"Wan requires (num_frames - 1) % {tvs} == 0; got {num_frames} "
                    f"(valid examples: 81, 49 for temporal VAE scale {tvs})"
                )
            _, ph, pw = getattr(config, "patch_size", (1, 2, 2))
            step_h = vae_sf * int(ph)
            step_w = vae_sf * int(pw)
            if h % step_h != 0 or w % step_w != 0:
                raise RuntimeError(
                    f"Wan requires height % {step_h} == 0 and width % {step_w} == 0 "
                    f"(vae_scale={vae_sf}, patch_size={getattr(config, 'patch_size', (1, 2, 2))}); "
                    f"got {w}x{h}. Example valid size: 480x704 for 480x720 intent."
                )
        if w % vae_sf != 0 or h % vae_sf != 0:
            raise RuntimeError(
                f"video width and height must be divisible by vae_scale={vae_sf}; got {w}x{h}"
            )
        tvs = getattr(config, "temporal_vae_scale", None)
        if tvs is not None and int(tvs) > 0:
            if (num_frames - 1) % int(tvs) != 0:
                raise RuntimeError(
                    f"video requires (num_frames - 1) % {int(tvs)} == 0; got {num_frames}"
                )

    def _snap_wan_pixel_dims_if_needed(
        self,
        family: str,
        config: Any,
        w: int,
        h: int,
        *,
        on_log: Callable | None = None,
    ) -> tuple[int, int]:
        if family != "wan":
            return w, h
        from backend.engine.families.wan.conditioning import snap_wan_pixel_dims

        vae_scale = int(getattr(config, "vae_scale", 16) or 16)
        patch_size = tuple(getattr(config, "patch_size", (1, 2, 2)))
        ow, oh = w, h
        w, h = snap_wan_pixel_dims(w, h, vae_scale=vae_scale, patch_size=patch_size)
        if (w, h) != (ow, oh) and on_log is not None:
            on_log("info", f"Wan adjusted output size {ow}x{oh} -> {w}x{h} for patch/VAE alignment")
        return w, h

    def _prepare_t5_context(self, family: str, config: Any) -> None:
        if family == "cogvideox":
            self._t5_max_seq_len = int(getattr(config, "max_text_seq_length", 226))
        elif family == "wan":
            self._t5_max_seq_len = int(getattr(config, "text_len", 512))
        else:
            self._t5_max_seq_len = 512
        self._t5 = None

    def _denoise_video(
        self,
        *,
        latents: Any,
        timesteps: Any,
        scheduler: Any,
        model: Any,
        txt_embeds: Any,
        neg_embeds: Any,
        guidance: float,
        config: Any,
        sigmas: Any,
        timestep_embed_schedule: list[float] | None,
        extra_cond: dict,
        rope_kw: dict,
        cfg_renorm: bool,
        cfg_renorm_min: float,
        ctx_exec: ExecutionContext,
        on_progress: Callable | None,
        on_log: Callable | None,
    ) -> Any | None:
        from backend.engine.common.schedulers import CogVideoXDPMScheduler

        init_sigma = float(getattr(scheduler, "init_noise_sigma", 1.0))
        if init_sigma != 1.0:
            latents = latents * init_sigma

        timesteps_list: list[int] = []
        if isinstance(scheduler, CogVideoXDPMScheduler):
            timesteps_list = list(getattr(scheduler, "_timesteps_list", []) or [])

        old_pred_original_sample = None
        n_steps = len(timesteps)

        for i, t in enumerate(timesteps):
            if ctx_exec.cancel_token.is_cancelled():
                return None

            latents_in = (
                scheduler.scale_model_input(latents, t)
                if hasattr(scheduler, "scale_model_input")
                else latents
            )

            model_kwargs: dict[str, Any] = {"txt_embeds": txt_embeds} if txt_embeds is not None else {}
            if extra_cond.get("txt_attn_mask") is not None:
                model_kwargs["txt_attn_mask"] = extra_cond["txt_attn_mask"]
            if extra_cond.get("txt_embeds_2") is not None:
                model_kwargs["txt_embeds_2"] = extra_cond["txt_embeds_2"]
            if extra_cond.get("txt_attn_mask_2") is not None:
                model_kwargs["txt_attn_mask_2"] = extra_cond["txt_attn_mask_2"]
            model_kwargs.update(rope_kw)
            skip_keys = {
                "txt_attn_mask", "txt_embeds_2", "txt_attn_mask_2",
                "neg_txt_attn_mask", "neg_txt_embeds_2", "neg_txt_attn_mask_2",
                "cond_latents", "mask_concat", "i2v_mode",
                "wan_i2v", "wan_cond_latent", "wan_i2v_mask", "wan_seq_len",
                "wan_expand_timesteps", "wan_bundle_root", "wan_size",
            }
            for k, v in extra_cond.items():
                if k not in skip_keys:
                    model_kwargs[k] = v
            if sigmas is not None:
                model_kwargs["sigmas"] = sigmas
            if timestep_embed_schedule is not None and i < len(timestep_embed_schedule):
                model_kwargs["timestep_embed_value"] = timestep_embed_schedule[i]
            wan_seq = int(extra_cond.get("wan_seq_len", 0))
            if wan_seq > 0:
                model_kwargs["seq_len"] = wan_seq
            if extra_cond.get("wan_expand_timesteps"):
                seq_len = int(extra_cond.get("wan_seq_len", 0))
                if seq_len > 0 and hasattr(model, "build_timestep_per_token"):
                    model_kwargs["timestep_per_token"] = model.build_timestep_per_token(
                        t if getattr(t, "ndim", 0) > 0 else self.ctx.array([float(t)]),
                        seq_len,
                        extra_cond.get("wan_i2v_mask"),
                    )

            if neg_embeds is not None and getattr(config, "supports_guidance", False):
                neg_kwargs: dict[str, Any] = {"txt_embeds": neg_embeds}
                if extra_cond.get("neg_txt_attn_mask") is not None:
                    neg_kwargs["txt_attn_mask"] = extra_cond["neg_txt_attn_mask"]
                if extra_cond.get("neg_txt_embeds_2") is not None:
                    neg_kwargs["txt_embeds_2"] = extra_cond["neg_txt_embeds_2"]
                if extra_cond.get("neg_txt_attn_mask_2") is not None:
                    neg_kwargs["txt_attn_mask_2"] = extra_cond["neg_txt_attn_mask_2"]
                neg_kwargs.update(rope_kw)
                for k, v in extra_cond.items():
                    if k not in skip_keys:
                        neg_kwargs[k] = v
                if sigmas is not None:
                    neg_kwargs["sigmas"] = sigmas
                if timestep_embed_schedule is not None and i < len(timestep_embed_schedule):
                    neg_kwargs["timestep_embed_value"] = timestep_embed_schedule[i]
                if wan_seq > 0:
                    neg_kwargs["seq_len"] = wan_seq
                if extra_cond.get("wan_expand_timesteps"):
                    seq_len = int(extra_cond.get("wan_seq_len", 0))
                    if seq_len > 0 and hasattr(model, "build_timestep_per_token"):
                        neg_kwargs["timestep_per_token"] = model.build_timestep_per_token(
                            t if getattr(t, "ndim", 0) > 0 else self.ctx.array([float(t)]),
                            seq_len,
                            extra_cond.get("wan_i2v_mask"),
                        )
                if hasattr(model, "predict_noise_cfg"):
                    noise_pred = model.predict_noise_cfg(
                        latents_in,
                        t,
                        guidance=guidance,
                        pos_kwargs=model_kwargs,
                        neg_kwargs=neg_kwargs,
                        cfg_renorm=cfg_renorm,
                        cfg_renorm_min=cfg_renorm_min,
                    )
                else:
                    noise_cond = model(latents_in, t, **model_kwargs)
                    noise_uncond = model(latents_in, t, **neg_kwargs)
                    noise_pred = model.combine_cfg_noise(noise_cond, noise_uncond, guidance)
                    if cfg_renorm:
                        noise_pred = model.refine_cfg_noise(
                            noise_cond, noise_pred, cfg_renorm_min=cfg_renorm_min,
                        )
            else:
                noise_pred = model(latents_in, t, **model_kwargs)

            if getattr(self.ctx, "backend", None) == "mlx":
                self.ctx.eval(noise_pred)

            if isinstance(scheduler, CogVideoXDPMScheduler):
                t_back = timesteps_list[i - 1] if i > 0 and timesteps_list else None
                latents, old_pred_original_sample = scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    old_pred_original_sample=old_pred_original_sample,
                    timestep_back=t_back,
                )
            else:
                latents = scheduler.step(noise_pred, t, latents)

            if hasattr(model, "reblend_i2v_latents"):
                latents = model.reblend_i2v_latents(latents)

            if getattr(self.ctx, "backend", None) == "mlx":
                self.ctx.eval(latents)

            model.step_callback(i, latents, noise_pred)

            if on_progress:
                _image_pipeline_emit_denoise_progress(on_progress, i + 1, n_steps)
            if on_log:
                on_log("info", f"Step {i+1}/{n_steps}")

        return latents

    # ------------------------------------------------------------------
    # 主入口 — 同步 run()，由 asyncio.to_thread 调用
    # ------------------------------------------------------------------

    def run(
        self,
        request: VideoGenerationRequest,
        ctx_exec: ExecutionContext,
        *,
        on_progress: Callable | None = None,
        on_log: Callable | None = None,
    ):
        """执行视频生成管线（同步，在线程池内调用）。

        Returns: ``(output_path, metadata_dict)`` 或 ``None``（取消）。
        """
        model_key, version_key = parse_model_version(request.model)
        w, h = parse_size(request.size)
        entry = self._registry.require(model_key)
        self._current_entry = entry
        config_cls = get_config_class(entry.family)
        config = config_cls()
        family = getattr(entry, "family", "ltx")
        num_frames = self._resolve_num_frames(request, entry)
        fps = self._resolve_fps(request, entry)
        seed = request.seed if request.seed is not None else random.randint(0, 2 ** 32 - 1)

        # ── Registry-driven parameter injection ──
        for param_key in ("vae_scale", "default_scheduler", "text_encoder_device", "vae_temporal_chunk_size"):
            val = self._registry_scalar_default(entry, param_key, None)
            if val is not None:
                setattr(config, param_key, val)

        sg = self._registry_scalar_default(entry, "supports_guidance", None)
        if sg is not None:
            config.supports_guidance = bool(sg)
        sd = self._registry_scalar_default(entry, "step_distill", None)
        if sd is not None:
            config.step_distill = bool(sd)
        vst = self._registry_scalar_default(entry, "vae_spatial_tiling", None)
        if vst is not None:
            config.vae_spatial_tiling = bool(vst)
        if family == "hunyuan":
            self._inject_hunyuan_text_encoder_paths(entry, config)

        if ctx_exec.cancel_token.is_cancelled():
            return None

        bundle_root = self._local_bundle_root(entry, version_key or None)
        self._merge_video_transformer_bundle_config(family, bundle_root, config)
        w, h = self._snap_wan_pixel_dims_if_needed(family, config, w, h, on_log=on_log)
        self._validate_generate_geometry(family, config, w, h, num_frames)
        encoder_type = self._resolve_video_encoder_type(config, family)
        if encoder_type == "t5":
            self._t5_bundle_root = self._effective_t5_bundle_root(entry, bundle_root)
            self._prepare_t5_context(family, config)
        else:
            self._t5_bundle_root = bundle_root

        steps_default = self._registry_scalar_default(entry, "steps", 40)
        guidance_default = self._resolve_guidance_default(entry)
        scheduler_registry = self._registry_scalar_default(entry, "scheduler", None)
        scheduler_default = scheduler_registry or getattr(config, "default_scheduler", "unipc")

        steps = int(request.steps) if request.steps is not None else int(steps_default)
        steps = max(1, steps)
        step_distill = bool(
            getattr(config, "step_distill", False)
            or self._registry_scalar_default(entry, "step_distill", False)
        )
        guidance = float(request.guidance) if request.guidance is not None else float(guidance_default)
        if step_distill or not getattr(config, "supports_guidance", True):
            guidance = 0.0

        # 1. Text encoding
        txt_embeds = None
        neg_embeds = None
        txt_mask = txt_mask_2 = neg_mask = neg_mask_2 = None
        neg_embeds_2 = None
        txt_embeds_2 = None
        if request.prompt and config.text_dim > 0:
            (
                txt_embeds, txt_mask, txt_embeds_2, txt_mask_2,
                neg_embeds, neg_mask, neg_embeds_2, neg_mask_2,
            ) = self._encode_video_text_with_cfg(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                config=config,
                family=family,
                bundle_root=bundle_root,
                guidance=guidance,
            )
            self._validate_wan_umt5_embeddings(family, txt_embeds, on_log)

        if ctx_exec.cancel_token.is_cancelled():
            return None

        self._release_video_t5_after_encode(family, encoder_type)

        # 2. Load model (registry-driven, zero family branching)
        latent_frames = self._latent_frame_count(config, num_frames)
        model = self._load_model(family, config, entry, version_key or None, latent_frames)
        if model is None:
            raise RuntimeError(f"Failed to load model: {model_key}")

        # ── Hook ①: after weight loading (LoRA / Adapter merging) ──
        model.after_load_weights(bundle_root=str(bundle_root) if bundle_root else None)

        # ── Hook ②: condition preparation ──
        extra_cond = model.prepare_conditioning(request,
                                                bundle_root=str(bundle_root) if bundle_root else None)
        extra_cond = self._apply_video_text_to_extra_cond(
            extra_cond,
            txt_embeds=txt_embeds,
            txt_mask=txt_mask,
            txt_embeds_2=txt_embeds_2,
            txt_mask_2=txt_mask_2,
            neg_embeds=neg_embeds,
            neg_mask=neg_mask,
            neg_embeds_2=neg_embeds_2,
            neg_mask_2=neg_mask_2,
        )

        # 3. Scheduler
        scheduler = self._create_video_scheduler(
            family=family,
            scheduler_name=scheduler_default,
            bundle_root=bundle_root,
        )
        vae_scale = getattr(config, "vae_scale", 8)
        wan_shift: float | None = None
        if step_distill and scheduler_default == "flow_match_euler":
            import numpy as np
            sigmas_arr = np.linspace(1.0, 0.0, steps + 1, dtype=np.float32)[:-1]
            timesteps = scheduler.set_timesteps(steps, use_empirical_mu=False)
            scheduler._sigmas = self.ctx.concat([
                self.ctx.array(sigmas_arr, dtype=self.ctx.float32()),
                self.ctx.zeros((1,), dtype=self.ctx.float32()),
            ], axis=0)
            scheduler._timesteps = self.ctx.array(
                sigmas_arr * float(scheduler.num_train_timesteps),
                dtype=self.ctx.float32(),
            )
        else:
            sched_kwargs: dict[str, Any] = {}
            shift_default = self._registry_scalar_default(entry, "shift", None)
            shift_val = float(request.shift) if request.shift is not None else (
                float(shift_default) if shift_default is not None else None
            )
            if shift_val is not None:
                sched_kwargs["shift"] = shift_val
            timesteps = scheduler.set_timesteps(steps, **sched_kwargs)
            if family == "wan":
                wan_shift = float(sched_kwargs.get("shift", getattr(scheduler, "_default_shift", 1.0)))
        sigmas = getattr(scheduler, 'sigmas', None)
        timestep_embed_schedule = _timestep_embed_schedule_from_scheduler(scheduler)
        _cfg_renorm = bool(self._registry_scalar_default(entry, "enable_cfg_renorm", False))
        _cfg_renorm_min = float(self._registry_scalar_default(entry, "cfg_renorm_min", 0.0))

        if on_log:
            parts = [
                f"infer model={model_key}",
                f"family={family}",
                f"version={version_key or 'default'}",
                f"size={w}x{h}",
                f"frames={num_frames}",
                f"fps={fps}",
                f"seed={seed}",
                f"steps={steps}",
                f"guidance={guidance}",
                f"scheduler={scheduler_default}",
                f"supports_guidance={getattr(config, 'supports_guidance', False)}",
                f"cfg_on={bool(neg_embeds is not None)}",
                f"vae_scale={vae_scale}",
                "mode=video_generate",
            ]
            if wan_shift is not None:
                parts.append(f"shift={wan_shift}")
            if _cfg_renorm:
                parts.append(f"cfg_renorm=True cfg_renorm_min={_cfg_renorm_min}")
            if family == "cogvideox" and hasattr(scheduler, "_prediction_type"):
                parts.append(f"sched_prediction={scheduler._prediction_type}")
                parts.append(f"sched_spacing={getattr(scheduler, '_timestep_spacing', '?')}")
            on_log("info", " ".join(parts))

        # 4. Initial noise [B, C, T_latent, H_lat, W_lat]
        latent_c = int(getattr(config, "vae_z_dim", None) or config.dim_in)
        latent_shape = (1, latent_c, latent_frames, h // vae_scale, w // vae_scale)
        if seed is not None:
            latents = self.ctx.seeded_randn(latent_shape, seed, dtype=self.ctx.float32())
        else:
            latents = self.ctx.randn(latent_shape, dtype=self.ctx.float32())

        # ── Hook ③: before denoise ──
        latents, extra_cond = model.before_denoise(latents, timesteps, sigmas, **extra_cond)
        rope_kw = self._cogvideox_rotary_model_kwargs(family, config, h, w, latents)

        latents = self._denoise_video(
            latents=latents,
            timesteps=timesteps,
            scheduler=scheduler,
            model=model,
            txt_embeds=txt_embeds,
            neg_embeds=neg_embeds,
            guidance=guidance,
            config=config,
            sigmas=sigmas,
            timestep_embed_schedule=timestep_embed_schedule,
            extra_cond=extra_cond,
            rope_kw=rope_kw,
            cfg_renorm=_cfg_renorm,
            cfg_renorm_min=_cfg_renorm_min,
            ctx_exec=ctx_exec,
            on_progress=on_progress,
            on_log=on_log,
        )
        if latents is None:
            return None

        if ctx_exec.cancel_token.is_cancelled():
            return None

        if getattr(self.ctx, "backend", None) == "mlx":
            if on_log:
                on_log("info", "Materializing denoised latents for VAE decode...")
            self.ctx.eval(latents)
            if _video_post_denoise_clear_cache(family):
                self.ctx.clear_cache()

        n_steps = len(timesteps)
        if on_log:
            on_log("info", "Decoding video latents (VAE)...")
        _image_pipeline_emit_post_progress(on_progress, n_steps=n_steps, within_post=0.1)

        # 6. VAE decode (frame-by-frame for video latents)
        def _vae_post_progress(frac: float) -> None:
            _image_pipeline_emit_post_progress(
                on_progress, n_steps=n_steps, within_post=0.1 + 0.75 * min(1.0, max(0.0, frac)),
            )

        def _vae_post_log(msg: str) -> None:
            if on_log:
                on_log("info", msg)

        frames = self._vae_decode_video(
            latents, entry, version_key or None,
            on_post_progress=_vae_post_progress,
            on_post_log=_vae_post_log,
        )

        if ctx_exec.cancel_token.is_cancelled():
            return None

        _image_pipeline_emit_post_progress(on_progress, n_steps=n_steps, within_post=0.85)
        if on_log:
            on_log("info", f"Saving video ({len(frames)} frames)...")

        # 7. Save (task work dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work = Path(ctx_exec.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        out_path = str(work / f"{model_key}_{seed}_{timestamp}.mp4")
        self._save_video(frames, out_path, fps=fps)

        _image_pipeline_emit_complete(on_progress, n_steps)

        metadata = {
            "model": request.model, "seed": seed,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt or "",
            "steps": steps,
            "guidance": guidance, "num_frames": num_frames,
            "fps": fps, "width": w, "height": h,
            "mime_type": "video/mp4",
        }

        return str(out_path), metadata

    def run_edit(
        self,
        request: VideoEditRequest,
        ctx_exec: ExecutionContext,
        *,
        on_progress: Callable | None = None,
        on_log: Callable | None = None,
    ):
        """执行视频编辑管线（同步，在线程池内调用）。

        当前仅 animate 操作（图生视频）。
        """
        model_key, version_key = parse_model_version(request.model)
        w, h = parse_size(request.size)
        entry = self._registry.require(model_key)
        self._current_entry = entry
        config_cls = get_config_class(entry.family)
        config = config_cls()
        family = getattr(entry, "family", "ltx")
        num_frames = self._resolve_num_frames(request, entry)
        fps = self._resolve_fps(request, entry)
        seed = request.seed if request.seed is not None else random.randint(0, 2 ** 32 - 1)

        # Registry-driven parameter injection
        for param_key in ("vae_scale", "default_scheduler", "text_encoder_device", "vae_temporal_chunk_size"):
            val = self._registry_scalar_default(entry, param_key, None)
            if val is not None:
                setattr(config, param_key, val)
        sg = self._registry_scalar_default(entry, "supports_guidance", None)
        if sg is not None:
            config.supports_guidance = bool(sg)
        sd = self._registry_scalar_default(entry, "step_distill", None)
        if sd is not None:
            config.step_distill = bool(sd)
        vst = self._registry_scalar_default(entry, "vae_spatial_tiling", None)
        if vst is not None:
            config.vae_spatial_tiling = bool(vst)
        if family == "hunyuan":
            self._inject_hunyuan_text_encoder_paths(entry, config)

        if ctx_exec.cancel_token.is_cancelled():
            return None

        bundle_root = self._local_bundle_root(entry, version_key or None)
        self._merge_video_transformer_bundle_config(family, bundle_root, config)
        w, h = self._snap_wan_pixel_dims_if_needed(family, config, w, h, on_log=on_log)
        self._validate_generate_geometry(family, config, w, h, num_frames)
        encoder_type = self._resolve_video_encoder_type(config, family)
        if encoder_type == "t5":
            self._t5_bundle_root = self._effective_t5_bundle_root(entry, bundle_root)
            self._prepare_t5_context(family, config)
        else:
            self._t5_bundle_root = bundle_root

        steps_default = self._registry_scalar_default(entry, "steps", 40)
        guidance_default = self._resolve_guidance_default(entry)
        scheduler_registry = self._registry_scalar_default(entry, "scheduler", None)
        scheduler_default = scheduler_registry or getattr(config, "default_scheduler", "unipc")

        steps = int(request.steps) if request.steps is not None else int(steps_default)
        steps = max(1, steps)
        step_distill = bool(
            getattr(config, "step_distill", False)
            or self._registry_scalar_default(entry, "step_distill", False)
        )
        guidance = float(request.guidance) if request.guidance is not None else float(guidance_default)
        if step_distill or not getattr(config, "supports_guidance", True):
            guidance = 0.0

        # 1. Text encoding
        txt_embeds = None
        neg_embeds = None
        txt_mask = txt_mask_2 = neg_mask = neg_mask_2 = None
        neg_embeds_2 = None
        txt_embeds_2 = None
        if request.prompt and config.text_dim > 0:
            (
                txt_embeds, txt_mask, txt_embeds_2, txt_mask_2,
                neg_embeds, neg_mask, neg_embeds_2, neg_mask_2,
            ) = self._encode_video_text_with_cfg(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                config=config,
                family=family,
                bundle_root=bundle_root,
                guidance=guidance,
            )
            self._validate_wan_umt5_embeddings(family, txt_embeds, on_log)

        if ctx_exec.cancel_token.is_cancelled():
            return None

        self._release_video_t5_after_encode(family, encoder_type)

        # 2. Load model
        latent_frames = self._latent_frame_count(config, num_frames)
        model = self._load_model(family, config, entry, version_key or None, latent_frames)
        if model is None:
            raise RuntimeError(f"Failed to load model: {model_key}")

        model.after_load_weights(bundle_root=str(bundle_root) if bundle_root else None)

        extra_cond = model.prepare_conditioning(request,
                                                bundle_root=str(bundle_root) if bundle_root else None)
        extra_cond = self._apply_video_text_to_extra_cond(
            extra_cond,
            txt_embeds=txt_embeds,
            txt_mask=txt_mask,
            txt_embeds_2=txt_embeds_2,
            txt_mask_2=txt_mask_2,
            neg_embeds=neg_embeds,
            neg_mask=neg_mask,
            neg_embeds_2=neg_embeds_2,
            neg_mask_2=neg_mask_2,
        )

        # 3. Scheduler
        scheduler = self._create_video_scheduler(
            family=family,
            scheduler_name=scheduler_default,
            bundle_root=bundle_root,
        )
        vae_scale = getattr(config, "vae_scale", 8)
        if step_distill and scheduler_default == "flow_match_euler":
            import numpy as np
            sigmas_arr = np.linspace(1.0, 0.0, steps + 1, dtype=np.float32)[:-1]
            timesteps = scheduler.set_timesteps(steps, use_empirical_mu=False)
            scheduler._sigmas = self.ctx.concat([
                self.ctx.array(sigmas_arr, dtype=self.ctx.float32()),
                self.ctx.zeros((1,), dtype=self.ctx.float32()),
            ], axis=0)
            scheduler._timesteps = self.ctx.array(
                sigmas_arr * float(scheduler.num_train_timesteps),
                dtype=self.ctx.float32(),
            )
        else:
            sched_kwargs: dict[str, Any] = {}
            shift_default = self._registry_scalar_default(entry, "shift", None)
            shift_val = float(request.shift) if request.shift is not None else (
                float(shift_default) if shift_default is not None else None
            )
            if shift_val is not None:
                sched_kwargs["shift"] = shift_val
            timesteps = scheduler.set_timesteps(steps, **sched_kwargs)
        sigmas = getattr(scheduler, 'sigmas', None)
        timestep_embed_schedule = _timestep_embed_schedule_from_scheduler(scheduler)
        _cfg_renorm = bool(self._registry_scalar_default(entry, "enable_cfg_renorm", False))
        _cfg_renorm_min = float(self._registry_scalar_default(entry, "cfg_renorm_min", 0.0))

        if on_log:
            parts = [
                f"infer model={model_key}",
                f"family={family}",
                f"version={version_key or 'default'}",
                f"size={w}x{h}",
                f"frames={num_frames}",
                f"fps={fps}",
                f"seed={seed}",
                f"steps={steps}",
                f"guidance={guidance}",
                f"scheduler={scheduler_default}",
                f"supports_guidance={getattr(config, 'supports_guidance', False)}",
                f"cfg_on={bool(neg_embeds is not None)}",
                f"vae_scale={vae_scale}",
                "mode=video_edit",
            ]
            if _cfg_renorm:
                parts.append(f"cfg_renorm=True cfg_renorm_min={_cfg_renorm_min}")
            if family == "cogvideox" and hasattr(scheduler, "_prediction_type"):
                parts.append(f"sched_prediction={scheduler._prediction_type}")
                parts.append(f"sched_spacing={getattr(scheduler, '_timestep_spacing', '?')}")
            on_log("info", " ".join(parts))

        # 4. Initial noise
        latent_c = int(getattr(config, "vae_z_dim", None) or config.dim_in)
        latent_shape = (1, latent_c, latent_frames, h // vae_scale, w // vae_scale)
        if seed is not None:
            latents = self.ctx.seeded_randn(latent_shape, seed, dtype=self.ctx.float32())
        else:
            latents = self.ctx.randn(latent_shape, dtype=self.ctx.float32())

        # Load source image condition if available
        if request.source_asset_id:
            src_asset = self._asset_store.get_asset(request.source_asset_id)
            if src_asset and src_asset.file_path:
                from PIL import Image
                src_img = Image.open(src_asset.file_path).convert("RGB")
                if family == "wan":
                    from backend.engine.families.wan.conditioning import prepare_wan_reference_image

                    src_img = prepare_wan_reference_image(src_img, w, h)
                else:
                    src_img = src_img.resize((w, h))
                import numpy as np
                src_array = np.array(src_img).astype(np.float32) / 127.5 - 1.0
                src_tensor = self.ctx.array(np.expand_dims(src_array, 0))
                # VAE encode first frame
                vae_latent = self._vae_encode_frame(src_tensor, entry, version_key or None)
                if vae_latent is not None:
                    if family == "hunyuan":
                        B, C, T, H, W = latents.shape
                        cond_lat = self.ctx.zeros((B, C, T, H, W), dtype=latents.dtype)
                        cond_lat = self.ctx.concat(
                            [vae_latent[:, :, :1, :, :], cond_lat[:, :, 1:, :, :]], axis=2,
                        )
                        mask = self.ctx.zeros((B, 1, T, H, W), dtype=latents.dtype)
                        mask = mask + 0.0
                        mask_slice = self.ctx.ones((B, 1, 1, H, W), dtype=latents.dtype)
                        mask = self.ctx.concat([mask_slice, mask[:, :, 1:, :, :]], axis=2)
                        extra_cond["cond_latents"] = cond_lat
                        extra_cond["mask_concat"] = mask
                        extra_cond["i2v_mode"] = True
                    elif family == "wan":
                        from backend.engine.families.wan.conditioning import (
                            expand_wan_cond_latent,
                            masks_like,
                        )

                        z = self.ctx.squeeze(vae_latent, 0)
                        noise_cthw = self.ctx.squeeze(latents, 0)
                        z = expand_wan_cond_latent(self.ctx, z, int(noise_cthw.shape[1]))
                        _, mask2 = masks_like(self.ctx, [noise_cthw], zero=True)
                        extra_cond["wan_cond_latent"] = z
                        extra_cond["wan_i2v_mask"] = mask2[0]
                        extra_cond["wan_i2v"] = True
                    else:
                        latents = self.ctx.concat(
                            [vae_latent[:, :, :1, :, :], latents[:, :, 1:, :, :]], axis=2
                        )
                else:
                    if family == "wan":
                        raise RuntimeError(
                            "Wan image-to-video (animate) failed to VAE-encode the source image. "
                            "Ensure the model bundle includes Wan2.2_VAE.pth (or vae/*.safetensors) "
                            "and text_encoder assets under the bundle root."
                        )
                    raise RuntimeError(
                        "Image-to-video (animate) requires encoding the first RGB frame into "
                        "video latents. DanQing does not yet implement the Lightricks "
                        "`AutoencoderKLLTXVideo`-class encoder in MLX; first-frame conditioning "
                        "cannot be applied. Use text-to-video (`create`) or add an MLX encoder "
                        "port for your bundle VAE."
                    )

        latents, extra_cond = model.before_denoise(latents, timesteps, sigmas, **extra_cond)
        rope_kw = self._cogvideox_rotary_model_kwargs(family, config, h, w, latents)

        latents = self._denoise_video(
            latents=latents,
            timesteps=timesteps,
            scheduler=scheduler,
            model=model,
            txt_embeds=txt_embeds,
            neg_embeds=neg_embeds,
            guidance=guidance,
            config=config,
            sigmas=sigmas,
            timestep_embed_schedule=timestep_embed_schedule,
            extra_cond=extra_cond,
            rope_kw=rope_kw,
            cfg_renorm=_cfg_renorm,
            cfg_renorm_min=_cfg_renorm_min,
            ctx_exec=ctx_exec,
            on_progress=on_progress,
            on_log=on_log,
        )
        if latents is None:
            return None

        if ctx_exec.cancel_token.is_cancelled():
            return None

        if getattr(self.ctx, "backend", None) == "mlx":
            if on_log:
                on_log("info", "Materializing denoised latents for VAE decode...")
            self.ctx.eval(latents)
            if _video_post_denoise_clear_cache(family):
                self.ctx.clear_cache()

        n_steps = len(timesteps)
        if on_log:
            on_log("info", "Decoding video latents (VAE)...")
        _image_pipeline_emit_post_progress(on_progress, n_steps=n_steps, within_post=0.1)

        # 6. VAE decode
        def _vae_post_progress(frac: float) -> None:
            _image_pipeline_emit_post_progress(
                on_progress, n_steps=n_steps, within_post=0.1 + 0.75 * min(1.0, max(0.0, frac)),
            )

        def _vae_post_log(msg: str) -> None:
            if on_log:
                on_log("info", msg)

        frames = self._vae_decode_video(
            latents, entry, version_key or None,
            on_post_progress=_vae_post_progress,
            on_post_log=_vae_post_log,
        )

        if ctx_exec.cancel_token.is_cancelled():
            return None

        _image_pipeline_emit_post_progress(on_progress, n_steps=n_steps, within_post=0.85)
        if on_log:
            on_log("info", f"Saving video ({len(frames)} frames)...")

        # 7. Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work = Path(ctx_exec.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        out_path = str(work / f"{model_key}_{seed}_{timestamp}.mp4")
        self._save_video(frames, out_path, fps=fps)

        _image_pipeline_emit_complete(on_progress, n_steps)

        metadata = {
            "model": request.model, "seed": seed,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt or "",
            "steps": steps,
            "guidance": guidance, "num_frames": num_frames,
            "fps": fps, "width": w, "height": h,
            "mime_type": "video/mp4",
        }

        return str(out_path), metadata

    def _merge_video_transformer_bundle_config(
        self, family: str, bundle_root: Path | None, config: Any,
    ) -> None:
        if family == "cogvideox":
            self._merge_cogvideox_transformer_bundle_config_if_applicable(
                family, bundle_root, config,
            )
        elif family == "hunyuan":
            from backend.engine.config.model_configs import (
                HunyuanVideoConfig,
                merge_hunyuan_transformer_config_from_bundle,
            )
            if isinstance(config, HunyuanVideoConfig):
                merge_hunyuan_transformer_config_from_bundle(config, bundle_root)
        elif family == "wan":
            from backend.engine.config.model_configs import WanConfig, merge_wan_bundle_config

            if isinstance(config, WanConfig):
                merge_wan_bundle_config(config, bundle_root)

    def _resolve_video_encoder_type(self, config: Any, family: str) -> str:
        enc = getattr(config, "encoder_type", None)
        if enc:
            return str(enc)
        if family == "hunyuan":
            return "hunyuan_video_dual"
        return "t5"

    def _encode_video_text(
        self,
        text: str,
        config: Any,
        family: str,
        bundle_root: Path | None,
    ) -> tuple[Any, Any | None, Any | None, Any | None, Any | None, Any | None]:
        encoder_type = self._resolve_video_encoder_type(config, family)
        if encoder_type == "t5":
            return self._encode_t5(text), None, None, None, None, None
        if bundle_root is None:
            raise RuntimeError(
                f"Video model family {family!r} with encoder_type={encoder_type!r} "
                "requires a local bundle with text encoder assets."
            )
        return _encode_video_prompt_fn(
            self.ctx, text, encoder_type=encoder_type, bundle_root=bundle_root, config=config,
        )

    def _encode_video_text_with_cfg(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        config: Any,
        family: str,
        bundle_root: Path | None,
        guidance: float,
    ) -> tuple[
        Any, Any | None, Any | None, Any | None,
        Any | None, Any | None, Any | None, Any | None,
    ]:
        """Encode prompt + optional CFG negative; Hunyuan batches both in one Qwen/ByT5 forward."""
        empty_neg = (None, None, None, None)
        if not prompt or config.text_dim <= 0:
            return None, None, None, None, *empty_neg

        use_cfg = bool(getattr(config, "supports_guidance", True) and guidance > 1.0)
        encoder_type = self._resolve_video_encoder_type(config, family)
        if (
            use_cfg
            and encoder_type == "hunyuan_video_dual"
            and bundle_root is not None
        ):
            from backend.engine.families.hunyuan.text_encoder import get_hunyuan_text_encoder

            neg_txt = negative_prompt.strip() if negative_prompt else " "
            enc = get_hunyuan_text_encoder(self.ctx, bundle_root, config)
            e1, m1, e2, m2 = enc.encode([prompt, neg_txt])
            return (
                e1[0:1], m1[0:1], e2[0:1], m2[0:1],
                e1[1:2], m1[1:2], e2[1:2], m2[1:2],
            )

        if use_cfg and encoder_type == "t5":
            if family == "wan":
                from backend.engine.families.wan.conditioning import WAN_SAMPLE_NEG_PROMPT

                neg_txt = (negative_prompt or "").strip() or WAN_SAMPLE_NEG_PROMPT
            else:
                neg_txt = negative_prompt.strip() if negative_prompt else " "
            embeds = self._encode_t5_texts([prompt, neg_txt])
            return embeds[0:1], None, None, None, embeds[1:2], None, None, None

        pos = self._encode_video_text(prompt, config, family, bundle_root)
        if not use_cfg:
            return (*pos, *empty_neg)
        if family == "wan":
            from backend.engine.families.wan.conditioning import WAN_SAMPLE_NEG_PROMPT

            neg_txt = (negative_prompt or "").strip() or WAN_SAMPLE_NEG_PROMPT
        else:
            neg_txt = negative_prompt.strip() if negative_prompt else " "
        neg = self._encode_video_text(neg_txt, config, family, bundle_root)
        return (*pos, *neg)

    def _apply_video_text_to_extra_cond(
        self,
        extra_cond: dict[str, Any],
        *,
        txt_embeds: Any,
        txt_mask: Any | None,
        txt_embeds_2: Any | None,
        txt_mask_2: Any | None,
        neg_embeds: Any | None,
        neg_mask: Any | None,
        neg_embeds_2: Any | None,
        neg_mask_2: Any | None,
    ) -> dict[str, Any]:
        if txt_mask is not None:
            extra_cond["txt_attn_mask"] = txt_mask
        if txt_embeds_2 is not None:
            extra_cond["txt_embeds_2"] = txt_embeds_2
        if txt_mask_2 is not None:
            extra_cond["txt_attn_mask_2"] = txt_mask_2
        if neg_mask is not None:
            extra_cond["neg_txt_attn_mask"] = neg_mask
        if neg_embeds_2 is not None:
            extra_cond["neg_txt_embeds_2"] = neg_embeds_2
        if neg_mask_2 is not None:
            extra_cond["neg_txt_attn_mask_2"] = neg_mask_2
        return extra_cond

    def _merge_wan_config_from_bundle_if_applicable(
        self, family: str, bundle_root: Path | None, config: Any,
    ) -> None:
        if family != "wan" or bundle_root is None:
            return
        from backend.engine.config.model_configs import WanConfig, merge_wan_bundle_config

        if isinstance(config, WanConfig):
            merge_wan_bundle_config(config, bundle_root)

    def _merge_cogvideox_transformer_bundle_config_if_applicable(
        self, family: str, bundle_root: Path | None, config: Any,
    ) -> None:
        if family != "cogvideox" or bundle_root is None:
            return
        from backend.engine.config.model_configs import (
            CogVideoXConfig,
            merge_cogvideox_transformer_config_from_bundle,
        )
        if isinstance(config, CogVideoXConfig):
            merge_cogvideox_transformer_config_from_bundle(config, bundle_root)

    def _cogvideox_rotary_model_kwargs(
        self, family: str, config: Any, pixel_h: int, pixel_w: int, latents: Any,
    ) -> dict[str, Any]:
        if family != "cogvideox" or not getattr(config, "use_rotary_positional_embeddings", False):
            return {}
        from backend.engine.families.cogvideox.rotary_mlx import prepare_cogvideox_image_rotary_emb

        lt = int(latents.shape[2])
        vae_sf = int(getattr(config, "vae_scale", 8))
        cos_sin = prepare_cogvideox_image_rotary_emb(
            self.ctx, config, int(pixel_h), int(pixel_w), lt, vae_sf,
        )
        return {"image_rotary_emb": cos_sin}

    def _create_video_scheduler(
        self,
        *,
        family: str,
        scheduler_name: str,
        bundle_root: Path | None,
    ) -> Any:
        """Instantiate the denoise scheduler; family bundles may supply JSON defaults."""
        import json

        ctor_kwargs: dict[str, Any] = {}
        if family == "cogvideox" and scheduler_name == "cogvideox_dpm":
            if bundle_root is None:
                raise RuntimeError("CogVideoX requires a local model bundle for scheduler config.")
            from backend.engine.config.model_configs import cogvideox_scheduler_kwargs_from_bundle

            ctor_kwargs = cogvideox_scheduler_kwargs_from_bundle(bundle_root)
        elif family == "wan" and scheduler_name == "wan_flow_unipc":
            ctor_kwargs["num_train_timesteps"] = 1000
            if bundle_root is not None:
                sched_cfg = bundle_root / "scheduler" / "scheduler_config.json"
                if sched_cfg.is_file():
                    try:
                        data = json.loads(sched_cfg.read_text(encoding="utf-8"))
                        if "flow_shift" in data:
                            ctor_kwargs["shift"] = float(data["flow_shift"])
                    except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
                        raise RuntimeError(
                            f"Wan: cannot read scheduler config {sched_cfg}: {e}"
                        ) from e
        return get_scheduler(scheduler_name, ctx=self.ctx, **ctor_kwargs)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _latent_frame_count(self, config: Any, requested_pixel_frames: int) -> int:
        """Pixel timeline frames → latent timeline frames (CogVideoX-style temporal VAE compression).

        Models without ``temporal_vae_scale`` use the requested count as-is (Wan/LTX latents).
        """
        tvs = getattr(config, "temporal_vae_scale", None)
        if tvs is not None and int(tvs) > 0:
            rf = max(int(requested_pixel_frames), 1)
            return (rf - 1) // int(tvs) + 1
        return int(requested_pixel_frames)

    def _encode_t5_texts(self, texts: list[str]) -> Any:
        """Batch T5 encode — one tokenizer + one forward for multiple prompts."""
        entry_family = getattr(getattr(self, "_current_entry", None), "family", "")
        t5_root = getattr(self, "_t5_bundle_root", None)
        if t5_root is None:
            raise RuntimeError("T5 encoding requires _t5_bundle_root")
        bundle_root = t5_root if isinstance(t5_root, Path) else Path(t5_root)
        max_seq_len = int(getattr(self, "_t5_max_seq_len", 512))
        if entry_family == "wan":
            if self._t5 is None:
                from backend.engine.families.wan.text_encoder_mlx import WanUMT5EncoderMLX

                pth_path, tok_dir = self._wan_t5_encoder_bundle_paths(bundle_root)
                self._t5 = WanUMT5EncoderMLX(
                    self.ctx,
                    pth_path,
                    tok_dir,
                    text_len=max_seq_len,
                )
            return self._t5.encode(texts)
        t5_dir, t5_tok_dir = _t5_encoder_bundle_paths(bundle_root)
        if self._t5 is None:
            self._t5 = T5Encoder(
                self.ctx, t5_dir, max_seq_len=max_seq_len, tokenizer_path=t5_tok_dir,
            )
        return self._t5.encode(texts)

    def _release_video_t5_after_encode(self, family: str, encoder_type: str) -> None:
        """Drop T5 weights before loading DiT so ~10GB headroom is available."""
        if encoder_type != "t5" or self._t5 is None:
            return
        if family not in ("cogvideox", "wan"):
            return
        self._t5.release_weights()
        self._t5 = None

    def _encode_t5(self, text: str) -> Any:
        """T5 文本编码（视频模型目前统一使用 T5）；权重来自当前请求的 bundle，不走 Hub。"""
        return self._encode_t5_texts([text])

    def _model_cache_key(self, entry, version_key: str | None, num_frames: int) -> str:
        return f"video:{entry.id}:{version_key or 'default'}:{num_frames}"

    def _load_model(self, family: str, config, entry,
                    version_key: str | None, num_frames: int) -> Any:
        """加载视频模型 — 注册表驱动，零 family 分支。"""
        cache_key = self._model_cache_key(entry, version_key, num_frames)
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        trans_cls = _get_video_transformer_class(family)
        model = trans_cls(config, self.ctx, num_frames=num_frames)
        remap_fn = _get_video_weight_remap(family)

        bundle_root = self._local_bundle_root(entry, version_key)
        tensor_root, shard_paths = resolve_video_transformer_weight_sources(
            bundle_root, family, entry.id
        )
        if tensor_root is None or not shard_paths:
            return None

        w: dict[str, Any] = {}
        for sf in shard_paths:
            w.update(self.ctx.load_weights(str(sf)))

        if family == "ltx" and looks_like_mlx_forge_ltx_transformer_keys(w):
            w = restore_diffusers_names_from_mlx_forge_ltx(w)

        if remap_fn:
            w = remap_fn(w)
            if family == "ltx":
                mx_blk = max_remapped_ltx_block_index(w)
                if mx_blk >= 0:
                    n_blocks = mx_blk + 1
                    if n_blocks != config.depth:
                        raise RuntimeError(
                            f"LTX weights map to {n_blocks} transformer blocks after remap, "
                            f"but LTXConfig.depth={config.depth} (diffusers LTXVideoTransformer3DModel). "
                            f"Public MLX-forge / dgrauet LTX-2.3 bundles use 48 layers and are not supported "
                            f"by this transformer implementation; use ``:original`` or a 28-block "
                            f"diffusers-compatible checkpoint, or extend the LTX family implementation."
                        )

        from backend.engine.common.safetensors_affine_quant import read_bundle_affine_bits_if_quantized

        bundle_affine_bits = read_bundle_affine_bits_if_quantized(w, tensor_root)

        model.load_weights(
            list(w.items()),
            strict=False,
            ctx=self.ctx,
            bundle_affine_bits=bundle_affine_bits,
        )
        self.ctx.eval(*[p for _, p in model.parameters()])
        if self._cache is not None:
            from backend.engine.common.weights import parse_size_gb

            ver = self._resolve_version_block(entry, version_key)
            size_str = ""
            if ver:
                size_str = str(ver.get("size") or "")
            if not size_str:
                raw = getattr(entry, "raw", {}) or {}
                size_str = str(raw.get("size") or "10GB")
            self._cache.put(cache_key, model, parse_size_gb(size_str))
        return model

    # ------------------------------------------------------------------
    # VAE decode (frame-by-frame for video)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_cogvideox_vae_frame_batch(entry) -> int:
        raw = VideoPipeline._registry_scalar_default(entry, "vae_frame_batch_size", 2)
        try:
            n = int(raw if raw is not None else 2)
        except (TypeError, ValueError):
            n = 2
        return max(1, min(n, 8))

    @staticmethod
    def _resolve_hunyuan_vae_temporal_chunk(entry, latents) -> int:
        """Registry ``vae_temporal_chunk_size``; skip chunking when T is small enough."""
        chunk = int(VideoPipeline._registry_scalar_default(entry, "vae_temporal_chunk_size", 8) or 0)
        if chunk <= 0:
            return 0
        if getattr(latents, "ndim", None) == 5:
            t = int(latents.shape[2])
            if t <= chunk:
                return 0
        return chunk

    @staticmethod
    def _resolve_hunyuan_vae_spatial_tiling(entry) -> bool:
        return bool(VideoPipeline._registry_scalar_default(entry, "vae_spatial_tiling", False))

    @staticmethod
    def _resolve_wan_vae_spatial_tiling(entry) -> bool:
        return bool(VideoPipeline._registry_scalar_default(entry, "vae_spatial_tiling", False))

    def _vae_decode_video(
        self,
        latents,
        entry,
        version_key,
        on_post_progress: Callable[[float], None] | None = None,
        on_post_log: Callable[[str], None] | None = None,
    ) -> list:
        """逐帧 VAE 解码视频 latent → PIL Image 列表。

        latents: [B, C, T, H, W] 5D 视频 latent
        Returns: list of PIL Image (RGB frames)
        """
        family = getattr(entry, "family", "")
        if family == "cogvideox":
            from backend.engine.families.cogvideox.vae import decode_cogvideox_latents_to_pil_frames

            bundle_root = self._local_bundle_root(entry, version_key)
            frame_batch = self._resolve_cogvideox_vae_frame_batch(entry)
            return decode_cogvideox_latents_to_pil_frames(
                self.ctx, latents, bundle_root,
                on_stage=on_post_progress,
                on_log=on_post_log,
                frame_batch_size=frame_batch,
            )
        if family == "hunyuan":
            from backend.engine.families.hunyuan.vae import decode_hunyuan_latents_to_pil_frames

            bundle_root = self._local_bundle_root(entry, version_key)
            chunk = self._resolve_hunyuan_vae_temporal_chunk(entry, latents)
            spatial = self._resolve_hunyuan_vae_spatial_tiling(entry)
            return decode_hunyuan_latents_to_pil_frames(
                self.ctx, latents, bundle_root,
                on_stage=on_post_progress,
                on_log=on_post_log,
                temporal_chunk_size=chunk,
                spatial_tiling=spatial,
            )
        if family == "wan":
            from backend.engine.families.wan.vae import decode_wan_latents_to_pil_frames

            bundle_root = self._local_bundle_root(entry, version_key)
            spatial = self._resolve_wan_vae_spatial_tiling(entry)
            spatial_scale = int(self._registry_scalar_default(entry, "vae_scale", 16) or 16)
            return decode_wan_latents_to_pil_frames(
                self.ctx,
                latents,
                bundle_root,
                on_stage=on_post_progress,
                on_log=on_post_log,
                spatial_tiling=spatial,
                spatial_scale=spatial_scale,
            )

        from PIL import Image
        from backend.engine.common.vae import VAEDecoder, remap_vae_weights, vae_output_to_uint8_hwc

        ctx = self.ctx
        bundle_root = self._local_bundle_root(entry, version_key)
        vae_dir = (bundle_root / "vae") if bundle_root else None

        scaling_factor = 1.0
        shift_factor = 0.0
        latent_cfg = 16
        vae_cfg: dict[str, Any] = {}
        if vae_dir and (vae_dir / "config.json").exists():
            import json
            with open(vae_dir / "config.json") as f:
                vae_cfg = json.load(f)
            scaling_factor = float(vae_cfg.get("scaling_factor", 1.0))
            shift_factor = float(vae_cfg.get("shift_factor", 0.0))
            latent_cfg = int(vae_cfg.get("latent_channels", 16))

        use_quant_path = bool(vae_cfg.get("use_quant_conv", False))
        use_post_quant_path = bool(vae_cfg.get("use_post_quant_conv", False))

        vae_weights = {}
        if vae_dir and vae_dir.exists():
            for sf in sorted(vae_dir.glob("*.safetensors")):
                vae_weights.update(ctx.load_weights(str(sf)))
        elif family == "ltx" and bundle_root is not None:
            dec_path = ltx_flat_vae_decoder_file(bundle_root)
            if dec_path is not None:
                raw_dec = ctx.load_weights(str(dec_path))
                vae_weights = {f"decoder.{k}": v for k, v in raw_dec.items()}

        # Extract frame dim
        if latents.ndim == 5:
            B, C, T, H, W = latents.shape
        else:
            B, C, H, W = latents.shape
            T = 1

        frames = []
        for t_idx in range(T):
            frame_latent = latents[:, :, t_idx, :, :]  # [B, C, H, W]

            # Flux2-style latent path — only when config enables quant/post_quant (see ImagePipeline._vae_decode).
            if (use_quant_path or use_post_quant_path) and (
                "bn.running_mean" in vae_weights or "post_quant_conv.weight" in vae_weights
            ):
                frame_latent = self._vae_preprocess_special(frame_latent, vae_weights, scaling_factor, shift_factor)
                sf = 1.0
                shf = 0.0
                ci = vae_weights.get("decoder.conv_in.weight", ctx.zeros((1,))).shape[0] if "decoder.conv_in.weight" in vae_weights else 16
            else:
                sf = scaling_factor
                shf = shift_factor
                ci = latent_cfg

            C_ = frame_latent.shape[1] if frame_latent.ndim >= 4 else ci
            vae = VAEDecoder(latent_channels=C_, ctx=ctx, scaling_factor=sf, shift_factor=shf)
            if vae_weights:
                decoder_w = remap_vae_weights(vae_weights)
                vae.load_weights(list(decoder_w.items()), strict=False)

            image = vae.forward(frame_latent)
            pixels = vae_output_to_uint8_hwc(image, ctx)
            frames.append(Image.fromarray(pixels))

        return frames

    def _vae_preprocess_special(self, latents, vae_weights, scaling_factor, shift_factor):
        """特殊 VAE 预处理 — flux2 风格（通过权重检测触发，非 family 硬编码）。"""
        ctx = self.ctx

        bn_mean = vae_weights.get("bn.running_mean", ctx.zeros((128,))).reshape(1, -1, 1, 1)
        bn_var = vae_weights.get("bn.running_var", ctx.ones((128,))).reshape(1, -1, 1, 1)
        latents = latents * ctx.sqrt(bn_var + 1e-4) + bn_mean

        B, C_, H_, W_ = latents.shape
        latents = latents.reshape(B, C_ // 4, 2, 2, H_, W_)
        latents = ctx.permute(latents, (0, 1, 4, 2, 5, 3))
        latents = latents.reshape(B, C_ // 4, H_ * 2, W_ * 2)

        latents = (latents / scaling_factor) + shift_factor
        latents = ctx.permute(latents, (0, 2, 3, 1))

        pw = vae_weights.get("post_quant_conv.weight")
        pb = vae_weights.get("post_quant_conv.bias")
        if pw is not None and pb is not None:
            latents = ctx.conv2d(latents, ctx.permute(pw, (0, 2, 3, 1)), stride=1, padding=0)
            latents = latents + pb.reshape(1, 1, 1, -1)

        latents = ctx.permute(latents, (0, 3, 1, 2))
        return latents

    def _vae_encode_frame(self, image_tensor, entry, version_key) -> Any:
        """VAE 编码单帧图像 → latent（用于 I2V 首帧条件）。"""
        family = getattr(entry, "family", "")
        bundle_root = self._local_bundle_root(entry, version_key)
        if family == "hunyuan" and bundle_root is not None:
            from backend.engine.families.hunyuan.vae import encode_hunyuan_rgb_to_latents

            ctx = self.ctx
            if image_tensor.ndim == 4:
                image_tensor = ctx.unsqueeze(image_tensor, axis=2)
            return encode_hunyuan_rgb_to_latents(ctx, image_tensor, bundle_root)
        if family == "wan" and bundle_root is not None:
            from backend.engine.families.wan.vae import encode_wan_image_to_latent

            ctx = self.ctx
            import numpy as np
            if image_tensor.ndim == 4:
                chw = image_tensor[0]
            else:
                chw = image_tensor
            if int(chw.shape[0]) == 3:
                pass
            else:
                chw = ctx.permute(chw, (2, 0, 1))
            latent = encode_wan_image_to_latent(ctx, chw, bundle_root)
            return ctx.expand_dims(latent, 0)
        del image_tensor, entry, version_key
        return None

    # ------------------------------------------------------------------
    # 视频保存
    # ------------------------------------------------------------------

    def _save_video(self, frames: list, output_path: str, fps: int = 16):
        """将 PIL Image 帧列表保存为 MP4 视频。

        优先使用 ffmpeg；降级到 imageio。
        """
        if not frames:
            raise RuntimeError("No frames to save")

        # Try ffmpeg first (most reliable for MP4)
        try:
            self._save_video_ffmpeg(frames, output_path, fps)
            return
        except (FileNotFoundError, subprocess.SubprocessError):
            pass

        # Fallback: imageio
        self._save_video_imageio(frames, output_path, fps)

    def _save_video_ffmpeg(self, frames: list, output_path: str, fps: int):
        """使用 ffmpeg 子进程保存 MP4。"""
        import numpy as np

        w, h = frames[0].size
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",
            output_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        for frame in frames:
            arr = np.array(frame.convert("RGB"))
            proc.stdin.write(arr.tobytes())
        proc.stdin.close()
        proc.wait(timeout=120)

    def _save_video_imageio(self, frames: list, output_path: str, fps: int):
        """使用 imageio 保存 MP4（降级方案）。"""
        import numpy as np
        import imageio
        writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
        for frame in frames:
            writer.append_data(np.array(frame.convert("RGB")))
        writer.close()
