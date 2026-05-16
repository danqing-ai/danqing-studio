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
    get_video_transformer_class as _get_video_transformer_class,
    get_video_weight_remap as _get_video_weight_remap,
)
from backend.engine.config.model_configs import get_config_class


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
from backend.engine.pipelines.image_pipeline import _t5_encoder_bundle_paths
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
        try:
            _t5_encoder_bundle_paths(bundle_root)
            return bundle_root
        except RuntimeError as err:
            alt = self._resolved_original_video_bundle_root(entry)
            if alt is not None:
                _t5_encoder_bundle_paths(alt)
                return alt
            raise RuntimeError(
                f"T5 text encoder assets not found under {bundle_root}, "
                f"and no installed ``original`` registry version for ``{entry.id}``. "
                f"MLX-forge / dgrauet transformer bundles omit T5: install the ``original`` "
                f"diffusers snapshot or place ``text_encoder`` + ``tokenizer`` under the bundle root."
            ) from err

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
        num_frames = request.num_frames or 81
        fps = request.fps or 16
        seed = request.seed if request.seed is not None else random.randint(0, 2 ** 32 - 1)
        entry = self._registry.require(model_key)
        config_cls = get_config_class(entry.family)
        config = config_cls()
        family = getattr(entry, "family", "ltx")

        # ── Registry-driven parameter injection ──
        for param_key in ("vae_scale", "default_scheduler"):
            val = self._registry_scalar_default(entry, param_key, None)
            if val is not None:
                setattr(config, param_key, val)

        sg = self._registry_scalar_default(entry, "supports_guidance", None)
        if sg is not None:
            config.supports_guidance = bool(sg)

        if ctx_exec.cancel_token.is_cancelled():
            return None

        bundle_root = self._local_bundle_root(entry, version_key or None)
        self._merge_cogvideox_transformer_bundle_config_if_applicable(family, bundle_root, config)
        self._t5_bundle_root = self._effective_t5_bundle_root(entry, bundle_root)
        self._t5 = None

        steps_default = self._registry_scalar_default(entry, "steps", 40)
        guidance_default = self._registry_scalar_default(entry, "guidance", 0.0)
        scheduler_registry = self._registry_scalar_default(entry, "scheduler", None)
        scheduler_default = scheduler_registry or getattr(config, "default_scheduler", "unipc")

        steps = int(request.steps) if request.steps is not None else int(steps_default)
        steps = max(1, steps)
        guidance = float(request.guidance) if request.guidance is not None else float(guidance_default)
        if not getattr(config, "supports_guidance", True):
            guidance = 0.0

        # 1. Text encoding
        txt_embeds = None
        neg_embeds = None
        if request.prompt and config.text_dim > 0:
            txt_embeds = self._encode_t5(request.prompt)
            if config.supports_guidance and guidance > 1.0:
                neg_txt = request.negative_prompt.strip() if request.negative_prompt else " "
                neg_embeds = self._encode_t5(neg_txt)

        if ctx_exec.cancel_token.is_cancelled():
            return None

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

        # 3. Scheduler
        scheduler = get_scheduler(scheduler_default, ctx=self.ctx)
        vae_scale = getattr(config, "vae_scale", 8)
        timesteps = scheduler.set_timesteps(steps)
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
            if _cfg_renorm:
                parts.append(f"cfg_renorm=True cfg_renorm_min={_cfg_renorm_min}")
            on_log("info", " ".join(parts))

        # 4. Initial noise [B, C, T_latent, H_lat, W_lat]
        latent_shape = (1, config.dim_in, latent_frames, h // vae_scale, w // vae_scale)
        if seed is not None:
            latents = self.ctx.seeded_randn(latent_shape, seed, dtype=self.ctx.float32())
        else:
            latents = self.ctx.randn(latent_shape, dtype=self.ctx.float32())

        # ── Hook ③: before denoise ──
        latents, extra_cond = model.before_denoise(latents, timesteps, sigmas, **extra_cond)
        _rope_kw = self._cogvideox_rotary_model_kwargs(family, config, h, w, latents)

        # ------------------------------------------------------------------
        # 5. Denoising loop (inline CFG, matching ImagePipeline pattern)
        # ------------------------------------------------------------------
        for i, t in enumerate(timesteps):
            if ctx_exec.cancel_token.is_cancelled():
                return None

            model_kwargs = {"txt_embeds": txt_embeds} if txt_embeds is not None else {}
            model_kwargs.update(_rope_kw)
            model_kwargs.update(extra_cond)
            if sigmas is not None:
                model_kwargs["sigmas"] = sigmas
            if timestep_embed_schedule is not None and i < len(timestep_embed_schedule):
                model_kwargs["timestep_embed_value"] = timestep_embed_schedule[i]

            noise_cond = model(latents, t, **model_kwargs)

            if neg_embeds is not None and getattr(config, "supports_guidance", False):
                neg_kwargs = {"txt_embeds": neg_embeds}
                neg_kwargs.update(_rope_kw)
                neg_kwargs.update(extra_cond)
                if sigmas is not None:
                    neg_kwargs["sigmas"] = sigmas
                if timestep_embed_schedule is not None and i < len(timestep_embed_schedule):
                    neg_kwargs["timestep_embed_value"] = timestep_embed_schedule[i]
                noise_uncond = model(latents, t, **neg_kwargs)
                noise_pred = noise_uncond + guidance * (noise_cond - noise_uncond)
                if _cfg_renorm and getattr(config, "supports_guidance", False):
                    noise_pred = model.refine_cfg_noise(
                        noise_cond, noise_pred, cfg_renorm_min=_cfg_renorm_min,
                    )
            else:
                noise_pred = noise_cond

            latents = scheduler.step(noise_pred, t, latents)

            # ── Hook ④: per-step callback ──
            model.step_callback(i, latents, noise_pred)

            if on_progress:
                on_progress((i + 1) / len(timesteps), i + 1, len(timesteps), None)
            if on_log:
                on_log("info", f"Step {i+1}/{len(timesteps)}")

        if ctx_exec.cancel_token.is_cancelled():
            return None

        # 6. VAE decode (frame-by-frame for video latents)
        frames = self._vae_decode_video(latents, entry, version_key or None)

        if ctx_exec.cancel_token.is_cancelled():
            return None

        # 7. Save (task work dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work = Path(ctx_exec.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        out_path = str(work / f"{model_key}_{seed}_{timestamp}.mp4")
        self._save_video(frames, out_path, fps=fps)

        metadata = {
            "model": request.model, "seed": seed,
            "prompt": request.prompt, "steps": steps,
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
        num_frames = request.num_frames or 81
        fps = request.fps or 16
        seed = request.seed if request.seed is not None else random.randint(0, 2 ** 32 - 1)
        entry = self._registry.require(model_key)
        config_cls = get_config_class(entry.family)
        config = config_cls()
        family = getattr(entry, "family", "ltx")

        # Registry-driven parameter injection
        for param_key in ("vae_scale", "default_scheduler"):
            val = self._registry_scalar_default(entry, param_key, None)
            if val is not None:
                setattr(config, param_key, val)
        sg = self._registry_scalar_default(entry, "supports_guidance", None)
        if sg is not None:
            config.supports_guidance = bool(sg)

        if ctx_exec.cancel_token.is_cancelled():
            return None

        bundle_root = self._local_bundle_root(entry, version_key or None)
        self._merge_cogvideox_transformer_bundle_config_if_applicable(family, bundle_root, config)
        self._t5_bundle_root = self._effective_t5_bundle_root(entry, bundle_root)
        self._t5 = None

        steps_default = self._registry_scalar_default(entry, "steps", 40)
        guidance_default = self._registry_scalar_default(entry, "guidance", 0.0)
        scheduler_registry = self._registry_scalar_default(entry, "scheduler", None)
        scheduler_default = scheduler_registry or getattr(config, "default_scheduler", "unipc")

        steps = int(request.steps) if request.steps is not None else int(steps_default)
        steps = max(1, steps)
        guidance = float(request.guidance) if request.guidance is not None else float(guidance_default)
        if not getattr(config, "supports_guidance", True):
            guidance = 0.0

        # 1. Text encoding
        txt_embeds = None
        neg_embeds = None
        if request.prompt and config.text_dim > 0:
            txt_embeds = self._encode_t5(request.prompt)
            if config.supports_guidance and guidance > 1.0:
                neg_txt = request.negative_prompt.strip() if request.negative_prompt else " "
                neg_embeds = self._encode_t5(neg_txt)

        if ctx_exec.cancel_token.is_cancelled():
            return None

        # 2. Load model
        latent_frames = self._latent_frame_count(config, num_frames)
        model = self._load_model(family, config, entry, version_key or None, latent_frames)
        if model is None:
            raise RuntimeError(f"Failed to load model: {model_key}")

        model.after_load_weights(bundle_root=str(bundle_root) if bundle_root else None)

        extra_cond = model.prepare_conditioning(request,
                                                bundle_root=str(bundle_root) if bundle_root else None)

        # 3. Scheduler
        scheduler = get_scheduler(scheduler_default, ctx=self.ctx)
        vae_scale = getattr(config, "vae_scale", 8)
        timesteps = scheduler.set_timesteps(steps)
        sigmas = getattr(scheduler, 'sigmas', None)

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
            on_log("info", " ".join(parts))

        # 4. Initial noise
        latent_shape = (1, config.dim_in, latent_frames, h // vae_scale, w // vae_scale)
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
                src_img = src_img.resize((w, h))
                import numpy as np
                src_array = np.array(src_img).astype(np.float32) / 127.5 - 1.0
                src_tensor = self.ctx.array(np.expand_dims(src_array, 0))
                # VAE encode first frame
                vae_latent = self._vae_encode_frame(src_tensor, entry, version_key or None)
                if vae_latent is not None:
                    # Replace first frame latent with encoded source
                    latents = self.ctx.concat(
                        [vae_latent[:, :, :1, :, :], latents[:, :, 1:, :, :]], axis=2
                    )
                else:
                    raise RuntimeError(
                        "Image-to-video (animate) requires encoding the first RGB frame into "
                        "video latents. DanQing does not yet implement the Lightricks "
                        "`AutoencoderKLLTXVideo`-class encoder in MLX; first-frame conditioning "
                        "cannot be applied. Use text-to-video (`create`) or add an MLX encoder "
                        "port for your bundle VAE."
                    )

        latents, extra_cond = model.before_denoise(latents, timesteps, sigmas, **extra_cond)
        _rope_kw = self._cogvideox_rotary_model_kwargs(family, config, h, w, latents)

        # 5. Denoising loop
        for i, t in enumerate(timesteps):
            if ctx_exec.cancel_token.is_cancelled():
                return None

            model_kwargs = {"txt_embeds": txt_embeds} if txt_embeds is not None else {}
            model_kwargs.update(_rope_kw)
            model_kwargs.update(extra_cond)
            if sigmas is not None:
                model_kwargs["sigmas"] = sigmas
            if timestep_embed_schedule is not None and i < len(timestep_embed_schedule):
                model_kwargs["timestep_embed_value"] = timestep_embed_schedule[i]

            noise_cond = model(latents, t, **model_kwargs)

            if neg_embeds is not None and getattr(config, "supports_guidance", False):
                neg_kwargs = {"txt_embeds": neg_embeds}
                neg_kwargs.update(_rope_kw)
                neg_kwargs.update(extra_cond)
                if sigmas is not None:
                    neg_kwargs["sigmas"] = sigmas
                if timestep_embed_schedule is not None and i < len(timestep_embed_schedule):
                    neg_kwargs["timestep_embed_value"] = timestep_embed_schedule[i]
                noise_uncond = model(latents, t, **neg_kwargs)
                noise_pred = noise_uncond + guidance * (noise_cond - noise_uncond)
                if _cfg_renorm and getattr(config, "supports_guidance", False):
                    noise_pred = model.refine_cfg_noise(
                        noise_cond, noise_pred, cfg_renorm_min=_cfg_renorm_min,
                    )
            else:
                noise_pred = noise_cond

            latents = scheduler.step(noise_pred, t, latents)

            model.step_callback(i, latents, noise_pred)

            if on_progress:
                on_progress((i + 1) / len(timesteps), i + 1, len(timesteps), None)
            if on_log:
                on_log("info", f"Step {i+1}/{len(timesteps)}")

        if ctx_exec.cancel_token.is_cancelled():
            return None

        # 6. VAE decode
        frames = self._vae_decode_video(latents, entry, version_key or None)

        if ctx_exec.cancel_token.is_cancelled():
            return None

        # 7. Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work = Path(ctx_exec.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        out_path = str(work / f"{model_key}_{seed}_{timestamp}.mp4")
        self._save_video(frames, out_path, fps=fps)

        metadata = {
            "model": request.model, "seed": seed,
            "prompt": request.prompt, "steps": steps,
            "guidance": guidance, "num_frames": num_frames,
            "fps": fps, "width": w, "height": h,
            "mime_type": "video/mp4",
        }

        return str(out_path), metadata

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

    def _encode_t5(self, text: str) -> Any:
        """T5 文本编码（视频模型目前统一使用 T5）；权重来自当前请求的 bundle，不走 Hub。"""
        t5_dir, t5_tok_dir = _t5_encoder_bundle_paths(getattr(self, "_t5_bundle_root", None))
        if self._t5 is None:
            self._t5 = T5Encoder(self.ctx, t5_dir, tokenizer_path=t5_tok_dir)
        return self._t5.encode([text])

    def _load_model(self, family: str, config, entry,
                    version_key: str | None, num_frames: int) -> Any:
        """加载视频模型 — 注册表驱动，零 family 分支。"""
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
        return model

    # ------------------------------------------------------------------
    # VAE decode (frame-by-frame for video)
    # ------------------------------------------------------------------

    def _vae_decode_video(self, latents, entry, version_key) -> list:
        """逐帧 VAE 解码视频 latent → PIL Image 列表。

        latents: [B, C, T, H, W] 5D 视频 latent
        Returns: list of PIL Image (RGB frames)
        """
        family = getattr(entry, "family", "")
        if family == "cogvideox":
            from backend.engine.families.cogvideox.vae import decode_cogvideox_latents_to_pil_frames

            bundle_root = self._local_bundle_root(entry, version_key)
            return decode_cogvideox_latents_to_pil_frames(self.ctx, latents, bundle_root)

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
        """VAE 编码单帧图像 → latent（用于 I2V 首帧条件）。

        Lightricks LTX / 多数视频 VAE 使用 ``AutoencoderKLLTXVideo`` 等 **非 SD-AE 结构**，
        当前仓库未提供与之等价的 MLX 编码器；调用方在 ``run_edit`` 中若传入 ``source_asset_id``
        且此处返回 ``None``，将 **显式失败**（见 ``run_edit``）。
        """
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
