"""VideoUpscalePipeline — HunyuanVideo-1.5 1080p SR（与 ``VideoPipeline`` 平级）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from backend.core.contracts import ExecutionContext, VideoUpscaleRequest, parse_model_version
from backend.engine.common.cache import ModelCache
from backend.engine.config.model_configs import get_config_class
from backend.engine.families.hunyuan.sr import run_hunyuan_video_sr
from backend.engine.families.hunyuan.text_encoder import get_hunyuan_text_encoder
from backend.engine.pipelines.video_pipeline import VideoPipeline
from backend.engine.runtime._base import RuntimeContext


class VideoUpscalePipeline:
    """注册表驱动的视频超分 — 当前支持 HunyuanVideo-1.5 SR bundle。"""

    def __init__(
        self,
        ctx: RuntimeContext,
        model_registry: Any,
        asset_store: Any,
        model_cache: ModelCache | None = None,
        project_root: Path | None = None,
    ) -> None:
        self.ctx = ctx
        self._registry = model_registry
        self._asset_store = asset_store
        self._cache = model_cache
        self._project_root = project_root or Path.cwd()

    def _local_bundle_root(self, entry, version_key: str | None) -> Path | None:
        vp = VideoPipeline(
            self.ctx, self._registry, self._asset_store,
            model_cache=self._cache, project_root=self._project_root,
        )
        return vp._local_bundle_root(entry, version_key)

    def run(
        self,
        request: VideoUpscaleRequest,
        ctx_exec: ExecutionContext,
        *,
        on_progress: Callable | None = None,
        on_log: Callable | None = None,
    ):
        if ctx_exec.cancel_token.is_cancelled():
            return None

        model_key, version_key = parse_model_version(request.model)
        entry = self._registry.require(model_key)
        if getattr(entry, "media", None) != "video":
            raise RuntimeError(
                f"Video upscale model {model_key!r} is not a video model "
                f"(media={getattr(entry, 'media', None)!r})."
            )

        family = getattr(entry, "family", "")
        if family != "hunyuan":
            raise RuntimeError(
                f"Video upscale is not implemented for family {family!r}; "
                "only hunyuan SR bundles are supported."
            )

        bundle_root = self._local_bundle_root(entry, version_key)
        if bundle_root is None:
            raise RuntimeError(f"HunyuanVideo SR bundle not installed for {model_key!r}")

        config_cls = get_config_class(family)
        config = config_cls()
        object.__setattr__(config, "use_meanflow", True)
        vst = VideoPipeline._registry_scalar_default(entry, "vae_spatial_tiling", None)
        if vst is not None:
            config.vae_spatial_tiling = bool(vst)
        VideoPipeline.apply_hunyuan_text_encoder_paths(entry, config, self._project_root)

        src_id = request.source_asset_id
        if not src_id:
            raise RuntimeError("Video upscale requires source_asset_id (low-res video latents or asset).")

        src_path = self._asset_store.get_file_path(src_id)
        if src_path is None or not src_path.exists():
            raise RuntimeError(f"Source asset not found: {src_id!r}")

        prompt = request.prompt or ""
        enc = get_hunyuan_text_encoder(self.ctx, bundle_root, config)
        txt_embeds, txt_mask, txt_embeds_2, txt_mask_2 = enc.encode([prompt])
        if getattr(self.ctx, "backend", None) == "mlx":
            self.ctx.clear_cache()

        import numpy as np
        from PIL import Image

        if src_path.suffix.lower() in (".mp4", ".webm", ".mov"):
            raise RuntimeError(
                "HunyuanVideo SR from file video is not yet wired; pass latents via future asset metadata."
            )

        img = Image.open(src_path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        pixels = self.ctx.array(arr[np.newaxis, np.newaxis, ...])

        from backend.engine.families.hunyuan.vae import encode_hunyuan_rgb_to_latents

        vae_root = Path(getattr(request, "vae_bundle", "") or str(bundle_root))
        low_latents = encode_hunyuan_rgb_to_latents(self.ctx, pixels, vae_root)

        sr_steps = int(getattr(request, "steps", None) or 6)
        chunk = VideoPipeline._resolve_hunyuan_vae_temporal_chunk(entry, low_latents)
        spatial = VideoPipeline._resolve_hunyuan_vae_spatial_tiling(entry)
        from backend.engine.common.pipeline_registry import resolve_version_block
        from backend.engine.common.weights import parse_size_gb

        ver = resolve_version_block(entry, version_key)
        raw = getattr(entry, "raw", {}) or {}
        size_str = str((ver or {}).get("size") or raw.get("size") or "10GB")
        sr_cache_key = f"upscale:video:{entry.id}:{version_key or 'default'}"
        _, frames = run_hunyuan_video_sr(
            self.ctx,
            config,
            bundle_root,
            low_latents,
            txt_embeds=txt_embeds,
            txt_attn_mask=txt_mask,
            txt_embeds_2=txt_embeds_2,
            txt_attn_mask_2=txt_mask_2,
            vae_bundle_root=vae_root,
            steps=sr_steps,
            temporal_chunk_size=chunk,
            spatial_tiling=spatial or getattr(config, "vae_spatial_tiling", True),
            model_cache=self._cache,
            cache_key=sr_cache_key,
            cache_size_gb=parse_size_gb(size_str),
        )

        work = Path(ctx_exec.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        out_path = str(work / f"{model_key}_sr_{src_id}.mp4")
        vp = VideoPipeline(
            self.ctx, self._registry, self._asset_store,
            project_root=self._project_root,
        )
        fps = int(getattr(request, "fps", None) or 24)
        vp._save_video(frames, out_path, fps=fps)

        if on_log:
            on_log("info", f"HunyuanVideo SR saved {out_path}")
        if on_progress:
            on_progress({"phase": "complete", "progress": 1.0})

        return out_path, {
            "model": request.model,
            "source_asset_id": src_id,
            "mime_type": "video/mp4",
            "sr_steps": sr_steps,
        }
