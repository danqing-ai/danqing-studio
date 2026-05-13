"""ImageUpscalePipeline — 图像超分装配线（与 ``ImagePipeline`` 平级）。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from backend.core.contracts import ExecutionContext, ImageUpscaleRequest, parse_model_version
from backend.engine.common.cache import ModelCache
from backend.engine.common.pipeline_registry import (
    local_bundle_root as _local_bundle_root_fn,
    resolve_project_path as _resolve_project_path_fn,
    resolve_version_block as _resolve_version_block_fn,
)
from backend.engine.runtime._base import RuntimeContext


class ImageUpscalePipeline:
    """注册表驱动的图像超分；当前 MLX 路径经 SeedVR2 ``job_mlx.run_seedvr2_upscale``。"""

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

    def _resolve_path(self, local_path: str) -> Path:
        return _resolve_project_path_fn(self._project_root, local_path)

    def _resolve_version_block(self, entry, version_key: str | None) -> dict | None:
        return _resolve_version_block_fn(entry, version_key)

    def _local_bundle_root(self, entry, version_key: str | None) -> Path | None:
        return _local_bundle_root_fn(self._project_root, entry, version_key)

    def run(
        self,
        request: ImageUpscaleRequest,
        ctx_exec: ExecutionContext,
        *,
        on_progress: Callable | None = None,
        on_log: Callable | None = None,
    ):
        """返回 ``(output_png_path, metadata_dict)``；取消时返回 ``None``。"""
        model_key, version_key = parse_model_version(request.model)
        entry = self._registry.require(model_key)
        family = getattr(entry, "family", "")
        if family != "seedvr2":
            raise RuntimeError(
                f"Image upscale on MLX is only implemented for family 'seedvr2'; "
                f"model {model_key!r} has family={family!r}."
            )

        if ctx_exec.cancel_token.is_cancelled():
            return None

        bundle_root = self._local_bundle_root(entry, version_key or None)
        if bundle_root is None:
            raise RuntimeError(
                f"Model {model_key!r} has no installed bundle (version={version_key or 'default'}); "
                "cannot run SeedVR2 upscale."
            )

        from PIL import Image

        src_path = ctx_exec.asset_store.get_file_path(request.source_asset_id)
        if not src_path.is_file():
            raise RuntimeError(f"Source asset file missing: {src_path}")

        scale = int(request.scale)
        seed = (request.metadata or {}).get("seed")
        if seed is not None:
            seed = int(seed)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work = Path(ctx_exec.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        out_path = work / f"{model_key}_up_{timestamp}.png"

        def _log(level: str, msg: str) -> None:
            if on_log:
                on_log(level, msg)

        from backend.engine.families.seedvr2.job_mlx import run_seedvr2_upscale

        extra = run_seedvr2_upscale(
            bundle_path=bundle_root,
            model_key=model_key,
            source_image=src_path,
            scale=scale,
            softness=float(request.denoise),
            seed=seed,
            output_png=out_path,
            on_log=_log,
        )

        if ctx_exec.cancel_token.is_cancelled():
            return None

        with Image.open(out_path) as pil:
            w, h = pil.size
        if on_progress:
            on_progress(1.0, 1, 1, None)

        meta = {
            "model": request.model,
            "width": w,
            "height": h,
            "mime_type": "image/png",
            "scale": scale,
            "denoise": float(request.denoise),
        }
        meta.update(extra)
        return str(out_path), meta
