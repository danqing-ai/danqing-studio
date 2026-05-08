"""
VideoPipeline — 视频请求 → 模型推理 → 资产落盘。

完全后端无关。
"""
from __future__ import annotations

import asyncio
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.core.contracts import (
    EngineResult, ExecutionContext, VideoGenerationRequest,
    VideoEditRequest, VideoUpscaleRequest,
    LogEvent, ProgressEvent, parse_model_version, parse_size,
)
from .common.cache import ModelCache
from .common.pipeline import DenoisingPipeline, GenerationCancelled
from .common.schedulers import get_scheduler
from .common.text_encoders import T5Encoder
from .common.weights import parse_size_gb
from .config.model_configs import get_config_class
from .runtime._base import RuntimeContext


class VideoPipeline:
    """视频生成管线 — 后端无关。"""

    def __init__(self, ctx: RuntimeContext, model_registry: Any,
                 asset_store: Any, model_cache: ModelCache | None = None,
                 text_encoders_path: str = ""):
        self.ctx = ctx
        self._registry = model_registry
        self._asset_store = asset_store
        self._cache = model_cache
        self._encoders_path = text_encoders_path
        self._t5: T5Encoder | None = None

    async def generate(self, request: VideoGenerationRequest,
                       ctx_exec: ExecutionContext) -> EngineResult:
        """文生视频 / 图生视频。"""
        model_key, version = parse_model_version(request.model)
        w, h = parse_size(request.size)
        num_frames = request.num_frames or 81
        fps = request.fps or 16
        seed = request.seed if request.seed is not None else random.randint(0, 2 ** 32 - 1)
        entry = self._registry.require(model_key)
        config_cls = get_config_class(entry.family)
        config = config_cls()

        # 1. 文本编码
        txt_embeds = None
        if request.prompt and config.text_dim > 0:
            txt_embeds = await self._encode_t5(request.prompt)

        # 2. 初始噪声
        latents = self.ctx.randn(
            (1, config.dim_in, num_frames, h // 8, w // 8),
            dtype=self.ctx.float32(),
        )

        # 3. 模型加载
        model = self._load_model(model_key, config, entry, num_frames)

        # 4. 调度器
        scheduler = get_scheduler(
            getattr(config, "default_scheduler", "linear"),
            ctx=self.ctx,
        )
        steps = request.steps or 40
        timesteps = scheduler.set_timesteps(steps)

        # 5. 去噪循环（与调度器共用 ExecutionContext.cancel_token，DELETE 取消方可生效）
        pipeline = DenoisingPipeline()
        cancel_token = ctx_exec.cancel_token

        def on_step(step, total, lat):
            ctx_exec.on_progress(ProgressEvent(
                progress=step / total,
                step=step, total=total,
            ))
            ctx_exec.on_log(LogEvent(level="info", message=f"Step {step}/{total}"))

        try:
            latents_out = pipeline.run(
                model=model,
                scheduler=scheduler,
                latents=latents,
                timesteps=timesteps,
                context={"txt_embeds": txt_embeds} if txt_embeds is not None else None,
                guidance_scale=float(request.guidance or 0.0),
                on_step=on_step,
                cancel_token=cancel_token,
            )
        except GenerationCancelled:
            return EngineResult(primary_asset_id="", metadata={"status": "cancelled"})
        finally:
            self._release_model(model)

        if ctx_exec.cancel_token.is_cancelled():
            return EngineResult(primary_asset_id="", metadata={"status": "cancelled"})

        # 6. 解码 + 落盘
        output_path = self._save_output(latents_out, model_key, seed, fps, ctx_exec)
        aid = self._asset_store.create_from_file(
            Path(output_path), kind="video", mime_type="video/mp4",
            source_task_id=ctx_exec.task_id,
            metadata={
                "model": request.model, "seed": seed,
                "prompt": request.prompt, "steps": steps,
                "num_frames": num_frames, "fps": fps,
                "width": w, "height": h,
                "mime_type": "video/mp4",
            },
            source_action="create",
        )
        return EngineResult(primary_asset_id=aid, asset_ids=[aid], output_paths=[output_path])

    async def edit(self, request: VideoEditRequest,
                   ctx_exec: ExecutionContext) -> EngineResult:
        raise NotImplementedError("video edit — to be implemented in P6")

    async def upscale(self, request: VideoUpscaleRequest,
                      ctx_exec: ExecutionContext) -> EngineResult:
        raise NotImplementedError("video upscale — to be implemented")

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    async def _encode_t5(self, text: str) -> Any:
        if self._t5 is None:
            self._t5 = T5Encoder(self.ctx, self._encoders_path or "google/t5-v1_1-xxl")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._t5.encode, [text])

    def _load_model(self, model_key: str, config, entry, num_frames: int) -> Any:
        cache_key = f"{model_key}_{self.ctx.backend}"

        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        family = getattr(entry, "family", "") or ""
        if family == "wan":
            from .models.video.wan import WanTransformer
            model = WanTransformer(config, self.ctx, num_frames=num_frames)
        elif family == "ltx":
            from .models.video.ltx import LTXTransformer
            model = LTXTransformer(config, self.ctx, num_frames=num_frames)
        elif family == "cogvideox":
            raise RuntimeError(
                f"Video family 'cogvideox' is not implemented (model={model_key!r})."
            )
        else:
            raise RuntimeError(
                f"Unsupported video family {family!r} for model {model_key!r}."
            )

        version = getattr(entry, 'default_version', "")
        if version:
            version_info = entry.versions.get(version, {})
            local_path = version_info.get("local_path", "")
            if local_path:
                from .common.weights import load_safetensors
                weights_path = Path(self._registry._project_root or ".") / local_path
                if weights_path.exists():
                    weights = load_safetensors(str(weights_path), self.ctx)
                    model.load_weights(list(weights.items()), strict=False)
                    self.ctx.eval(model.parameters())

        if self._cache is not None:
            size_gb = parse_size_gb(getattr(entry, 'size', '0GB'))
            self._cache.put(cache_key, model, size_gb)

        return model

    def _release_model(self, model: Any) -> None:
        pass

    def _save_output(
        self,
        latents: Any,
        model: str,
        seed: int,
        fps: int,
        ctx_exec: ExecutionContext,
    ) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work = Path(ctx_exec.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        out_path = work / f"{model}_{seed}_{timestamp}.mp4"
        return str(out_path)
