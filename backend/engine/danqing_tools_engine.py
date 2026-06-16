"""DanQingToolsEngine — model merge and other offline MLX tools."""
from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from backend.core.contracts import EngineResult, ExecutionContext, LogEvent, ZImageMergeRequest, parse_model_version
from backend.core.interfaces import IPathResolver
from backend.core.tools_interface import IToolsEngine
from backend.engine.tools.z_image_merge import run_z_image_merge


class DanQingToolsEngine(IToolsEngine):
    engine_id: ClassVar[str] = "danqing-tools"

    def __init__(
        self,
        path_resolver: IPathResolver,
        registry: Any,
        runtimes: dict[str, Any],
    ):
        self._paths = path_resolver
        self._registry = registry
        self._runtimes = runtimes

    def is_available(self) -> bool:
        return "mlx" in self._runtimes

    def supports_z_image_merge(self, model_id: str) -> bool:
        mid, _ = parse_model_version(model_id)
        try:
            entry = self._registry.require(mid)
        except Exception:
            return False
        return str(getattr(entry, "family", "")) == "z_image" and "mlx" in (entry.backends or [])

    def _mlx_ctx(self) -> Any:
        rt = self._runtimes.get("mlx")
        if rt is None:
            raise RuntimeError("Z-Image merge requires MLX runtime (Apple Silicon)")
        return rt

    async def merge_z_image(self, request: ZImageMergeRequest, ctx: ExecutionContext) -> EngineResult:
        if not self.supports_z_image_merge(request.model_a):
            raise RuntimeError(f"model_a {request.model_a!r} is not a mergeable z_image MLX bundle")
        mlx = self._mlx_ctx()
        project_root = self._paths.get_project_root()

        def _log(level: str, msg: str) -> None:
            ctx.on_log(LogEvent(level=level, message=msg))

        result = await asyncio.to_thread(
            run_z_image_merge,
            registry=self._registry,
            project_root=project_root,
            ctx=mlx,
            method=request.method,
            model_a=request.model_a,
            model_b=request.model_b,
            model_c=request.model_c,
            alpha=float(request.alpha),
            output_name=request.output_name,
            work_dir=ctx.work_dir,
            on_log=_log,
            auto_register=bool(request.auto_register),
            registry_path=self._paths.get_models_registry_path(),
            config_dir=self._paths.get_workspace_config_dir(),
            task_id=ctx.task_id or "",
        )
        if request.auto_register and result.get("registered_model_id"):
            self._registry.reload()
        return EngineResult(primary_asset_id="", metadata={"z_image_merge": result})
