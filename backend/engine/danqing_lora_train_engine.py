"""
DanQingLoraTrainEngine — LoRA training (Flux.1-dev + Z-Image Base).
"""
from __future__ import annotations

from typing import Any, ClassVar

from backend.core.contracts import EngineResult, ExecutionContext, LoraTrainingRequest, parse_model_version
from backend.core.interfaces import IPathResolver
from backend.core.lora_train_interface import ILoraTrainEngine
from backend.engine.training.flux_dreambooth_mlx import run_flux_dreambooth_training
from backend.engine.training.presets import TRAINABLE_BASE_MODELS
from backend.engine.training.z_image_dreambooth_mlx import run_z_image_dreambooth_training


class DanQingLoraTrainEngine(ILoraTrainEngine):
    engine_id: ClassVar[str] = "danqing-lora-train"

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

    def supports_base_model(self, base_model_id: str) -> bool:
        mid, _ = parse_model_version(base_model_id)
        if mid not in TRAINABLE_BASE_MODELS:
            return False
        try:
            entry = self._registry.require(mid)
        except Exception:
            return False
        params = getattr(entry, "parameters", None) or {}
        if isinstance(params, dict) and not params.get("lora_support", True):
            return False
        return "mlx" in (entry.backends or [])

    def _resolve_mlx_runtime(self) -> Any:
        rt = self._runtimes.get("mlx")
        if rt is None:
            raise RuntimeError("LoRA training requires MLX runtime (Apple Silicon)")
        return rt

    async def train(self, request: LoraTrainingRequest, ctx: ExecutionContext) -> EngineResult:
        import asyncio

        if not self.supports_base_model(request.base_model):
            mid = request.base_model.split(":", 1)[0]
            raise RuntimeError(
                f"Base model {mid!r} is not supported for LoRA training in this release "
                f"(trainable: {sorted(TRAINABLE_BASE_MODELS)})"
            )
        runtime = self._resolve_mlx_runtime()
        mid, _ = parse_model_version(request.base_model)
        if mid == "z-image-turbo":
            raise RuntimeError(
                "LoRA training supports Z-Image Base (z-image) only; "
                "z-image-turbo is distilled and not trainable."
            )
        entry = self._registry.require(mid)
        family = str(getattr(entry, "family", ""))
        if family == "flux1":
            runner = run_flux_dreambooth_training
        elif family == "z_image":
            runner = run_z_image_dreambooth_training
        else:
            raise RuntimeError(
                f"Base model {mid!r} (family={family!r}) has no LoRA training runner"
            )
        result = await asyncio.to_thread(
            runner,
            request,
            ctx,
            registry=self._registry,
            project_root=self._paths.get_project_root(),
            runtime=runtime,
            path_resolver=self._paths,
        )
        return EngineResult(
            primary_asset_id="",
            metadata={
                "training": result,
                "user_lora_id": result.get("user_lora_id", ""),
                "adapter_path": result.get("adapter_path", ""),
                "output_name": result.get("output_name", ""),
            },
        )
