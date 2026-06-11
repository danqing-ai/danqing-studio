"""
DanQingLoraTrainEngine — LoRA training (Flux.1-dev + Z-Image Base + Qwen-Image).
"""
from __future__ import annotations

from typing import Any, ClassVar

from backend.core.contracts import EngineResult, ExecutionContext, LoraTrainingRequest, parse_model_version
from backend.core.interfaces import IPathResolver
from backend.core.lora_train_interface import ILoraTrainEngine
from backend.engine.training.presets import TRAINABLE_BASE_MODELS
from backend.engine.memory_policy import prepare_host_for_lora_worker
from backend.engine.training.subprocess_runner import run_lora_training_subprocess


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
        if not self.supports_base_model(request.base_model):
            mid = request.base_model.split(":", 1)[0]
            raise RuntimeError(
                f"Base model {mid!r} is not supported for LoRA training in this release "
                f"(trainable: {sorted(TRAINABLE_BASE_MODELS)})"
            )
        mid, _ = parse_model_version(request.base_model)
        if mid == "z-image-turbo":
            raise RuntimeError(
                "LoRA training supports Z-Image Base (z-image) only; "
                "z-image-turbo is distilled and not trainable."
            )
        entry = self._registry.require(mid)
        family = str(getattr(entry, "family", ""))
        if family not in ("flux1", "z_image", "qwen_image"):
            raise RuntimeError(
                f"Base model {mid!r} (family={family!r}) has no LoRA training runner"
            )
        worker_memory_gb = prepare_host_for_lora_worker(mlx_runtime=self._runtimes.get("mlx"))
        result = await run_lora_training_subprocess(
            runner_family=family,
            request=request,
            exec_ctx=ctx,
            bootstrap_root=self._paths.get_bootstrap_root(),
            worker_memory_gb=worker_memory_gb,
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
