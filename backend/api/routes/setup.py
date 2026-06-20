"""Quick Setup API — hardware-aware model recommendations."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.api.deps import get_model_registry
from backend.core.container import get_container
from backend.core.interfaces import ISettingsService
from backend.core.model_registry import ModelRegistry
from backend.services.setup_recommendations import build_setup_recommendations
from backend.utils.path_utils import get_system_info

router = APIRouter(prefix="/api/setup", tags=["setup"])


def _settings_service() -> ISettingsService:
    return get_container().resolve(ISettingsService)


def _slot_to_dict(slot) -> dict:
    return {
        "slot": slot.slot,
        "status": slot.status,
        "model_id": slot.model_id,
        "version_key": slot.version_key,
        "estimated_gb": slot.estimated_gb,
        "warning": slot.warning,
        "reason": slot.reason,
        "installed": slot.installed,
        "name": slot.name,
        "version_name": slot.version_name,
        "size_human": slot.size_human,
    }


@router.get("/recommendations")
def get_setup_recommendations(
    reg: ModelRegistry = Depends(get_model_registry),
):
    service = _settings_service()
    settings = service.get_settings()
    sys_info = get_system_info()
    detailed = service.get_models_detailed_status()
    rec = build_setup_recommendations(
        reg,
        memory_gb=float(sys_info.get("memory_gb") or 0),
        mlx_memory_limit=int(settings.mlx_memory_limit or 120),
        detailed_status=detailed,
    )
    return {
        "reference_memory_gb": rec.reference_memory_gb,
        "memory_tier": rec.memory_tier,
        "available_backends": rec.available_backends,
        "primary_backend": rec.primary_backend,
        "system": {
            "platform": sys_info.get("platform"),
            "architecture": sys_info.get("architecture"),
            "memory_gb": sys_info.get("memory_gb"),
            "mlx_memory_limit": settings.mlx_memory_limit,
        },
        "slots": [_slot_to_dict(s) for s in rec.slots],
    }
