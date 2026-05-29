"""GET /api/registry — 完整 models_registry.json + 派生索引。"""

from fastapi import APIRouter, Depends

from backend.api.deps import get_model_registry
from backend.core.model_registry import ModelRegistry

router = APIRouter(prefix="/api", tags=["registry"])


@router.get("/registry")
def get_public_registry(reg: ModelRegistry = Depends(get_model_registry)):
    expanded = ModelRegistry.expanded_document(reg.json_source_path)
    index = {
        mid: {
            "media": e.media,
            "family": e.family,
            "engine": e.engine,
            "actions": sorted(e.actions),
        }
        for mid, e in reg.all().items()
    }
    return {**expanded, "_index": index}
