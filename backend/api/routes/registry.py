"""GET /api/registry — CatalogResponse DTO (v2/v3 loader projection + index)."""

from fastapi import APIRouter, Depends

from backend.api.deps import get_model_registry
from backend.catalog.api_dto import build_catalog_response
from backend.catalog.loader import load_catalog_json
from backend.core.model_registry import ModelRegistry

router = APIRouter(prefix="/api", tags=["registry"])


@router.get("/registry")
def get_public_registry(reg: ModelRegistry = Depends(get_model_registry)):
    data = load_catalog_json(reg.json_source_path)
    index = {
        mid: {
            "media": e.media,
            "family": e.family,
            "engine": e.engine,
            "actions": sorted(e.actions),
        }
        for mid, e in reg.all().items()
    }
    return build_catalog_response(data, index=index)
