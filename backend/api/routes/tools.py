"""Offline model tools — Z-Image merge, export, …"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.api.deps import get_engine_registry, get_task_scheduler
from backend.core import task_kinds as TK
from backend.core.contracts import ZImageMergeRequest
from backend.core.interfaces import IPathResolver
from backend.core.container import get_container
from backend.engine.engine_registry import EngineRegistry
from backend.scheduler.task_scheduler import TaskScheduler

router = APIRouter(prefix="/api/tools", tags=["tools"])


def _paths() -> IPathResolver:
    return get_container().resolve(IPathResolver)


@router.get("/z-image/merge/models")
def list_z_image_merge_models(
    engines: EngineRegistry = Depends(get_engine_registry),
):
    """List registry z_image MLX models eligible for DiT merge (no download)."""
    from backend.core.container import get_container
    from backend.core.model_registry import ModelRegistry

    registry: ModelRegistry = get_container().resolve(ModelRegistry)
    tools = engines.get_tools()
    rows: list[dict[str, object]] = []
    for mid, entry in registry.all().items():
        if str(getattr(entry, "family", "")) != "z_image":
            continue
        if "mlx" not in (entry.backends or []):
            continue
        if not tools.supports_z_image_merge(mid):
            continue
        raw_name = (entry.raw or {}).get("name") if hasattr(entry, "raw") else None
        rows.append(
            {
                "id": mid,
                "name": raw_name or mid,
                "media": getattr(entry, "media", "image"),
            }
        )
    rows.sort(key=lambda r: str(r.get("id", "")))
    return {"models": rows, "mlx_available": tools.is_available()}


@router.get("/z-image/merge/merged")
def list_user_merged_z_image_models():
    """List user-merged Z-Image bundles registered in workspace config."""
    from backend.engine.tools.user_merged_model_registry import list_user_merged_models

    config_dir = _paths().get_workspace_config_dir()
    items = list_user_merged_models(config_dir)
    return {"items": items}


@router.post("/z-image/merge", status_code=202)
async def submit_z_image_merge(
    body: ZImageMergeRequest,
    sched: TaskScheduler = Depends(get_task_scheduler),
    engines: EngineRegistry = Depends(get_engine_registry),
):
    tools = engines.get_tools()
    if not tools.is_available():
        raise HTTPException(
            409,
            detail={"code": "mlx_required", "message": "Z-Image merge requires MLX runtime (Apple Silicon)"},
        )
    if not tools.supports_z_image_merge(body.model_a):
        raise HTTPException(
            409,
            detail={"code": "unsupported", "message": f"model_a {body.model_a!r} is not a z_image MLX bundle"},
        )
    if not tools.supports_z_image_merge(body.model_b):
        raise HTTPException(
            409,
            detail={"code": "unsupported", "message": f"model_b {body.model_b!r} is not a z_image MLX bundle"},
        )
    if body.method == "add_difference":
        if not body.model_c:
            raise HTTPException(
                400,
                detail={"code": "invalid", "message": "add_difference merge requires model_c"},
            )
        if not tools.supports_z_image_merge(body.model_c):
            raise HTTPException(
                409,
                detail={"code": "unsupported", "message": f"model_c {body.model_c!r} is not a z_image MLX bundle"},
            )
    name = (body.output_name or "").strip()
    if not name:
        raise HTTPException(400, detail={"code": "invalid", "message": "output_name is required"})
    out_dir = _paths().get_project_root() / "models" / "Image" / f"{name}-fp16"
    if out_dir.exists():
        raise HTTPException(
            409,
            detail={"code": "exists", "message": f"output bundle already exists: {out_dir.name}"},
        )
    if body.auto_register:
        from backend.engine.tools.user_merged_model_registry import merged_model_id_from_output_name

        try:
            mid = merged_model_id_from_output_name(name)
        except RuntimeError as exc:
            raise HTTPException(400, detail={"code": "invalid", "message": str(exc)}) from exc
        from backend.core.container import get_container
        from backend.core.model_registry import ModelRegistry

        registry: ModelRegistry = get_container().resolve(ModelRegistry)
        if registry.get(mid) is not None:
            raise HTTPException(
                409,
                detail={"code": "exists", "message": f"registry model id already exists: {mid}"},
            )

    r = await sched.submit(
        kind=TK.TOOLS_Z_IMAGE_MERGE,
        model_id=body.model_a.split(":", 1)[0],
        params=body.model_dump(),
        priority=body.priority,
    )
    return {"task": r}
