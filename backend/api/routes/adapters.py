"""Plan adapters：统一适配器列表（当前为已安装 LoRA；registry_slots 预留给注册表扩展）。"""

from typing import Any, List, Optional

from fastapi import APIRouter, Query

from backend.core.container import get_container
from backend.core.interfaces import ISettingsService

router = APIRouter(prefix="/api/adapters", tags=["adapters"])


@router.get("")
def list_adapters(
    for_model: Optional[str] = Query(None, description="图像模型 id，仅返回与之兼容的 LoRA"),
):
    service = get_container().resolve(ISettingsService)
    pick = getattr(service, "lora_adapter_picklist", None)
    items: List[dict[str, Any]] = list(pick(for_model)) if pick else []
    return {"items": items, "registry_slots": []}
