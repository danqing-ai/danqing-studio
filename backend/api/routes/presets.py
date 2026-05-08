"""Plan presets.py — 与 IPresetStore 对齐的只读列表（写入仍可用 /api/settings/presets）。"""

from fastapi import APIRouter, Depends

from backend.core.container import get_container
from backend.core.interfaces import IPresetStore

router = APIRouter(prefix="/api/presets", tags=["presets"])


def get_preset_store() -> IPresetStore:
    return get_container().resolve(IPresetStore)


@router.get("")
def list_presets(store: IPresetStore = Depends(get_preset_store)):
    return {"presets": store.load_all()}
