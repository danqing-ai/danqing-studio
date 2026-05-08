# Plan 7.10 路由模块（媒体 + 注册表 + 系统）
from .adapters import router as adapters_router
from .assets import router as assets_router
from .audios import router as audios_router
from .download import router as download_router
from .gallery import router as gallery_router
from .images import router as images_router
from .models import router as models_router
from .presets import router as presets_router
from .queue import router as queue_router
from .registry import router as registry_router
from .settings import router as settings_router
from .system import router as system_router
from .tasks import router as tasks_router
from .videos import router as videos_router

__all__ = [
    "adapters_router",
    "assets_router",
    "audios_router",
    "download_router",
    "gallery_router",
    "images_router",
    "models_router",
    "presets_router",
    "queue_router",
    "registry_router",
    "settings_router",
    "system_router",
    "tasks_router",
    "videos_router",
]
