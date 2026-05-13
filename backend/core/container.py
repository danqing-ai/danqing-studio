"""
Dependency injection container
"""

from typing import Type, TypeVar, Dict, Any, Optional

from backend.core.interfaces import (
    IConfigStore,
    IPresetStore,
    IDownloadService,
    ISettingsService,
    IPathResolver,
)
from backend.core.media_interfaces import IImageEngine, IVideoEngine
from backend.core.model_registry import ModelRegistry
from backend.engine.engine_registry import EngineRegistry
from backend.persistence.asset_store import SQLiteAssetStore
from backend.scheduler.task_scheduler import TaskScheduler

T = TypeVar("T")


class Container:
    def __init__(self):
        self._registrations: Dict[Type, Any] = {}
        self._singletons: Dict[Type, Any] = {}

    def register_instance(self, interface: Type[T], instance: T) -> None:
        self._singletons[interface] = instance

    def register_factory(self, interface: Type[T], factory: callable) -> None:
        self._registrations[interface] = factory

    def resolve(self, interface: Type[T]) -> T:
        if interface in self._singletons:
            return self._singletons[interface]
        if interface in self._registrations:
            instance = self._registrations[interface](self)
            self._singletons[interface] = instance
            return instance
        raise KeyError(f"Unregistered interface: {interface.__name__}")

    def try_resolve(self, interface: Type[T]) -> Optional[T]:
        try:
            return self.resolve(interface)
        except KeyError:
            return None

    def register_named(self, name: str, instance: Any) -> None:
        self._singletons[name] = instance

    def try_resolve_named(self, name: str) -> Optional[Any]:
        return self._singletons.get(name)


container = Container()


def get_container() -> Container:
    return container


def register_services(
    path_resolver: IPathResolver,
    config_store: IConfigStore,
    preset_store: IPresetStore,
    model_registry: ModelRegistry,
    engine_registry: EngineRegistry,
    image_media_engine: IImageEngine,
    download_service: IDownloadService,
    settings_service: ISettingsService,
    video_media_engine: Optional[IVideoEngine] = None,
    task_scheduler: Optional[TaskScheduler] = None,
    asset_store_v3: Optional[SQLiteAssetStore] = None,
    shared_model_cache=None,
) -> Container:
    c = get_container()
    c.register_instance(IPathResolver, path_resolver)
    c.register_instance(IConfigStore, config_store)
    c.register_instance(IPresetStore, preset_store)
    c.register_instance(ModelRegistry, model_registry)
    c.register_instance(EngineRegistry, engine_registry)
    c.register_instance(IImageEngine, image_media_engine)
    c.register_instance(IDownloadService, download_service)
    c.register_instance(ISettingsService, settings_service)
    if video_media_engine is not None:
        c.register_instance(IVideoEngine, video_media_engine)
    if task_scheduler is not None:
        c.register_instance(TaskScheduler, task_scheduler)
    if asset_store_v3 is not None:
        c.register_instance(SQLiteAssetStore, asset_store_v3)
    if shared_model_cache is not None:
        c.register_named("shared_model_cache", shared_model_cache)
    return c
