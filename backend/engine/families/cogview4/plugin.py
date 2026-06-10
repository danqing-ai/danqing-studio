"""Cogview4Transformer v3 ``FamilyPlugin`` factory."""

from __future__ import annotations

from pathlib import Path

from backend.engine.config.model_configs import get_config_class
from backend.engine.families._image_backbone import ImagePluginBackbone
from backend.engine.platform.session import PlatformSession
from backend.engine.protocols.plugin import FamilyPlugin
from backend.engine.protocols.spec_from_config import family_spec_from_config
from backend.engine.registry.family_registry import register_family


def build_cogview4_plugin(
    platform: PlatformSession,
    *,
    model_id: str,
    bundle_root: Path,
    version_key: str | None = None,
) -> FamilyPlugin:
    _ = platform, model_id, bundle_root, version_key
    config = get_config_class("cogview4")()
    spec = family_spec_from_config("cogview4", config, media="image")
    return FamilyPlugin(
        family_id="cogview4",
        spec=spec,
        backbone=ImagePluginBackbone(spec),
    )


def register_cogview4_plugin() -> None:
    register_family("cogview4", build_cogview4_plugin)
