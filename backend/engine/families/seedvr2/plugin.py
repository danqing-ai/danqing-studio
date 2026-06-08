"""SeedVR2 v3 ``FamilyPlugin`` factory (job / upscale paradigm)."""

from __future__ import annotations

from pathlib import Path

from backend.engine.config.model_configs import get_config_class
from backend.engine.families._upscale_backbone import UpscalePluginBackbone
from backend.engine.platform.session import PlatformSession
from backend.engine.protocols.plugin import FamilyPlugin
from backend.engine.protocols.spec_from_config import family_spec_from_config
from backend.engine.registry.family_registry import register_family


def build_seedvr2_plugin(
    platform: PlatformSession,
    *,
    model_id: str,
    bundle_root: Path,
    version_key: str | None = None,
) -> FamilyPlugin:
    _ = platform, model_id, bundle_root, version_key
    config = get_config_class("seedvr2")()
    spec = family_spec_from_config("seedvr2", config, media="image")
    return FamilyPlugin(family_id="seedvr2", spec=spec, backbone=UpscalePluginBackbone(spec))


def register_seedvr2_plugin() -> None:
    register_family("seedvr2", build_seedvr2_plugin)
