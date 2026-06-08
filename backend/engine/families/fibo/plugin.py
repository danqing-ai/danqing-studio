"""FIBO v3 ``FamilyPlugin`` factory (Phase 2 — phased create via session helpers)."""

from __future__ import annotations

from pathlib import Path
from backend.engine.config.model_configs import get_config_class
from backend.engine.families._image_backbone import ImagePluginBackbone
from backend.engine.platform.session import PlatformSession
from backend.engine.protocols.plugin import FamilyPlugin, FamilySpec
from backend.engine.protocols.spec_from_config import family_spec_from_config
from backend.engine.registry.family_registry import register_family



def build_fibo_plugin(
    platform: PlatformSession,
    *,
    model_id: str,
    bundle_root: Path,
    version_key: str | None = None,
) -> FamilyPlugin:
    _ = platform, model_id, bundle_root, version_key
    config = get_config_class("fibo")()
    spec = family_spec_from_config("fibo", config, media="image")
    return FamilyPlugin(family_id="fibo", spec=spec, backbone=ImagePluginBackbone(spec))


def register_fibo_plugin() -> None:
    register_family("fibo", build_fibo_plugin)
