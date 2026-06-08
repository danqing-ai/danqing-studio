"""Flux1 v3 ``FamilyPlugin`` factory (Phase 2 — phased create via session helpers)."""

from __future__ import annotations

from pathlib import Path
from backend.engine.config.model_configs import get_config_class
from backend.engine.families._image_backbone import ImagePluginBackbone
from backend.engine.platform.session import PlatformSession
from backend.engine.protocols.plugin import FamilyPlugin, FamilySpec
from backend.engine.protocols.spec_from_config import family_spec_from_config
from backend.engine.registry.family_registry import register_family



def build_flux1_plugin(
    platform: PlatformSession,
    *,
    model_id: str,
    bundle_root: Path,
    version_key: str | None = None,
) -> FamilyPlugin:
    _ = platform, model_id, bundle_root, version_key
    config = get_config_class("flux1")()
    spec = family_spec_from_config("flux1", config, media="image")
    return FamilyPlugin(
        family_id="flux1",
        spec=spec,
        backbone=ImagePluginBackbone(spec),
    )


def register_flux1_plugin() -> None:
    register_family("flux1", build_flux1_plugin)
