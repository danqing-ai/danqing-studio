"""DiffRhythm audio v3 ``FamilyPlugin`` factory."""

from __future__ import annotations

from pathlib import Path
from backend.engine.config.model_configs import get_config_class
from backend.engine.families._audio_backbone import AudioPluginBackbone
from backend.engine.platform.session import PlatformSession
from backend.engine.protocols.plugin import FamilyPlugin, FamilySpec
from backend.engine.protocols.spec_from_config import family_spec_from_config
from backend.engine.registry.family_registry import register_family



def build_diffrhythm_plugin(
    platform: PlatformSession,
    *,
    model_id: str,
    bundle_root: Path,
    version_key: str | None = None,
) -> FamilyPlugin:
    _ = platform, model_id, bundle_root, version_key
    config = get_config_class("diffrhythm")()
    spec = family_spec_from_config("diffrhythm", config, media="audio")
    return FamilyPlugin(family_id="diffrhythm", spec=spec, backbone=AudioPluginBackbone(spec))


def register_diffrhythm_plugin() -> None:
    register_family("diffrhythm", build_diffrhythm_plugin)
