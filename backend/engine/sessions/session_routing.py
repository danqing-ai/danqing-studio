"""Session vs pipeline routing helpers (registry actions + FamilyPlugin)."""

from __future__ import annotations

from typing import Any

from backend.core.contracts import parse_model_version
from backend.engine.registry.family_registry import is_family_plugin_registered

_IMAGE_CREATE_ACTION = "generate"
_IMAGE_EDIT_ACTION = "edit"
_AUDIO_EDIT_ACTION = "edit"
_UPSCALE_ACTION = "upscale"


def family_has_registered_plugin(
    model_field: str,
    registry: Any,
    *,
    expected_media: str | None = None,
) -> bool:
    """True when the model's family has a registered ``FamilyPlugin``."""
    model_key, _ = parse_model_version(model_field)
    entry = registry.get(model_key)
    if entry is None:
        return False
    if expected_media is not None and entry.media != expected_media:
        return False
    return is_family_plugin_registered(entry.family)


def _registry_entry(model_field: str, registry: Any) -> Any | None:
    model_key, _ = parse_model_version(model_field)
    return registry.get(model_key)


def entry_declares_action(entry: Any, api_action: str) -> bool:
    """True when expanded registry entry includes an API-level action (see ``api_action_frozenset``)."""
    acts = getattr(entry, "actions", frozenset())
    if not isinstance(acts, (frozenset, set, list, tuple)):
        return False
    return api_action in acts


def routes_with_plugin_and_action(
    model_field: str,
    registry: Any,
    *,
    expected_media: str,
    api_action: str,
) -> bool:
    if not family_has_registered_plugin(
        model_field, registry, expected_media=expected_media
    ):
        return False
    entry = _registry_entry(model_field, registry)
    if entry is None:
        return False
    return entry_declares_action(entry, api_action)
