"""models_registry.json profile / parameter_templates expansion (fail loud).

v2 expansion lives in ``backend.catalog.expand_v2``; runtime loads via ``expand_catalog_document``.
"""

from __future__ import annotations

from typing import Any

from backend.catalog.expand_v2 import deep_merge as _deep_merge
from backend.catalog.expand_v2 import expand_registry_document

__all__ = [
    "_deep_merge",
    "apply_standard_profile",
    "audit_registry_document",
    "expand_registry_document",
    "validate_registry_document",
]


def validate_registry_document(data: dict[str, Any]) -> list[str]:
    """Validate profiles/templates and expanded model records; return error messages."""
    if isinstance(data, dict) and int(data.get("schema_version", 2)) >= 3:
        from backend.catalog.validate import validate_v3_document

        return validate_v3_document(data)

    errors: list[str] = []
    try:
        expanded = expand_registry_document(data)
    except ValueError as exc:
        return [str(exc)]

    profiles = data.get("profiles") or {}
    if isinstance(profiles, dict):
        for profile_id, profile in profiles.items():
            if not isinstance(profile, dict):
                errors.append(f"Profile {profile_id!r}: must be an object")
                continue
            if "engine" not in profile and "media" not in profile and "parameters" not in profile:
                errors.append(
                    f"Profile {profile_id!r}: should declare at least one of engine/media/parameters"
                )

    models = expanded.get("models") or {}
    for model_id, model in models.items():
        if not isinstance(model, dict):
            errors.append(f"Model {model_id!r}: expanded entry must be an object")
            continue
        fam = model.get("family")
        if fam is None or (isinstance(fam, str) and not fam.strip()):
            errors.append(f"Model {model_id!r}: missing required non-empty 'family' after profile merge")
        actions = model.get("actions")
        if actions is not None and not isinstance(actions, (dict, list)):
            errors.append(f"Model {model_id!r}: 'actions' must be object or list when set")

        profile_name = (data.get("models") or {}).get(model_id, {}).get("profile")
        if profile_name and profile_name not in (data.get("profiles") or {}):
            errors.append(f"Model {model_id!r}: references missing profile {profile_name!r}")

    return errors


def audit_registry_document(data: dict[str, Any]) -> list[str]:
    """Non-blocking shrink hints: duplicate profile fields, unused profiles, inheritance chain."""
    hints: list[str] = []
    if not isinstance(data, dict):
        return hints

    profiles = data.get("profiles") or data.get("ui_profiles") or {}
    models = data.get("models") or {}
    if not isinstance(profiles, dict) or not isinstance(models, dict):
        return hints

    referenced: set[str] = set()
    for raw in models.values():
        if not isinstance(raw, dict):
            continue
        profile_name = raw.get("profile")
        if isinstance(profile_name, str) and profile_name.strip():
            referenced.add(profile_name.strip())
        ui = raw.get("ui")
        if isinstance(ui, dict):
            extends = ui.get("extends")
            if isinstance(extends, str) and extends.strip():
                referenced.add(extends.strip())

    for profile_id in profiles:
        if profile_id not in referenced:
            hints.append(f"profile {profile_id!r}: unused (no model references it)")

    try:
        from backend.catalog.loader import expand_catalog_document

        expanded = expand_catalog_document(data)
    except ValueError:
        return hints

    for model_id, raw in models.items():
        if not isinstance(raw, dict):
            continue
        profile_name = raw.get("profile")
        if not isinstance(profile_name, str) or not profile_name.strip():
            ui = raw.get("ui")
            if isinstance(ui, dict):
                profile_name = ui.get("extends")
        if not isinstance(profile_name, str) or not profile_name.strip():
            continue
        profile_key = profile_name.strip()
        profile = profiles.get(profile_key)
        if not isinstance(profile, dict):
            continue

        for key in _STRIP_TOP_KEYS:
            catalog = raw.get("catalog") if isinstance(raw.get("catalog"), dict) else raw
            if key in catalog and catalog.get(key) == profile.get(key):
                hints.append(
                    f"model {model_id!r}: top-level {key!r} duplicates profile "
                    f"{profile_key!r} (strip via apply_standard_profile)"
                )

        raw_params = raw.get("parameters")
        if isinstance(raw.get("ui"), dict) and isinstance(raw["ui"].get("parameters"), dict):
            raw_params = raw["ui"]["parameters"]
        profile_params = profile.get("parameters") if isinstance(profile.get("parameters"), dict) else {}
        if isinstance(raw_params, dict) and isinstance(profile_params, dict):
            for param_key, param_val in raw_params.items():
                if param_key in profile_params and profile_params.get(param_key) == param_val:
                    hints.append(
                        f"model {model_id!r}: parameters.{param_key!r} duplicates profile "
                        f"{profile_key!r}"
                    )

    return hints


PROFILE_STANDARD = "image_dit_standard"
PROFILE_VIDEO = "video_dit_standard"
_STRIP_PARAM_KEYS = frozenset({"preview_mode", "preview_interval_steps", "preview_max_edge"})
_STRIP_TOP_KEYS = frozenset({"category", "engine", "type"})


def _apply_profile_shrink(
    data: dict[str, Any],
    *,
    profile_key: str,
    engine_id: str,
    param_has_key: str,
    extra_strip_params: frozenset[str],
) -> int:
    profiles = data.get("profiles") or data.get("ui_profiles") or {}
    profile = profiles.get(profile_key)
    if not isinstance(profile, dict):
        raise ValueError(f"missing profiles.{profile_key}")

    profile_params = profile.get("parameters") if isinstance(profile.get("parameters"), dict) else {}
    strip_params = _STRIP_PARAM_KEYS | extra_strip_params
    edits = 0
    for raw in (data.get("models") or {}).values():
        if not isinstance(raw, dict):
            continue
        catalog = raw.get("catalog") if isinstance(raw.get("catalog"), dict) else raw
        if catalog.get("category") == "loras":
            continue
        params = raw.get("parameters")
        if isinstance(raw.get("ui"), dict) and isinstance(raw["ui"].get("parameters"), dict):
            params = raw["ui"]["parameters"]
        if not isinstance(params, dict) or param_has_key not in params:
            continue
        engine = catalog.get("engine") or profile.get("engine")
        if engine != engine_id:
            continue

        if raw.get("profile") != profile_key:
            if int(data.get("schema_version", 2)) >= 3:
                ui = raw.setdefault("ui", {})
                if isinstance(ui, dict) and ui.get("extends") != profile_key:
                    ui["extends"] = profile_key
                    edits += 1
            else:
                raw["profile"] = profile_key
                edits += 1

        for key in _STRIP_TOP_KEYS:
            if key in catalog and catalog.get(key) == profile.get(key):
                catalog.pop(key, None)
                edits += 1

        for key in strip_params:
            if key in params:
                params.pop(key)
                edits += 1

        for key in ("lora_support", "seed_support"):
            if key in params and params.get(key) is True and profile_params.get(key) is True:
                params.pop(key)
                edits += 1

    return edits


def apply_standard_profile(data: dict[str, Any]) -> int:
    """Attach standard profiles and drop duplicated fields (registry shrink, expanded doc unchanged)."""
    edits = _apply_profile_shrink(
        data,
        profile_key=PROFILE_STANDARD,
        engine_id="danqing-image",
        param_has_key="preview_mode",
        extra_strip_params=frozenset(),
    )
    edits += _apply_profile_shrink(
        data,
        profile_key=PROFILE_VIDEO,
        engine_id="danqing-video",
        param_has_key="seed_support",
        extra_strip_params=frozenset(),
    )
    return edits
