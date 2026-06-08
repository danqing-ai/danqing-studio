"""Schema v2 profile / template expansion (catalog-owned; legacy shim in registry_profiles)."""

from __future__ import annotations

import copy
from typing import Any


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key == "profile":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_template_ref(value: Any, templates: dict[str, Any], *, context: str) -> Any:
    if isinstance(value, str) and value.startswith("$templates."):
        key = value[len("$templates.") :]
        if key not in templates:
            raise ValueError(f"{context}: unknown parameter template {key!r}")
        return copy.deepcopy(templates[key])
    return value


def expand_model_record(
    model_id: str,
    raw: dict[str, Any],
    *,
    profiles: dict[str, Any],
    templates: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Model {model_id!r}: entry must be an object")

    profile_name = raw.get("profile")
    if profile_name is not None and not isinstance(profile_name, str):
        raise ValueError(f"Model {model_id!r}: 'profile' must be a string when set")

    merged = copy.deepcopy(raw)
    if profile_name:
        profile_key = profile_name.strip()
        if not profile_key:
            raise ValueError(f"Model {model_id!r}: 'profile' must be non-empty")
        profile = profiles.get(profile_key)
        if not isinstance(profile, dict):
            raise ValueError(
                f"Model {model_id!r}: unknown profile {profile_key!r} "
                "(define it under top-level 'profiles' in models_registry.json)"
            )
        merged = deep_merge(profile, raw)

    params = merged.get("parameters")
    if isinstance(params, str):
        merged["parameters"] = resolve_template_ref(
            params, templates, context=f"Model {model_id!r} parameters"
        )

    return merged


def expand_registry_document(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of schema v2 registry JSON with profiles/templates applied."""
    if not isinstance(data, dict):
        raise ValueError("models_registry.json root must be an object")

    profiles = data.get("profiles") or data.get("ui_profiles") or {}
    templates = data.get("parameter_templates") or {}
    if profiles and not isinstance(profiles, dict):
        raise ValueError("'profiles' must be an object when present")
    if templates and not isinstance(templates, dict):
        raise ValueError("'parameter_templates' must be an object when present")

    out = copy.deepcopy(data)
    raw_models = data.get("models") or {}
    if not isinstance(raw_models, dict):
        raise ValueError("'models' must be an object")

    expanded_models: dict[str, Any] = {}
    for model_id, raw in raw_models.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Model {model_id!r}: entry must be an object")
        expanded_models[model_id] = expand_model_record(
            model_id,
            raw,
            profiles=profiles,
            templates=templates,
        )
    out["models"] = expanded_models
    return out
