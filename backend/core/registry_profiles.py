"""models_registry.json profile / parameter_templates expansion (fail loud)."""

from __future__ import annotations

import copy
from typing import Any


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key == "profile":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
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
        merged = _deep_merge(profile, raw)

    params = merged.get("parameters")
    if isinstance(params, str):
        merged["parameters"] = resolve_template_ref(
            params, templates, context=f"Model {model_id!r} parameters"
        )

    return merged


def expand_registry_document(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of registry JSON with profiles/templates applied to each model."""
    if not isinstance(data, dict):
        raise ValueError("models_registry.json root must be an object")

    profiles = data.get("profiles") or {}
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


def validate_registry_document(data: dict[str, Any]) -> list[str]:
    """Validate profiles/templates and expanded model records; return error messages."""
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

    profiles = data.get("profiles") or {}
    models = data.get("models") or {}
    if not isinstance(profiles, dict) or not isinstance(models, dict):
        return hints

    referenced: set[str] = set()
    for raw in models.values():
        if isinstance(raw, dict):
            profile_name = raw.get("profile")
            if isinstance(profile_name, str) and profile_name.strip():
                referenced.add(profile_name.strip())

    for profile_id in profiles:
        if profile_id not in referenced:
            hints.append(f"profile {profile_id!r}: unused (no model references it)")

    try:
        expanded = expand_registry_document(data)
    except ValueError:
        return hints

    for model_id, raw in models.items():
        if not isinstance(raw, dict):
            continue
        profile_name = raw.get("profile")
        if not isinstance(profile_name, str) or not profile_name.strip():
            continue
        profile_key = profile_name.strip()
        profile = profiles.get(profile_key)
        if not isinstance(profile, dict):
            continue

        for key in _STRIP_TOP_KEYS:
            if key in raw and raw.get(key) == profile.get(key):
                hints.append(
                    f"model {model_id!r}: top-level {key!r} duplicates profile "
                    f"{profile_key!r} (strip via apply_standard_profile)"
                )

        raw_params = raw.get("parameters")
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
    profile = (data.get("profiles") or {}).get(profile_key)
    if not isinstance(profile, dict):
        raise ValueError(f"missing profiles.{profile_key}")

    profile_params = profile.get("parameters") if isinstance(profile.get("parameters"), dict) else {}
    strip_params = _STRIP_PARAM_KEYS | extra_strip_params
    edits = 0
    for raw in (data.get("models") or {}).values():
        if not isinstance(raw, dict) or raw.get("category") == "loras":
            continue
        params = raw.get("parameters")
        if not isinstance(params, dict) or param_has_key not in params:
            continue
        engine = raw.get("engine") or profile.get("engine")
        if engine != engine_id:
            continue

        if raw.get("profile") != profile_key:
            raw["profile"] = profile_key
            edits += 1

        for key in _STRIP_TOP_KEYS:
            if raw.get(key) == profile.get(key):
                raw.pop(key, None)
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
