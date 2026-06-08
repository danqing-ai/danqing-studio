"""One-shot v2 → v3 catalog migration."""

from __future__ import annotations

import copy
from typing import Any

from backend.catalog.schema_v3 import (
    CATALOG_MODEL_KEYS,
    DISTRIBUTION_KEYS,
    ENGINE_PARAMETER_KEYS,
    SCHEMA_VERSION_V3,
)
from backend.catalog.expand_v2 import expand_registry_document


def _scalar_from_param_spec(value: Any) -> Any:
    if isinstance(value, dict) and "default" in value:
        return value["default"]
    return value


def _family_record_from_config(family_id: str, backends: list[str]) -> dict[str, Any]:
    from backend.catalog.family_spec_loader import family_spec_from_model_config
    from backend.engine.config.model_configs import get_config_class

    config = get_config_class(family_id)()
    media = "audio" if family_id in ("ace_step", "diffrhythm") else (
        "video" if family_id in ("wan", "ltx", "hunyuan") else "image"
    )
    spec = family_spec_from_model_config(family_id, config, media=media)
    record: dict[str, Any] = {
        "paradigm": spec.paradigm,
        "media": [spec.media],
        "backends": sorted(set(backends)),
        "latent_layout": spec.latent_layout,
        "cfg_mode": spec.cfg_mode,
        "latent_channels": spec.latent_channels,
        "vae_scale": spec.vae_scale,
        "hooks": sorted(spec.hooks),
    }
    if spec.step_kwargs_profile:
        record["step_kwargs_profile"] = spec.step_kwargs_profile
    encoder_type = getattr(config, "encoder_type", None)
    if encoder_type:
        record["encoder_type"] = encoder_type
    return record


def _split_engine_ui_parameters(parameters: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ui_params: dict[str, Any] = {}
    overrides: dict[str, Any] = {}
    for key, value in parameters.items():
        if key in ENGINE_PARAMETER_KEYS:
            overrides[key] = _scalar_from_param_spec(value)
        else:
            ui_params[key] = copy.deepcopy(value)
    return ui_params, overrides


def _split_model_v3(
    model_id: str,
    raw: dict[str, Any],
    expanded: dict[str, Any],
    *,
    profile_name: str | None,
) -> dict[str, Any]:
    catalog = {k: copy.deepcopy(expanded[k]) for k in CATALOG_MODEL_KEYS if k in expanded}

    runtime: dict[str, Any] = {
        "family": expanded["family"],
    }
    backends = expanded.get("backends")
    if isinstance(backends, list) and backends:
        runtime["backends"] = [str(b) for b in backends]

    params = expanded.get("parameters") if isinstance(expanded.get("parameters"), dict) else {}
    ui_params, overrides = _split_engine_ui_parameters(params)
    if overrides:
        runtime["overrides"] = overrides

    ui: dict[str, Any] = {}
    if ui_params:
        ui["parameters"] = ui_params
    if profile_name:
        ui["extends"] = profile_name

    distribution: dict[str, Any] = {}
    for key in DISTRIBUTION_KEYS:
        if key in expanded:
            distribution[key] = copy.deepcopy(expanded[key])

    out: dict[str, Any] = {
        "catalog": catalog,
        "runtime": runtime,
        "actions": copy.deepcopy(expanded.get("actions") or {}),
    }
    if ui:
        out["ui"] = ui
    if distribution:
        out["distribution"] = distribution
    return out


def migrate_v2_to_v3(data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Convert schema v2 document to v3. Returns (document, validation report lines)."""
    report: list[str] = []
    if not isinstance(data, dict):
        raise ValueError("catalog root must be an object")

    ver = int(data.get("schema_version", 2))
    if ver >= SCHEMA_VERSION_V3:
        report.append("already schema_version >= 3; returning a deep copy unchanged")
        return copy.deepcopy(data), report

    expanded = expand_registry_document(data)
    profiles = data.get("profiles") or {}
    if profiles and not isinstance(profiles, dict):
        raise ValueError("'profiles' must be an object when present")

    family_backends: dict[str, set[str]] = {}
    expanded_models = expanded.get("models") or {}
    for mid, model in expanded_models.items():
        if not isinstance(model, dict):
            continue
        fam = model.get("family")
        if not fam:
            report.append(f"model {mid!r}: missing family after expand (skipped for families)")
            continue
        fam_s = str(fam).strip()
        backends = model.get("backends") or ["mlx"]
        if isinstance(backends, list):
            family_backends.setdefault(fam_s, set()).update(str(b) for b in backends)

    families: dict[str, Any] = {}
    for family_id, backends in sorted(family_backends.items()):
        try:
            families[family_id] = _family_record_from_config(family_id, sorted(backends))
        except KeyError:
            report.append(
                f"family {family_id!r}: no model_configs entry; emitting minimal family row"
            )
            families[family_id] = {
                "paradigm": "diffusion",
                "media": ["image"],
                "backends": sorted(backends),
            }

    raw_models = data.get("models") or {}
    v3_models: dict[str, Any] = {}
    for model_id, raw in raw_models.items():
        if not isinstance(raw, dict):
            report.append(f"model {model_id!r}: not an object (skipped)")
            continue
        expanded_model = expanded_models.get(model_id)
        if not isinstance(expanded_model, dict):
            report.append(f"model {model_id!r}: missing after profile expand (skipped)")
            continue
        profile_name = raw.get("profile")
        if profile_name is not None and not isinstance(profile_name, str):
            report.append(f"model {model_id!r}: invalid profile type")
            profile_name = None
        v3_models[model_id] = _split_model_v3(
            model_id,
            raw,
            expanded_model,
            profile_name=profile_name.strip() if isinstance(profile_name, str) and profile_name.strip() else None,
        )

    referenced_profiles: set[str] = set()
    for raw in raw_models.values():
        if isinstance(raw, dict):
            pn = raw.get("profile")
            if isinstance(pn, str) and pn.strip():
                referenced_profiles.add(pn.strip())
    for profile_id in profiles:
        if profile_id not in referenced_profiles:
            report.append(f"profile {profile_id!r}: unused after migration")

    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_V3,
        "engines": copy.deepcopy(data.get("engines") or {}),
        "categories": copy.deepcopy(data.get("categories") or {}),
        "ui_profiles": copy.deepcopy(profiles),
        "families": families,
        "models": v3_models,
    }
    templates = data.get("parameter_templates")
    if templates:
        out["parameter_templates"] = copy.deepcopy(templates)
    return out, report
