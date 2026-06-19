"""Validate schema v3 catalog documents."""

from __future__ import annotations

from typing import Any

from backend.catalog.loader import expand_catalog_document
from backend.core.registry_format import media_from_record


def validate_v3_document(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        expanded = expand_catalog_document(data)
    except ValueError as exc:
        return [str(exc)]

    families = data.get("families") or {}
    if not isinstance(families, dict):
        errors.append("'families' must be an object in schema v3")

    ui_profiles = data.get("ui_profiles") or {}
    if ui_profiles and not isinstance(ui_profiles, dict):
        errors.append("'ui_profiles' must be an object when present")

    models = data.get("models") or {}
    for model_id, raw in models.items():
        if not isinstance(raw, dict):
            errors.append(f"Model {model_id!r}: entry must be an object")
            continue
        catalog = raw.get("catalog")
        if isinstance(catalog, dict):
            for field in ("successor", "distilled_from", "distilled_variant"):
                value = catalog.get(field)
                if value is None:
                    continue
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"Model {model_id!r}: catalog.{field} must be a non-empty string")
                elif value.strip() == model_id:
                    errors.append(f"Model {model_id!r}: catalog.{field} must not point to itself")
        runtime = raw.get("runtime")
        if not isinstance(runtime, dict):
            errors.append(f"Model {model_id!r}: missing runtime object")
            continue
        fam = runtime.get("family")
        if fam is None or (isinstance(fam, str) and not fam.strip()):
            errors.append(f"Model {model_id!r}: runtime.family required")
        elif isinstance(families, dict) and fam not in families:
            errors.append(f"Model {model_id!r}: runtime.family {fam!r} not in families")

        ui = raw.get("ui")
        if isinstance(ui, dict):
            extends = ui.get("extends")
            if extends and extends not in ui_profiles:
                errors.append(f"Model {model_id!r}: ui.extends references missing profile {extends!r}")

    for model_id, model in (expanded.get("models") or {}).items():
        if not isinstance(model, dict):
            continue
        try:
            media_from_record(model)
        except Exception as exc:
            errors.append(f"Model {model_id!r}: {exc}")

    return errors
