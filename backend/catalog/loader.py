"""Load catalog JSON (v2 or v3) and expand to flat model records for engine + API."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from backend.catalog.expand_v2 import deep_merge as _deep_merge
from backend.catalog.expand_v2 import expand_registry_document
from backend.catalog.schema_v3 import SCHEMA_VERSION_V3


def schema_version(data: dict[str, Any]) -> int:
    raw = data.get("schema_version", 2)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 2


def load_catalog_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: catalog root must be an object")
    return data


def _scalar_from_param_spec(value: Any) -> Any:
    if isinstance(value, dict) and "default" in value:
        return value["default"]
    return value


def flatten_v3_model(
    model_id: str,
    record: dict[str, Any],
    doc: dict[str, Any],
) -> dict[str, Any]:
    """Project v3 nested model → v2-expanded flat record (engine + frontend compat)."""
    if not isinstance(record, dict):
        raise ValueError(f"Model {model_id!r}: entry must be an object")

    ui_profiles = doc.get("ui_profiles") or {}
    catalog = record.get("catalog") if isinstance(record.get("catalog"), dict) else {}
    runtime = record.get("runtime") if isinstance(record.get("runtime"), dict) else {}
    ui = record.get("ui") if isinstance(record.get("ui"), dict) else {}
    distribution = record.get("distribution") if isinstance(record.get("distribution"), dict) else {}

    merged: dict[str, Any] = {}
    extends = ui.get("extends")
    if isinstance(extends, str) and extends.strip():
        profile = ui_profiles.get(extends.strip())
        if isinstance(profile, dict):
            merged = copy.deepcopy(profile)

    merged.update(copy.deepcopy(catalog))

    family = runtime.get("family")
    if family is None or (isinstance(family, str) and not family.strip()):
        raise ValueError(f"Model {model_id!r}: runtime.family is required in schema v3")
    merged["family"] = str(family).strip()

    backends = runtime.get("backends")
    if isinstance(backends, list) and backends:
        merged["backends"] = [str(b) for b in backends]

    merged["actions"] = copy.deepcopy(record.get("actions") or {})
    if isinstance(distribution.get("versions"), dict):
        merged["versions"] = copy.deepcopy(distribution["versions"])
    if "dependencies" in distribution:
        merged["dependencies"] = copy.deepcopy(distribution["dependencies"])

    params: dict[str, Any] = {}
    profile_params = merged.get("parameters")
    if isinstance(profile_params, dict):
        params = copy.deepcopy(profile_params)
    ui_params = ui.get("parameters")
    if isinstance(ui_params, dict):
        params = _deep_merge(params, ui_params)

    overrides = runtime.get("overrides")
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            params[key] = value

    merged["parameters"] = params
    return merged


def expand_catalog_document(data: dict[str, Any]) -> dict[str, Any]:
    """Return catalog copy with each model expanded to the legacy flat shape."""
    if not isinstance(data, dict):
        raise ValueError("catalog root must be an object")

    ver = schema_version(data)
    if ver < SCHEMA_VERSION_V3:
        return expand_registry_document(data)

    out = copy.deepcopy(data)
    raw_models = data.get("models") or {}
    if not isinstance(raw_models, dict):
        raise ValueError("'models' must be an object")

    expanded_models: dict[str, Any] = {}
    for model_id, raw in raw_models.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Model {model_id!r}: entry must be an object")
        expanded_models[model_id] = flatten_v3_model(model_id, raw, data)
    out["models"] = expanded_models
    return out
