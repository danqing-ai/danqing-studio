"""``GET /api/registry`` view model — flattened models + index (not raw file mirror)."""

from __future__ import annotations

from typing import Any

from backend.catalog.loader import expand_catalog_document, schema_version
from backend.catalog.schema_v3 import SCHEMA_VERSION_V3


def build_catalog_response(
    data: dict[str, Any],
    *,
    index: dict[str, Any],
) -> dict[str, Any]:
    """Assemble API response from on-disk catalog (v2 or v3)."""
    expanded = expand_catalog_document(data)
    ver = schema_version(data)

    response: dict[str, Any] = {
        "schema_version": ver,
        "engines": data.get("engines") or {},
        "categories": data.get("categories") or {},
        "models": expanded.get("models") or {},
        "_index": index,
    }

    if ver >= SCHEMA_VERSION_V3:
        response["ui_profiles"] = data.get("ui_profiles") or {}
        response["families"] = data.get("families") or {}
    else:
        if data.get("profiles"):
            response["profiles"] = data.get("profiles")
        if data.get("parameter_templates"):
            response["parameter_templates"] = data.get("parameter_templates")

    return response
