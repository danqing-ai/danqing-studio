"""Catalog v3 — on-disk schema, loader, API DTO projection."""

from backend.catalog.api_dto import build_catalog_response
from backend.catalog.family_spec_loader import (
    clear_families_cache,
    family_spec_from_catalog,
    load_families_block,
)
from backend.catalog.loader import expand_catalog_document, load_catalog_json, schema_version

__all__ = [
    "build_catalog_response",
    "clear_families_cache",
    "expand_catalog_document",
    "family_spec_from_catalog",
    "load_catalog_json",
    "load_families_block",
    "schema_version",
]


def migrate_v2_to_v3(*args, **kwargs):
    from backend.catalog.migrate_v2 import migrate_v2_to_v3 as _fn

    return _fn(*args, **kwargs)
