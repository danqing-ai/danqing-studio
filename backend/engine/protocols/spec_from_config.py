"""Build v3 ``FamilySpec`` — catalog ``families`` block first, ``model_configs`` for tensor defaults."""

from __future__ import annotations

from typing import Any

from backend.catalog.family_spec_loader import family_spec_from_catalog
from backend.engine.protocols.plugin import FamilySpec


def family_spec_from_config(family_id: str, config: Any, *, media: str = "image") -> FamilySpec:
    """Build ``FamilySpec`` from on-disk catalog (required) with optional config field overrides."""
    return family_spec_from_catalog(family_id, config=config, media=media)
