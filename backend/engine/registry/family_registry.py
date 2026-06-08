"""Family plugin factory registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from backend.engine.platform.session import PlatformSession
from backend.engine.protocols.plugin import FamilyPlugin

FamilyBuilder = Callable[..., FamilyPlugin]

_BUILDERS: dict[str, FamilyBuilder] = {}


def register_family(family_id: str, builder: FamilyBuilder) -> None:
    _BUILDERS[family_id] = builder


def is_family_plugin_registered(family_id: str) -> bool:
    return family_id in _BUILDERS


def registered_family_ids() -> frozenset[str]:
    return frozenset(_BUILDERS.keys())


def build_family_plugin(
    family_id: str,
    platform: PlatformSession,
    *,
    model_id: str,
    bundle_root: Path,
    version_key: str | None = None,
) -> FamilyPlugin:
    """Instantiate ``FamilyPlugin`` for a family. Fail loud if not registered."""
    factory = _BUILDERS.get(family_id)
    if factory is None:
        raise RuntimeError(
            f"No FamilyPlugin registered for family {family_id!r}; "
            "register in bootstrap_family_plugins()."
        )
    return factory(
        platform,
        model_id=model_id,
        bundle_root=bundle_root,
        version_key=version_key,
    )
