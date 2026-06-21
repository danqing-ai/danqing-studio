"""Parse registry ``distribution.dependencies`` entries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DependencySpec:
    model_id: str
    version: str | None = None


def parse_dependencies(raw: Any) -> list[DependencySpec]:
    if not isinstance(raw, list):
        return []
    out: list[DependencySpec] = []
    for item in raw:
        if isinstance(item, str):
            model_id = item.strip()
            if model_id:
                out.append(DependencySpec(model_id=model_id))
            continue
        if isinstance(item, dict):
            model_id = str(item.get("model_id") or item.get("model") or "").strip()
            if not model_id:
                continue
            version = item.get("version")
            version_key = str(version).strip() if isinstance(version, str) and version.strip() else None
            out.append(DependencySpec(model_id=model_id, version=version_key))
    return out


def dependency_model_ids(raw: Any) -> list[str]:
    return [spec.model_id for spec in parse_dependencies(raw)]
