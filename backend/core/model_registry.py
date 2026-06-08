"""models_registry.json 强类型视图 — plan EngineRegistry 依赖。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, FrozenSet, Literal, Optional, cast

from backend.catalog.loader import expand_catalog_document
from backend.core.registry_format import api_action_frozenset, media_from_record

MediaKind = Literal["image", "video", "audio", "llm"]


def _require_family(model_id: str, raw: dict[str, Any], media: MediaKind) -> str:
    fam = raw.get("family")
    if fam is None or (isinstance(fam, str) and not fam.strip()):
        raise ValueError(
            f"Model {model_id!r} (media={media}) missing required non-empty 'family' "
            "in models_registry.json; do not rely on inference heuristics."
        )
    return str(fam).strip()


@dataclass(frozen=True)
class ModelEntry:
    id: str
    raw: dict[str, Any]
    family: str
    media: MediaKind
    engine: str
    actions: FrozenSet[str]

    @property
    def parameters(self) -> dict[str, Any]:
        return self.raw.get("parameters") or {}

    @property
    def backends(self) -> tuple[str, ...]:
        """Preferred runtime order from ``models_registry.json`` (e.g. mlx / cuda)."""
        raw_b = self.raw.get("backends")
        if isinstance(raw_b, list) and raw_b:
            return tuple(str(x) for x in raw_b)
        return ("mlx",)


class ModelRegistry:
    def __init__(self, path: Path, models: dict[str, ModelEntry]):
        self._path = path
        self._models = models

    @property
    def json_source_path(self) -> Path:
        """磁盘上的 models_registry.json（供 GET /api/registry 原样返回）。"""
        return self._path

    @classmethod
    def load(cls, registry_json: Path) -> ModelRegistry:
        data = json.loads(registry_json.read_text(encoding="utf-8"))
        expanded = expand_catalog_document(data)
        raw_models = expanded.get("models") or {}
        built: dict[str, ModelEntry] = {}
        for mid, raw in raw_models.items():
            if not isinstance(raw, dict):
                continue
            media = cast(MediaKind, media_from_record(raw))
            eng = str(raw.get("engine") or "")
            acts_block = raw.get("actions") if isinstance(raw.get("actions"), dict) else {}
            actions = api_action_frozenset(acts_block, media=media)
            fam = _require_family(mid, raw, media)
            built[mid] = ModelEntry(
                id=mid,
                raw=raw,
                family=fam,
                media=media,
                engine=eng,
                actions=actions,
            )
        return cls(registry_json, built)

    @classmethod
    def expanded_document(cls, registry_json: Path) -> dict[str, Any]:
        """Parse registry JSON and apply profiles/templates (for API responses)."""
        data = json.loads(registry_json.read_text(encoding="utf-8"))
        return expand_catalog_document(data)

    def get(self, model_id: str) -> Optional[ModelEntry]:
        key = model_id.split(":", 1)[0]
        return self._models.get(key)

    def require(self, model_id: str) -> ModelEntry:
        e = self.get(model_id)
        if not e:
            raise KeyError(f"unknown model: {model_id}")
        return e

    def all(self) -> dict[str, ModelEntry]:
        return dict(self._models)
