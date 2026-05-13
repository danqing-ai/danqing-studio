"""models_registry.json 强类型视图 — plan EngineRegistry 依赖。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, FrozenSet, Literal, Optional, cast

from backend.core.registry_format import api_action_frozenset, media_from_record

MediaKind = Literal["image", "video", "audio"]


def _infer_audio_family(model_id: str) -> str:
    return "stub"


def _infer_image_family(model_id: str) -> str:
    m = model_id.split(":", 1)[0]
    if m.startswith("seedvr2"):
        return "seedvr2"
    if m == "flux-redux":
        return "redux"
    if "controlnet" in m or m in (
        "flux-canny-controlnet",
        "flux-depth-controlnet",
        "flux-fill-controlnet",
    ):
        return "controlnet"
    if m == "flux1-kontext":
        return "kontext"
    if m.startswith("flux2-"):
        return "flux2"
    if m.startswith("z-image"):
        return "z_image"
    if m.startswith("fibo"):
        return "fibo"
    if m == "qwen-image":
        return "qwen_image"
    return "flux1"


def _infer_video_family(model_id: str) -> str:
    m = model_id.split(":", 1)[0].lower()
    if m.startswith("ltx"):
        return "ltx"
    return "wan"


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
        raw_models = data.get("models") or {}
        built: dict[str, ModelEntry] = {}
        for mid, raw in raw_models.items():
            if not isinstance(raw, dict):
                continue
            media = cast(MediaKind, media_from_record(raw))
            eng = str(raw.get("engine") or "")
            acts_block = raw.get("actions") if isinstance(raw.get("actions"), dict) else {}
            actions = api_action_frozenset(acts_block, media=media)
            if media == "video":
                fam = raw.get("family") or _infer_video_family(mid)
            elif media == "audio":
                fam = raw.get("family") or _infer_audio_family(mid)
            else:
                fam = raw.get("family") or _infer_image_family(mid)
            built[mid] = ModelEntry(
                id=mid,
                raw=raw,
                family=str(fam),
                media=media,
                engine=eng,
                actions=actions,
            )
        return cls(registry_json, built)

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
