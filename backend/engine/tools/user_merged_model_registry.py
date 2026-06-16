"""User-merged Z-Image model registry (workspace ``config/user_merged_models.json`` + registry patch)."""

from __future__ import annotations

import copy
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.core.contracts import parse_model_version


def user_merged_models_path(config_dir: Path) -> Path:
    return config_dir / "user_merged_models.json"


def _load_index(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"items": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {"items": []}
    items = data.get("items")
    if not isinstance(items, list):
        data["items"] = []
    return data


def _save_index(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def list_user_merged_models(config_dir: Path) -> list[dict[str, Any]]:
    return list(_load_index(user_merged_models_path(config_dir)).get("items") or [])


def merged_model_id_from_output_name(output_name: str) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", output_name.strip().lower()).strip("-")
    if not slug:
        raise RuntimeError("output_name must contain at least one alphanumeric character")
    return f"z-image-merged-{slug}"


def _read_registry_document(registry_path: Path) -> dict[str, Any]:
    if not registry_path.is_file():
        raise RuntimeError(f"models registry not found: {registry_path}")
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"{registry_path}: registry root must be an object")
    return data


def _write_registry_document(registry_path: Path, data: dict[str, Any]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _v3_template_entry(doc: dict[str, Any], template_model_id: str) -> dict[str, Any]:
    models = doc.get("models")
    if not isinstance(models, dict):
        raise RuntimeError("models_registry.json missing models object")
    raw = models.get(template_model_id)
    if not isinstance(raw, dict):
        raise RuntimeError(
            f"template model {template_model_id!r} not found in models_registry.json"
        )
    return copy.deepcopy(raw)


def _build_merged_registry_entry(
    template: dict[str, Any],
    *,
    model_id: str,
    display_name: str,
    local_path: str,
    merge_manifest: dict[str, Any],
) -> dict[str, Any]:
    entry = copy.deepcopy(template)
    catalog = entry.setdefault("catalog", {})
    if not isinstance(catalog, dict):
        catalog = {}
        entry["catalog"] = catalog
    catalog["name"] = {"zh": display_name, "en": display_name}
    catalog["description"] = {
        "zh": f"本地 Z-Image DiT 合并模型（{merge_manifest.get('merge_method', 'merge')}）",
        "en": f"Local Z-Image DiT merge ({merge_manifest.get('merge_method', 'merge')})",
    }
    catalog["source"] = "local"
    catalog["recommended"] = False
    catalog["type"] = catalog.get("type") or "diffusion"
    catalog["category"] = catalog.get("category") or "base_models"
    catalog["engine"] = catalog.get("engine") or "danqing-image"
    catalog["media"] = catalog.get("media") or "image"

    runtime = entry.setdefault("runtime", {})
    if isinstance(runtime, dict):
        runtime["family"] = "z_image"
        runtime["backends"] = ["mlx"]

    entry["distribution"] = {
        "versions": {
            "fp16": {
                "name": {"zh": "FP16 本地合并", "en": "FP16 local merge"},
                "size": "31GB",
                "default": True,
                "local_path": local_path,
                "source_type": "local",
            }
        }
    }
    entry["user_merged"] = {
        "model_id": model_id,
        "merge_manifest": merge_manifest,
    }
    return entry


def register_merged_z_image_model(
    *,
    registry_path: Path,
    config_dir: Path,
    output_name: str,
    local_path: str,
    template_model_id: str,
    merge_manifest: dict[str, Any],
    task_id: str = "",
) -> dict[str, Any]:
    """Append merged model to workspace registry + user index; returns index row."""
    model_id = merged_model_id_from_output_name(output_name)
    doc = _read_registry_document(registry_path)
    models = doc.setdefault("models", {})
    if not isinstance(models, dict):
        raise RuntimeError("models_registry.json models must be an object")
    if model_id in models:
        raise RuntimeError(f"registry model id already exists: {model_id}")

    template_mid, _ = parse_model_version(template_model_id)
    template = _v3_template_entry(doc, template_mid)
    display_name = output_name.strip() or model_id
    models[model_id] = _build_merged_registry_entry(
        template,
        model_id=model_id,
        display_name=display_name,
        local_path=local_path,
        merge_manifest=merge_manifest,
    )
    _write_registry_document(registry_path, doc)

    index_path = user_merged_models_path(config_dir)
    index = _load_index(index_path)
    items: list[dict[str, Any]] = list(index.get("items") or [])
    row = {
        "id": model_id,
        "name": display_name,
        "local_path": local_path,
        "template_model": template_mid,
        "merge_manifest": merge_manifest,
        "task_id": task_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    items.insert(0, row)
    index["items"] = items
    _save_index(index_path, index)
    return row
