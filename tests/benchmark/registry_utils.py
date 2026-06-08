"""Registry + workspace helpers for image eval benchmark."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_REGISTRY_PATH = _REPO_ROOT / "default_config" / "models_registry.json"
_WORKSPACE_POINTER = _REPO_ROOT / "default_config" / "workspace.pointer.json"

EDIT_ACTIONS = frozenset({"rewrite", "retouch", "extend"})

_MIN_SAFETENSORS_BYTES = 1024


def repo_root() -> Path:
    return _REPO_ROOT


def resolve_benchmark_data_root() -> Path:
    """``default_config/workspace.pointer.json`` → workspace root, else repo root."""
    if _WORKSPACE_POINTER.is_file():
        try:
            data = json.loads(_WORKSPACE_POINTER.read_text(encoding="utf-8"))
            custom = str(data.get("custom_workspace_dir") or "").strip()
            if custom:
                root = Path(custom).expanduser().resolve()
                if root.is_dir():
                    return root
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return _REPO_ROOT


def load_registry() -> dict[str, Any]:
    from backend.catalog.loader import expand_catalog_document, load_catalog_json

    return expand_catalog_document(load_catalog_json(_REGISTRY_PATH))


def _profile_engine(reg: dict[str, Any], profile_id: str) -> str:
    profiles = reg.get("profiles") or reg.get("ui_profiles") or {}
    profile = profiles.get(profile_id) or {}
    return str(profile.get("engine") or "").strip()


def _model_engine(reg: dict[str, Any], spec: dict[str, Any]) -> str:
    explicit = str(spec.get("engine") or "").strip()
    if explicit:
        return explicit
    profile_id = str(spec.get("profile") or "").strip()
    if profile_id:
        return _profile_engine(reg, profile_id)
    return ""


def param_default(spec: dict[str, Any], name: str, fallback: float | int | str) -> float | int | str:
    params = spec.get("parameters") or {}
    node = params.get(name) or {}
    if isinstance(node, dict) and "default" in node:
        return node["default"]
    return fallback


def resolve_default_bundle_dir(model_id: str, *, reg: dict[str, Any] | None = None) -> Path | None:
    reg = reg or load_registry()
    spec = (reg.get("models") or {}).get(model_id)
    if not isinstance(spec, dict):
        return None
    versions = spec.get("versions") or {}
    if not isinstance(versions, dict):
        return None

    chosen: dict[str, Any] | None = None
    for _key, ver in versions.items():
        if isinstance(ver, dict) and ver.get("default"):
            chosen = ver
            break
    if chosen is None and isinstance(versions.get("fp16"), dict):
        chosen = versions["fp16"]
    if chosen is None:
        for ver in versions.values():
            if isinstance(ver, dict) and ver.get("local_path"):
                chosen = ver
                break
    if not chosen:
        return None

    rel = str(chosen.get("local_path") or "").strip()
    if not rel:
        return None
    root = resolve_benchmark_data_root()
    path = (root / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
    return path


def _incomplete_download_reason(bundle_root: Path) -> str | None:
    for path in bundle_root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.endswith(".incomplete"):
            return f"incomplete download ({path.name})"
        if path.suffix == ".safetensors" and path.stat().st_size < _MIN_SAFETENSORS_BYTES:
            return f"empty weight file ({path.name})"
    return None


def _bundle_manifest_module():
    import importlib.util
    import sys

    name = "dq_bundle_manifest"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached

    path = _REPO_ROOT / "backend" / "core" / "bundle_manifest.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load bundle manifest module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def bundle_ready(model_id: str, *, reg: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Return (ready, skip_reason). Uses FamilyBundleContract when family is known."""
    path = resolve_default_bundle_dir(model_id, reg=reg)
    if path is None or not path.is_dir():
        return False, "missing default bundle"

    incomplete = _incomplete_download_reason(path)
    if incomplete:
        return False, incomplete

    reg = reg or load_registry()
    spec = (reg.get("models") or {}).get(model_id)
    if not isinstance(spec, dict):
        return False, "unknown model"
    family = str(spec.get("family") or "").strip()
    if not family:
        return True, ""

    try:
        manifest = _bundle_manifest_module()
        manifest.assert_bundle_ready_for_family(path, family=family, model_id=model_id)
    except RuntimeError as exc:
        return False, str(exc)
    return True, ""


def bundle_installed(model_id: str, *, reg: dict[str, Any] | None = None) -> bool:
    ready, _ = bundle_ready(model_id, reg=reg)
    return ready


def iter_image_eval_models(*, reg: dict[str, Any] | None = None) -> Iterator[tuple[str, dict[str, Any]]]:
    reg = reg or load_registry()
    models = reg.get("models") or {}
    for model_id, spec in sorted(models.items()):
        if not isinstance(spec, dict):
            continue
        if spec.get("media") != "image":
            continue
        actions = spec.get("actions") or {}
        if not isinstance(actions, dict) or not actions:
            continue
        engine = _model_engine(reg, spec)
        if engine and engine != "danqing-image":
            continue
        yield model_id, spec
