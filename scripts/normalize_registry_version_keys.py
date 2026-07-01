#!/usr/bin/env python3
"""Normalize MLX quant version keys in default_config/models_registry.json.

Run from repo root:
  python scripts/normalize_registry_version_keys.py
  python scripts/normalize_registry_version_keys.py --check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "default_config" / "models_registry.json"

sys.path.insert(0, str(REPO_ROOT))

from backend.core.version_keys import (  # noqa: E402
    canonical_primary_local_path,
    canonical_version_key,
    is_forbidden_vague_version_key,
    is_legacy_quant_local_path,
    is_shared_bundle_local_path,
    is_valid_mlx_quant_version_key,
)
from backend.core.version_labels import expected_version_display_name  # noqa: E402


def _target_key(old_key: str, ver: dict[str, Any]) -> str:
    return canonical_version_key(old_key, version_entry=ver) or old_key


def _skip_version_label(model: dict[str, Any], version_key: str) -> bool:
    catalog = model.get("catalog") or {}
    catalog_type = str(catalog.get("type") or "").lower()
    catalog_category = str(catalog.get("category") or "").lower()
    if catalog_type in ("lora", "controlnet") or catalog_category in ("loras", "controlnets"):
        return True
    vk = str(version_key or "").lower()
    return bool(re.match(r"^v[\w-]+$", vk) or re.search(r"\d+steps$", vk))


def _normalize_version_names(model: dict[str, Any], ver: dict[str, Any], new_key: str) -> bool:
    if _skip_version_label(model, new_key):
        return False
    label = expected_version_display_name(new_key, ver)
    if label is not None and ver.get("name") != label:
        ver["name"] = label
        return True
    return False


def _maybe_migrate_llm_int4(old_key: str, ver: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if old_key != "int4":
        return old_key, ver
    repo = str(ver.get("repo_id") or "").lower()
    if "4bit" not in repo and "mlx-community" not in repo:
        return old_key, ver
    new_ver = deepcopy(ver)
    new_ver["source_type"] = "prequantized"
    if isinstance(new_ver.get("local_path"), str):
        lp = new_ver["local_path"].replace("\\", "/")
        if lp.endswith("-int4"):
            new_ver["local_path"] = lp[: -len("-int4")] + "-mlx-q4"
    return "mlx-q4", new_ver


def _migrate_version_local_paths(model_id: str, version_key: str, ver: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    lp = ver.get("local_path")
    if isinstance(lp, str) and lp.strip() and not is_shared_bundle_local_path(lp):
        new_lp = canonical_primary_local_path(lp, version_key)
        if new_lp != lp.replace("\\", "/"):
            changes.append(f"local_path {lp!r} -> {new_lp!r} ({version_key})")
            ver["local_path"] = new_lp
    repos = ver.get("bundle_repos")
    if not isinstance(repos, list):
        return changes
    for item in repos:
        if not isinstance(item, dict):
            continue
        blp = item.get("local_path")
        if not isinstance(blp, str) or not blp.strip() or is_shared_bundle_local_path(blp):
            continue
        new_blp = canonical_primary_local_path(blp, version_key)
        if new_blp != blp.replace("\\", "/"):
            changes.append(f"bundle_repos local_path {blp!r} -> {new_blp!r} ({version_key})")
            item["local_path"] = new_blp
    return changes


def migrate_versions(versions: dict[str, Any], *, model: dict[str, Any], model_id: str) -> tuple[dict[str, Any], list[str]]:
    changes: list[str] = []
    out: dict[str, Any] = {}
    for old_key, ver in versions.items():
        if not isinstance(ver, dict):
            out[old_key] = ver
            continue
        new_key = _target_key(old_key, ver)
        new_ver = deepcopy(ver)
        llm_key, new_ver = _maybe_migrate_llm_int4(new_key, new_ver)
        if llm_key != new_key:
            changes.append(f"{new_key} -> {llm_key}")
            new_key = llm_key
        if new_key in out:
            raise ValueError(f"version key collision: {old_key!r} -> {new_key!r}")
        if new_key != old_key:
            changes.append(f"{old_key} -> {new_key}")
        if _normalize_version_names(model, new_ver, new_key):
            changes.append(f"name normalized ({new_key})")
        for ch in _migrate_version_local_paths(model_id, new_key, new_ver):
            changes.append(ch)
        out[new_key] = new_ver

    for new_key, ver in out.items():
        if not isinstance(ver, dict):
            continue
        parent = ver.get("from_version")
        if isinstance(parent, str) and parent.strip():
            resolved = _target_key(parent.strip(), out.get(parent.strip(), {}))
            if resolved != parent:
                changes.append(f"from_version {parent} -> {resolved} ({new_key})")
                ver["from_version"] = resolved

    return out, changes


def migrate_registry(data: dict[str, Any]) -> list[str]:
    all_changes: list[str] = []
    models = data.get("models") or {}
    for model_id, model in models.items():
        if not isinstance(model, dict):
            continue
        dist = model.get("distribution") or {}
        versions = dist.get("versions")
        if not isinstance(versions, dict):
            continue
        new_versions, changes = migrate_versions(versions, model=model, model_id=model_id)
        if changes:
            dist["versions"] = new_versions
            for ch in changes:
                all_changes.append(f"{model_id}: {ch}")
    return all_changes


def validate_registry(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    models = data.get("models") or {}
    for model_id, model in models.items():
        if not isinstance(model, dict):
            continue
        dist = model.get("distribution") or {}
        versions = dist.get("versions")
        if not isinstance(versions, dict):
            continue
        catalog = model.get("catalog") or {}
        catalog_type = str(catalog.get("type") or "").lower()
        catalog_category = str(catalog.get("category") or "").lower()
        skip_dtype_key_check = catalog_type in ("lora", "controlnet") or catalog_category in (
            "loras",
            "controlnets",
        )
        for vk, ver in versions.items():
            if not isinstance(ver, dict):
                continue
            st = str(ver.get("source_type") or "")
            canonical = canonical_version_key(vk, version_entry=ver)
            if canonical != vk:
                errors.append(f"{model_id}.{vk}: legacy key (expected {canonical})")
            lower = vk.lower()
            if lower.startswith("mlx-") and "-bit" in lower:
                errors.append(f"{model_id}.{vk}: forbidden mlx-*bit key")
            if lower.startswith("community-") and "bit" in lower:
                errors.append(f"{model_id}.{vk}: forbidden community-*bit key")
            if lower in ("mlx-int4", "mlx-int8") and st == "derived":
                errors.append(f"{model_id}.{vk}: derived must use int4/int8")
            if lower in ("mlx-int4", "mlx-int8") and st == "prequantized":
                errors.append(f"{model_id}.{vk}: prequantized must use mlx-q*")
            if vk == "mlx" and st in ("full", "prequantized"):
                errors.append(f"{model_id}.{vk}: bare mlx version key forbidden")
            if is_forbidden_vague_version_key(vk):
                errors.append(f"{model_id}.{vk}: forbidden vague version key (use fp16/bf16/fp8 or descriptive variant)")
            lower_stem = lower.split("-")[0]
            allowed_stems = {
                "int4", "int8", "fp16", "bf16", "fp8", "mlx", "xl", "encoders",
            }
            if (
                not skip_dtype_key_check
                and lower not in ("xl-sft", "encoders")
                and not lower.startswith("mlx-")
                and lower_stem not in allowed_stems
            ):
                errors.append(
                    f"{model_id}.{vk}: version key must express weight dtype (got {vk!r})"
                )
            lp = ver.get("local_path")
            if isinstance(lp, str) and lp.endswith("-original"):
                errors.append(f"{model_id}.{vk}: legacy local_path suffix -original ({lp!r})")
            if isinstance(lp, str) and lp.endswith("-turbo-quant"):
                errors.append(f"{model_id}.{vk}: legacy local_path suffix -turbo-quant ({lp!r})")
            q_match = re.match(r"^mlx-q(\d+)", lower)
            if q_match and int(q_match.group(1)) not in (4, 8):
                errors.append(f"{model_id}.{vk}: disallowed MLX quant tier (only mlx-q4/mlx-q8/mlx-bf16)")
            if st in ("derived", "prequantized") and lower.startswith("mlx-") and not is_valid_mlx_quant_version_key(
                vk, source_type=st
            ):
                errors.append(f"{model_id}.{vk}: invalid key for source_type={st}")
            lp = ver.get("local_path")
            if isinstance(lp, str) and is_legacy_quant_local_path(lp):
                errors.append(f"{model_id}.{vk}: legacy local_path {lp!r}")
            if isinstance(lp, str) and lp.strip() and not is_shared_bundle_local_path(lp):
                expected_lp = canonical_primary_local_path(lp, vk)
                if expected_lp != lp.replace("\\", "/"):
                    errors.append(
                        f"{model_id}.{vk}: local_path must be {expected_lp!r} (got {lp!r})"
                    )
            for br in ver.get("bundle_repos") or []:
                if not isinstance(br, dict):
                    continue
                blp = br.get("local_path")
                if not isinstance(blp, str) or not blp.strip() or is_shared_bundle_local_path(blp):
                    continue
                expected_blp = canonical_primary_local_path(blp, vk)
                if expected_blp != blp.replace("\\", "/"):
                    errors.append(
                        f"{model_id}.{vk}: bundle_repos local_path must be {expected_blp!r} (got {blp!r})"
                    )
            if not skip_dtype_key_check and not _skip_version_label(model, vk):
                expected_name = expected_version_display_name(vk, ver)
                if expected_name is not None and ver.get("name") != expected_name:
                    errors.append(
                        f"{model_id}.{vk}: version name must be {expected_name!r} (got {ver.get('name')!r})"
                    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate only; exit 1 if registry is not normalized.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=REGISTRY_PATH,
        help=f"Registry JSON path (default: {REGISTRY_PATH})",
    )
    args = parser.parse_args()

    path = args.path
    data = json.loads(path.read_text(encoding="utf-8"))

    if args.check:
        errors = validate_registry(data)
        if errors:
            print("Registry version key violations:", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            return 1
        print(f"OK: {path}")
        return 0

    changes = migrate_registry(data)
    if not changes:
        print("No version key changes needed.")
        return 0

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {path} ({len(changes)} changes):")
    for ch in changes:
        print(f"  - {ch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
