#!/usr/bin/env python3
"""Validate models_registry structural contracts used by runtime/tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "default_config" / "models_registry.json"

HUNYUAN_REQUIRED_IDS = (
    "hunyuan-video-1.5-480p-t2v",
    "hunyuan-video-1.5-480p-i2v",
    "hunyuan-video-1.5-i2v-step-distill",
    "hunyuan-video-1.5-1080p-sr",
)
HUNYUAN_REPO_ID = "Tencent-Hunyuan/HunyuanVideo-1.5"


def _load_registry() -> dict:
    with REGISTRY.open("r", encoding="utf-8") as f:
        return json.load(f)


def _check_hunyuan_presence(models: dict, failures: list[str]) -> None:
    for mid in HUNYUAN_REQUIRED_IDS:
        if mid not in models:
            failures.append(f"missing required Hunyuan model: {mid}")


def _check_hunyuan_versions(models: dict, failures: list[str]) -> None:
    for mid in HUNYUAN_REQUIRED_IDS:
        model = models.get(mid)
        if not isinstance(model, dict):
            continue
        if model.get("source") != "modelscope":
            failures.append(f"{mid}: source must be modelscope")
        versions = model.get("versions")
        if not isinstance(versions, dict) or "original" not in versions:
            failures.append(f"{mid}: versions.original is required")
            continue
        ver = versions["original"]
        if not isinstance(ver, dict):
            failures.append(f"{mid}: versions.original must be object")
            continue
        if not ver.get("hunyuan_ms_variant"):
            failures.append(f"{mid}: versions.original.hunyuan_ms_variant is required")
        bundle = ver.get("bundle_repos")
        if not isinstance(bundle, list) or not bundle:
            failures.append(f"{mid}: versions.original.bundle_repos must be non-empty array")
            continue
        first = bundle[0] if isinstance(bundle[0], dict) else {}
        if first.get("repo_id") != HUNYUAN_REPO_ID:
            failures.append(f"{mid}: first bundle repo must be {HUNYUAN_REPO_ID}")
        if "companion_repo_id" in ver:
            failures.append(f"{mid}: versions.original should not contain companion_repo_id")
        if "shared_te_local_path" in ver:
            failures.append(f"{mid}: versions.original should not contain shared_te_local_path")


def _check_t2v_specific(models: dict, failures: list[str]) -> None:
    mid = "hunyuan-video-1.5-480p-t2v"
    model = models.get(mid)
    if not isinstance(model, dict):
        return
    params = model.get("parameters", {})
    if params.get("text_encoder_qwen_local") != "models/Text/qwen2.5-vl-7b-instruct":
        failures.append(f"{mid}: parameters.text_encoder_qwen_local mismatch")
    if not params.get("text_encoder_release_after_encode"):
        failures.append(f"{mid}: parameters.text_encoder_release_after_encode must be true")
    ver = ((model.get("versions") or {}).get("original") or {})
    bundle = ver.get("bundle_repos") if isinstance(ver, dict) else None
    if not isinstance(bundle, list) or len(bundle) < 3:
        failures.append(f"{mid}: bundle_repos must include Hunyuan + Qwen + ByT5")
        return
    repo_ids = [r.get("repo_id") for r in bundle if isinstance(r, dict)]
    if "Qwen/Qwen2.5-VL-7B-Instruct" not in repo_ids:
        failures.append(f"{mid}: bundle_repos must include Qwen/Qwen2.5-VL-7B-Instruct")
    if "google/byt5-small" not in repo_ids:
        failures.append(f"{mid}: bundle_repos must include google/byt5-small")


def _check_distill_flags(models: dict, failures: list[str]) -> None:
    mid = "hunyuan-video-1.5-i2v-step-distill"
    model = models.get(mid)
    if not isinstance(model, dict):
        return
    params = model.get("parameters", {})
    if params.get("supports_guidance") is not False:
        failures.append(f"{mid}: parameters.supports_guidance must be false")
    if params.get("negative_prompt_support") is not False:
        failures.append(f"{mid}: parameters.negative_prompt_support must be false")
    if "guide_scale" in params:
        failures.append(f"{mid}: parameters.guide_scale must not be present")


def _check_sr_flags(models: dict, failures: list[str]) -> None:
    mid = "hunyuan-video-1.5-1080p-sr"
    model = models.get(mid)
    if not isinstance(model, dict):
        return
    params = model.get("parameters", {})
    if not params.get("vae_spatial_tiling"):
        failures.append(f"{mid}: parameters.vae_spatial_tiling must be true")


def main() -> int:
    data = _load_registry()
    models = data.get("models") or {}
    failures: list[str] = []

    _check_hunyuan_presence(models, failures)
    _check_hunyuan_versions(models, failures)
    _check_t2v_specific(models, failures)
    _check_distill_flags(models, failures)
    _check_sr_flags(models, failures)

    if failures:
        print("Models registry contract check failed:", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print("Models registry contracts OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

