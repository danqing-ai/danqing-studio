"""Assemble DanQing diffusers-style Hunyuan bundles from ModelScope native trees."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

_HUNYUAN_MS_CONFIG_ONLY_VARIANTS = frozenset({"480p_t2v"})
_GENERATED_BUNDLE_FILES = frozenset({"model_index.json"})

# SigLIP vision encoder is NOT shipped in Tencent-Hunyuan/HunyuanVideo-1.5 (see upstream
# checkpoints-download.md); fetch image_encoder/ from FLUX.1-Redux-dev via bundle_repos.
HUNYUAN_SIGLIP_REPO_ID = "black-forest-labs/FLUX.1-Redux-dev"
HUNYUAN_SIGLIP_ALLOW_PATTERNS = ["image_encoder/**"]


def is_hunyuan_ms_bundle_assembled(bundle_root: Path) -> bool:
    """True after ``assemble_hunyuan_modelscope_bundle`` (flat ``transformer/config.json``)."""
    return (Path(bundle_root) / "transformer" / "config.json").is_file()


def hunyuan_raw_download_patterns(variant: str) -> list[str]:
    """ModelScope snapshot globs for the native ``transformer/<variant>/`` tree."""
    v = str(variant or "").strip()
    if not v:
        raise RuntimeError("hunyuan_ms_variant is required for ModelScope HunyuanVideo allow_patterns.")
    if v in _HUNYUAN_MS_CONFIG_ONLY_VARIANTS:
        return [f"transformer/{v}/config.json", "vae/**"]
    return [f"transformer/{v}/**", "vae/**"]


def hunyuan_i2v_variant(variant: str) -> bool:
    return "i2v" in str(variant or "").lower()


def hunyuan_assembled_bundle_patterns() -> list[str]:
    """Validate layout after assembly (variant subdir hoisted to ``transformer/``)."""
    return ["transformer/config.json", "vae/**"]


def hunyuan_bundle_ready_patterns(variant: str) -> list[str]:
    """Post-install readiness globs (includes SigLIP for I2V variants)."""
    patterns = list(hunyuan_assembled_bundle_patterns())
    if hunyuan_i2v_variant(variant):
        patterns.extend(HUNYUAN_SIGLIP_ALLOW_PATTERNS)
    return patterns


def hunyuan_modelscope_allow_patterns(variant: str) -> list[str]:
    """Default ModelScope partial-download globs for a native ``transformer/<variant>/`` tree."""
    return hunyuan_raw_download_patterns(variant)


def _strip_generated_allow_patterns(patterns: list[str]) -> list[str]:
    return [p for p in patterns if str(p).strip() not in _GENERATED_BUNDLE_FILES]


def resolve_hunyuan_modelscope_allow_patterns(
    ver_config: dict[str, Any] | None,
    *,
    primary_spec: dict[str, Any] | None = None,
) -> list[str] | None:
    """Explicit registry patterns, else defaults from ``hunyuan_ms_variant``."""
    if not ver_config:
        return None
    for spec in (primary_spec, ver_config):
        if not isinstance(spec, dict):
            continue
        raw = spec.get("allow_patterns")
        if isinstance(raw, list) and raw:
            return _strip_generated_allow_patterns([str(p) for p in raw])
    variant = ver_config.get("hunyuan_ms_variant")
    if variant:
        return hunyuan_modelscope_allow_patterns(str(variant))
    return None


def assemble_hunyuan_modelscope_bundle(
    bundle_root: Path,
    variant: str,
) -> None:
    """Hoist ``transformer/<variant>/`` to diffusers ``transformer/`` layout."""
    root = Path(bundle_root)
    if not variant:
        raise RuntimeError("hunyuan_ms_variant is required for ModelScope HunyuanVideo assembly.")

    native = root / "transformer" / variant
    if not native.is_dir():
        raise RuntimeError(
            f"ModelScope HunyuanVideo bundle missing transformer/{variant}/ under {root}. "
            f"Check allow_patterns and repo_id Tencent-Hunyuan/HunyuanVideo-1.5."
        )

    flat = root / "transformer"
    staging = root / "_hunyuan_ms_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    for item in native.iterdir():
        dest = staging / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    if flat.exists():
        shutil.rmtree(flat)
    staging.rename(flat)

    # Drop leftover native variant subdirs if a full repo was downloaded.
    for sub in list(flat.iterdir()):
        if sub.is_dir() and sub.name in {
            "480p_t2v", "480p_i2v", "480p_i2v_step_distilled", "480p_i2v_distilled",
            "480p_t2v_distilled", "720p_t2v", "720p_i2v", "1080p_sr_distilled",
            "720p_sr_distilled", "720p_i2v_distilled_sparse",
        }:
            shutil.rmtree(sub)

    if not (flat / "config.json").is_file():
        raise RuntimeError(
            f"Assembled HunyuanVideo transformer/ missing config.json under {root} "
            f"(variant={variant!r})."
        )

    index_path = root / "model_index.json"
    if not index_path.is_file():
        index_path.write_text(
            json.dumps(
                {
                    "_class_name": "HunyuanVideo15Pipeline",
                    "_hunyuan_ms_variant": variant,
                    "_danqing_bundle_source": "modelscope",
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def ensure_hunyuan_ms_bundle_assembled(bundle_root: Path, variant: str) -> None:
    """Hoist native variant tree when needed; no-op if already assembled."""
    if is_hunyuan_ms_bundle_assembled(bundle_root):
        return
    assemble_hunyuan_modelscope_bundle(bundle_root, variant)
