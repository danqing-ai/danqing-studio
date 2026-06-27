"""Build ``FamilySpec`` from catalog v3 ``families`` block (single source for engine)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.catalog.loader import load_catalog_json, schema_version
from backend.engine.protocols.plugin import CfgMode, FamilySpec, LatentLayout, ParadigmKind

_VALID_PARADIGMS: frozenset[str] = frozenset(
    {"diffusion", "flow_matching", "block_ar", "two_stage", "job"}
)
_VALID_LAYOUTS: frozenset[str] = frozenset(
    {"nchw", "packed_seq", "qwen_grid", "video_5d", "audio_1d", "pixel_patch"}
)
_VALID_CFG: frozenset[str] = frozenset({"fused", "batched", "dual", "none"})
_VALID_MEDIA: frozenset[str] = frozenset({"image", "video", "audio"})


def _default_registry_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    workspace = repo_root / "config" / "models_registry.json"
    if workspace.is_file():
        return workspace
    factory = repo_root / "default_config" / "models_registry.json"
    if factory.is_file():
        return factory
    try:
        from backend.utils.path_utils import PathResolver

        return PathResolver().get_models_registry_path()
    except RuntimeError:
        return factory


@lru_cache(maxsize=1)
def load_families_block(registry_path: str | None = None) -> dict[str, Any]:
    """Return ``families`` object from on-disk catalog (v3). Fail loud if missing."""
    path = Path(registry_path) if registry_path else _default_registry_path()
    data = load_catalog_json(path)
    if schema_version(data) < 3:
        raise RuntimeError(
            f"{path}: schema_version must be >= 3 for catalog-driven FamilySpec "
            "(run make sync-models-registry)."
        )
    families = data.get("families")
    if not isinstance(families, dict) or not families:
        raise RuntimeError(f"{path}: catalog 'families' block is required for engine v3")
    return families


def clear_families_cache() -> None:
    load_families_block.cache_clear()


def _primary_media(record: dict[str, Any], *, family_id: str) -> str:
    media = record.get("media")
    if isinstance(media, list) and media:
        primary = str(media[0]).strip()
        if primary in _VALID_MEDIA:
            return primary
    if isinstance(media, str) and media.strip() in _VALID_MEDIA:
        return media.strip()
    raise RuntimeError(
        f"catalog families.{family_id}.media must be a non-empty list of image|video|audio"
    )


def family_spec_from_catalog_record(
    family_id: str,
    record: dict[str, Any],
    *,
    config: Any | None = None,
    media_override: str | None = None,
) -> FamilySpec:
    """Project one catalog family row → ``FamilySpec``."""
    if not isinstance(record, dict):
        raise RuntimeError(f"catalog families.{family_id!r} must be an object")

    paradigm_raw = str(record.get("paradigm") or "diffusion")
    if paradigm_raw not in _VALID_PARADIGMS:
        raise RuntimeError(
            f"catalog families.{family_id}.paradigm={paradigm_raw!r} is invalid"
        )
    paradigm: ParadigmKind = paradigm_raw  # type: ignore[assignment]

    cfg_raw = str(record.get("cfg_mode") or "dual")
    if cfg_raw not in _VALID_CFG:
        raise RuntimeError(f"catalog families.{family_id}.cfg_mode={cfg_raw!r} is invalid")
    cfg_mode: CfgMode = cfg_raw  # type: ignore[assignment]

    media = media_override or _primary_media(record, family_id=family_id)
    if media not in _VALID_MEDIA:
        raise RuntimeError(f"catalog families.{family_id}: invalid media {media!r}")

    layout_raw = str(record.get("latent_layout") or ("audio_1d" if media == "audio" else "nchw"))
    if layout_raw not in _VALID_LAYOUTS:
        raise RuntimeError(
            f"catalog families.{family_id}.latent_layout={layout_raw!r} is invalid"
        )
    latent_layout: LatentLayout = layout_raw  # type: ignore[assignment]

    hooks_raw = record.get("hooks") or []
    hooks: frozenset[str] = frozenset()
    if isinstance(hooks_raw, list):
        hooks = frozenset(str(h) for h in hooks_raw if str(h).strip())

    backends_raw = record.get("backends") or ["mlx"]
    backends: tuple[str, ...] = ("mlx",)
    if isinstance(backends_raw, list) and backends_raw:
        backends = tuple(str(b) for b in backends_raw)

    latent_channels = int(record.get("latent_channels") or 16)
    vae_scale = int(record.get("vae_scale") or 8)
    if config is not None:
        latent_channels = int(getattr(config, "in_channels", latent_channels) or latent_channels)
        vae_scale = int(getattr(config, "vae_scale", vae_scale) or vae_scale)

    supports_guidance = record.get("supports_guidance")
    if supports_guidance is None:
        supports_guidance = cfg_mode != "none"
    else:
        supports_guidance = bool(supports_guidance)

    step_profile = record.get("step_kwargs_profile")
    if step_profile is not None:
        step_profile = str(step_profile).strip() or None

    scheduler = str(record.get("default_scheduler") or "flow_match_euler")

    return FamilySpec(
        family_id=family_id,
        media=media,  # type: ignore[arg-type]
        paradigm=paradigm,
        latent_layout=latent_layout,
        latent_channels=latent_channels,
        vae_scale=vae_scale,
        cfg_mode=cfg_mode,
        supports_guidance=supports_guidance,
        default_scheduler=scheduler,
        step_kwargs_profile=step_profile,
        hooks=hooks,
        backends=backends,
    )


def family_spec_from_model_config(
    family_id: str,
    config: Any,
    *,
    media: str = "image",
) -> FamilySpec:
    """Derive ``FamilySpec`` from ``model_configs`` only (v2→v3 migration seeding)."""
    from backend.engine.protocols.plugin import FamilySpec as _FS

    latent_layout: LatentLayout = "nchw"
    if media == "audio":
        latent_layout = "audio_1d"
    elif getattr(config, "latent_noise_packed", False):
        latent_layout = "packed_seq"
    elif getattr(config, "encoder_step_kwargs", None) == "qwen_image":
        latent_layout = "qwen_grid"
    elif family_id == "hidream_o1":
        latent_layout = "pixel_patch"

    cfg_mode: CfgMode = "dual"
    if getattr(config, "use_mlx_cfg_fusion", False):
        cfg_mode = "fused"
    if not getattr(config, "supports_guidance", True):
        cfg_mode = "none"

    paradigm: ParadigmKind = "diffusion"
    if family_id == "ace_step":
        paradigm = "flow_matching"
    elif family_id == "diffrhythm":
        paradigm = "block_ar"
    elif family_id == "seedvr2":
        paradigm = "job"
    elif family_id == "ltx" or getattr(config, "video_pipeline_shape", None) == "family_generator":
        paradigm = "two_stage"

    hooks: set[str] = set()
    if family_id in ("flux1", "flux2", "z_image", "qwen_image"):
        hooks.add("lora_merge")

    return _FS(
        family_id=family_id,
        media=media,  # type: ignore[arg-type]
        paradigm=paradigm,
        latent_layout=latent_layout,
        latent_channels=int(getattr(config, "in_channels", 16) or 16),
        vae_scale=int(getattr(config, "vae_scale", 8) or 8),
        cfg_mode=cfg_mode,
        supports_guidance=bool(getattr(config, "supports_guidance", True)),
        default_scheduler="flow_match_euler",
        step_kwargs_profile=family_id if family_id in ("flux2", "qwen_image") else None,
        hooks=frozenset(hooks),
    )


def family_spec_from_catalog(
    family_id: str,
    *,
    config: Any | None = None,
    media: str | None = None,
    registry_path: str | None = None,
) -> FamilySpec:
    """Load ``FamilySpec`` for *family_id* from catalog ``families`` block."""
    families = load_families_block(registry_path)
    record = families.get(family_id)
    if not isinstance(record, dict):
        raise RuntimeError(
            f"catalog families.{family_id!r} is missing; add a row to models_registry.json "
            "or run make sync-models-registry after model_configs migration."
        )
    return family_spec_from_catalog_record(
        family_id,
        record,
        config=config,
        media_override=media,
    )
