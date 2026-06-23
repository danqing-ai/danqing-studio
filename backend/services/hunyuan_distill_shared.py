"""Link Hunyuan shared encoders (VAE + transformer config) into a distill install directory."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

_HUNYUAN_DISTILL_BASE: dict[str, tuple[str, tuple[str, ...]]] = {
    "t2v_480p": ("hunyuan-video-1.5-shared", ("encoders",)),
}

_SHARED_DIRS = ("vae",)
_SHARED_FILES = ("model_index.json",)
_TRANSFORMER_CONFIG = "transformer/config.json"


def _resolve_shared_source(
    *,
    base_model_id: str,
    version_keys: tuple[str, ...],
    resolve_local_path: Callable[[str, str], Path],
) -> Path | None:
    for version_key in version_keys:
        try:
            local_path = resolve_local_path(base_model_id, version_key)
        except (KeyError, ValueError):
            continue
        root = Path(local_path)
        if not root.is_dir():
            continue
        vae = root / "vae"
        cfg = root / "transformer" / "config.json"
        if vae.is_dir() and cfg.is_file():
            return root
    return None


def _link_or_skip(dest: Path, src: Path) -> None:
    if dest.exists() or dest.is_symlink():
        try:
            if dest.resolve() == src.resolve():
                return
        except OSError:
            pass
        if dest.is_dir() and not dest.is_symlink():
            return
        dest.unlink(missing_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.symlink_to(src, target_is_directory=src.is_dir())


def link_hunyuan_distill_shared_assets(
    *,
    distill_root: Path,
    variant: str,
    resolve_local_path: Callable[[str, str], Path],
) -> None:
    """Symlink VAE + transformer config from shared encoders bundle into ``distill_root``."""
    root = Path(distill_root)
    if not root.is_dir():
        raise RuntimeError(f"Hunyuan distill bundle root not found: {root}")

    mapping = _HUNYUAN_DISTILL_BASE.get(str(variant))
    if mapping is None:
        known = ", ".join(sorted(_HUNYUAN_DISTILL_BASE))
        raise RuntimeError(f"Unknown hunyuan_distill_variant {variant!r} (known: {known}).")

    base_model_id, version_keys = mapping
    source_root = _resolve_shared_source(
        base_model_id=base_model_id,
        version_keys=version_keys,
        resolve_local_path=resolve_local_path,
    )
    if source_root is None:
        raise RuntimeError(
            f"Hunyuan distill requires shared encoders from {base_model_id!r} "
            f"(install version {' or '.join(version_keys)} first)."
        )

    for name in _SHARED_DIRS:
        src = source_root / name
        if src.is_dir():
            _link_or_skip(root / name, src)

    for name in _SHARED_FILES:
        src = source_root / name
        if src.is_file():
            _link_or_skip(root / name, src)

    cfg_src = source_root / _TRANSFORMER_CONFIG
    if cfg_src.is_file():
        _link_or_skip(root / _TRANSFORMER_CONFIG, cfg_src)

    vae = root / "vae"
    cfg = root / "transformer" / "config.json"
    if not vae.is_dir() or not cfg.is_file():
        missing = []
        if not vae.is_dir():
            missing.append("vae/")
        if not cfg.is_file():
            missing.append("transformer/config.json")
        raise RuntimeError(
            f"Hunyuan distill bundle still missing shared assets after link from {source_root}: "
            + ", ".join(missing)
        )
