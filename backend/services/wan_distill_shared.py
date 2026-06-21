"""Link Wan base encoders (T5/VAE/tokenizer) into a distill install directory."""
from __future__ import annotations

import fnmatch
from collections.abc import Callable
from pathlib import Path

_WAN_DISTILL_BASE: dict[str, tuple[str, tuple[str, ...]]] = {
    "i2v_720p": ("wan-2.2-i2v-14b", ("shared", "original")),
    "i2v_fp8": ("wan-2.2-i2v-14b", ("shared", "original")),
    "t2v_fp8": ("wan-2.2-t2v-14b", ("shared", "original")),
}

_SHARED_PATTERNS = (
    "google/**",
    "configuration.json",
    "models_t5*.pth",
    "Wan2.1_VAE.pth",
)


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
        t5_hits = list(root.glob("models_t5*.pth"))
        if t5_hits and t5_hits[0].stat().st_size >= 1024 ** 3:
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


def link_wan_distill_shared_assets(
    *,
    distill_root: Path,
    variant: str,
    resolve_local_path: Callable[[str, str], Path],
) -> None:
    """Symlink T5/VAE/google from an installed Wan base bundle into ``distill_root``."""
    root = Path(distill_root)
    if not root.is_dir():
        raise RuntimeError(f"Wan distill bundle root not found: {root}")

    mapping = _WAN_DISTILL_BASE.get(str(variant))
    if mapping is None:
        known = ", ".join(sorted(_WAN_DISTILL_BASE))
        raise RuntimeError(f"Unknown wan_distill_variant {variant!r} (known: {known}).")

    base_model_id, version_keys = mapping
    source_root = _resolve_shared_source(
        base_model_id=base_model_id,
        version_keys=version_keys,
        resolve_local_path=resolve_local_path,
    )
    if source_root is None:
        raise RuntimeError(
            f"Wan distill requires Wan base encoders from {base_model_id!r} "
            f"(install version {' or '.join(version_keys)} first)."
        )

    for pattern in _SHARED_PATTERNS:
        if pattern.endswith("/**"):
            sub_name = pattern[:-3].rstrip("/")
            src = source_root / sub_name
            if src.is_dir():
                _link_or_skip(root / sub_name, src)
            continue
        if any(ch in pattern for ch in "*?[]"):
            for src in source_root.glob("*"):
                if src.is_file() and fnmatch.fnmatch(src.name, pattern):
                    _link_or_skip(root / src.name, src)
            continue
        src = source_root / pattern
        if src.is_file():
            _link_or_skip(root / pattern, src)

    t5 = next((p for p in root.glob("models_t5*.pth") if p.stat().st_size >= 1024 ** 3), None)
    vae = root / "Wan2.1_VAE.pth"
    google = root / "google"
    if t5 is None or not vae.is_file() or not google.is_dir():
        missing = []
        if t5 is None:
            missing.append("models_t5*.pth")
        if not vae.is_file():
            missing.append("Wan2.1_VAE.pth")
        if not google.is_dir():
            missing.append("google/**")
        raise RuntimeError(
            f"Wan distill bundle still missing shared assets after link from {source_root}: "
            + ", ".join(missing)
        )
