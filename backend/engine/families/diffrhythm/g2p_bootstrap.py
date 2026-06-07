"""
Resolve Amphion G2P assets for DiffRhythm 2 Chinese lyrics.

G2P is installed at ``{bundle_root}/g2p/`` via registry ``bundle_repos`` (ModelScope
weights + HuggingFace ``g2p/**`` follow-up). INT8/INT4 quant bundles reuse fp16 G2P when
the quant directory has no local copy.
"""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_G2P_PKG_DIR = "g2p"  # contains g2p_generation.py + subpackage g2p/
_G2P_ENTRY = "g2p_generation.py"
_G2P_REPO_ID = "ASLP-lab/DiffRhythm2"


def _g2p_package_dir(root: Path) -> Path:
    return root / _G2P_PKG_DIR


def _g2p_entry_exists(root: Path) -> bool:
    return (_g2p_package_dir(root) / _G2P_ENTRY).is_file()


def _fp16_g2p_sibling(bundle_root: Path) -> Path | None:
    name = bundle_root.name
    for suffix in ("-int8", "-int4"):
        if name.endswith(suffix):
            return bundle_root.parent / name[: -len(suffix)]
    return None


def resolve_g2p_bundle_root(bundle_root: Path | None) -> Path | None:
    """Return bundle directory that contains ``g2p/g2p_generation.py``."""
    if bundle_root is None:
        return None
    bundle = Path(bundle_root)
    if _g2p_entry_exists(bundle):
        return bundle
    sibling = _fp16_g2p_sibling(bundle)
    if sibling is not None and _g2p_entry_exists(sibling):
        return sibling
    return None


def bundle_g2p_ready(bundle_root: Path | None = None) -> bool:
    return resolve_g2p_bundle_root(bundle_root) is not None


def amphion_g2p_ready(bundle_root: Path | None = None) -> bool:
    """Deprecated alias for :func:`bundle_g2p_ready`."""
    return bundle_g2p_ready(bundle_root)


def _install_torch_free_chinese_g2p_shim() -> None:
    """Replace upstream ``chinese_model_g2p`` (PyTorch) before mandarin imports it."""
    from backend.engine.families.diffrhythm import chinese_poly_g2p

    sys.modules["g2p.g2p.chinese_model_g2p"] = chinese_poly_g2p


def install_bundle_g2p_path(bundle_root: Path | None = None) -> Path:
    resolved = resolve_g2p_bundle_root(bundle_root)
    if resolved is None:
        hint = f"{bundle_root}/g2p" if bundle_root is not None else "model bundle/g2p"
        raise RuntimeError(
            "DiffRhythm 2 Amphion g2p is not installed. "
            f"Re-install diffrhythm-v2 (fp16) or place upstream g2p/ under {hint}."
        )
    parent = str(resolved)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    _install_torch_free_chinese_g2p_shim()
    return _g2p_package_dir(resolved)


def install_amphion_g2p_path(bundle_root: Path | None = None) -> Path:
    """Deprecated alias for :func:`install_bundle_g2p_path`."""
    return install_bundle_g2p_path(bundle_root)


def ensure_bundle_g2p(bundle_root: Path) -> Path:
    """Ensure Chinese G2P assets exist; download into ``{bundle_root}/g2p/`` when missing."""
    resolved = resolve_g2p_bundle_root(bundle_root)
    if resolved is not None:
        return install_bundle_g2p_path(resolved)

    bundle = Path(bundle_root)
    logger.info(
        "Downloading DiffRhythm 2 g2p frontend to %s (HuggingFace, one-time fallback)",
        bundle,
    )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "DiffRhythm 2 Chinese lyrics require g2p assets under "
            f"{bundle / _G2P_PKG_DIR}. Re-install diffrhythm-v2 (fp16) from the model "
            "manager or install huggingface_hub for on-demand g2p fetch."
        ) from exc

    staging = bundle / "_g2p_dl"
    snapshot_download(
        repo_id=_G2P_REPO_ID,
        local_dir=str(staging),
        allow_patterns=[f"{_G2P_PKG_DIR}/**"],
    )
    src = staging / _G2P_PKG_DIR
    if not (src / _G2P_ENTRY).is_file():
        raise RuntimeError(
            f"DiffRhythm 2 g2p download completed but {src / _G2P_ENTRY} is missing. "
            "Check network / HuggingFace access."
        )

    dest = _g2p_package_dir(bundle)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(src), str(dest))
    shutil.rmtree(staging, ignore_errors=True)

    if not bundle_g2p_ready(bundle):
        raise RuntimeError(f"DiffRhythm 2 g2p bootstrap failed under {bundle}")

    logger.info("DiffRhythm 2 g2p frontend ready at %s", dest)
    return install_bundle_g2p_path(bundle)


def ensure_amphion_g2p(bundle_root: Path) -> Path:
    """Deprecated alias for :func:`ensure_bundle_g2p`."""
    return ensure_bundle_g2p(bundle_root)
