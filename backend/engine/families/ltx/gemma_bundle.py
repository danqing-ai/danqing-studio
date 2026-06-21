"""LTX Gemma 3 text encoder bundle — registry local path + one-time HF cache migration."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

DEFAULT_GEMMA_HF_REPO = "mlx-community/gemma-3-12b-it-4bit"
DEFAULT_GEMMA_LOCAL_REL = "models/Text/gemma-3-12b-it-4bit"


def gemma_bundle_usable(root: Path) -> bool:
    """True when ``root`` looks like a complete mlx-lm Gemma bundle."""
    if not root.is_dir():
        return False
    config = root / "config.json"
    if not config.is_file():
        return False
    for path in root.rglob("*.safetensors"):
        try:
            if path.is_file():
                return True
        except OSError:
            continue
    return False


def _hf_hub_repo_dir(repo_id: str) -> Path:
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    safe = repo_id.replace("/", "--")
    return cache_root / "hub" / f"models--{safe}"


def _find_hf_hub_snapshot(repo_id: str) -> Path | None:
    snapshots = _hf_hub_repo_dir(repo_id) / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = sorted(
        (p for p in snapshots.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for snap in candidates:
        if gemma_bundle_usable(snap):
            return snap
    return None


def _materialize_hf_snapshot(snapshot: Path, target: Path) -> None:
    """Copy a hub snapshot tree, dereferencing blob symlinks into real files."""
    if target.is_dir():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    for item in snapshot.iterdir():
        dest = target / item.name
        src = item.resolve() if item.is_symlink() else item
        if src.is_file():
            shutil.copy2(src, dest)
        elif src.is_dir():
            shutil.copytree(src, dest)


def ensure_gemma_local_bundle(
    target: Path,
    *,
    hf_repo_id: str = DEFAULT_GEMMA_HF_REPO,
) -> Path:
    """Ensure Gemma weights exist under registry ``models/Text/…`` path.

    One-time migration from HuggingFace hub cache when present (offline, no network).
    Otherwise fail loud — install via model manager / batch install.
    """
    target = Path(target)
    if gemma_bundle_usable(target):
        return target

    snapshot = _find_hf_hub_snapshot(hf_repo_id)
    if snapshot is not None:
        _materialize_hf_snapshot(snapshot, target)
        if gemma_bundle_usable(target):
            return target

    raise RuntimeError(
        f"LTX 2.3 Gemma text encoder not found at {target}. "
        f"Install {hf_repo_id!r} to {DEFAULT_GEMMA_LOCAL_REL} "
        "(Settings → Models, or batch install for LTX)."
    )


def resolve_gemma_load_path(config: object, *, project_root: Path | None = None) -> Path:
    """Resolve absolute Gemma directory from ``LTXConfig`` registry injection."""
    local = str(getattr(config, "text_encoder_gemma_local", "") or "").strip()
    if not local:
        local = DEFAULT_GEMMA_LOCAL_REL
        if project_root is not None:
            local = str((project_root / local).resolve())
    path = Path(local)
    if not path.is_absolute() and project_root is not None:
        path = (project_root / path).resolve()
    hf_repo = str(getattr(config, "gemma_model_id", None) or DEFAULT_GEMMA_HF_REPO)
    return ensure_gemma_local_bundle(path, hf_repo_id=hf_repo)
