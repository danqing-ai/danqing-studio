"""HeartMuLa bundle layout — official heartlib / ModelScope checkpoints."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Official heartlib layout: happy-new-year weights land in ``HeartMuLa-oss-3B/``.
MULA_DIR_NAMES = (
    "HeartMuLa-oss-3B",
    "HeartMuLa-oss-3B-happy-new-year",
)
CODEC_DIR_NAMES = (
    "HeartCodec-oss",
    "HeartCodec-oss-20260123",
)


@dataclass(frozen=True)
class HeartMuLaBundlePaths:
    """Resolved paths under registry ``local_path`` (bundle root)."""

    root: Path
    mula_torch: Path
    codec_torch: Path
    tokenizer: Path
    gen_config: Path


def _first_existing_dir(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def resolve_heartmula_bundle(bundle_root: Path) -> HeartMuLaBundlePaths:
    """Resolve paths for HeartMuLa-oss-3B-happy-new-year bundle.

    Expected layout (same as https://github.com/HeartMuLa/heartlib )::

        {bundle_root}/
          tokenizer.json          # HeartMuLa/HeartMuLaGen
          gen_config.json
          HeartMuLa-oss-3B/       # HeartMuLa/HeartMuLa-oss-3B-happy-new-year
          HeartCodec-oss/         # HeartMuLa/HeartCodec-oss-20260123
    """
    root = Path(bundle_root)
    mula_torch = _first_existing_dir(root, MULA_DIR_NAMES)
    if mula_torch is None:
        raise FileNotFoundError(
            f"HeartMuLa LM not found under {root}. "
            f"Expected one of: {', '.join(MULA_DIR_NAMES)}"
        )
    codec_torch = _first_existing_dir(root, CODEC_DIR_NAMES)
    if codec_torch is None:
        raise FileNotFoundError(
            f"HeartCodec not found under {root}. "
            f"Expected one of: {', '.join(CODEC_DIR_NAMES)} "
            "(download HeartMuLa/HeartCodec-oss-20260123)"
        )
    tokenizer = root / "tokenizer.json"
    if not tokenizer.is_file():
        raise FileNotFoundError(
            f"tokenizer.json not found under {root} "
            "(download HeartMuLa/HeartMuLaGen to bundle root)"
        )
    gen_config = root / "gen_config.json"
    if not gen_config.is_file():
        raise FileNotFoundError(
            f"gen_config.json not found under {root} "
            "(download HeartMuLa/HeartMuLaGen to bundle root)"
        )
    return HeartMuLaBundlePaths(
        root=root,
        mula_torch=mula_torch,
        codec_torch=codec_torch,
        tokenizer=tokenizer,
        gen_config=gen_config,
    )


MLX_WEIGHTS_NAME = "model.safetensors"


def mlx_weights_path(component_dir: Path) -> Path:
    """MLX weights written by install hook: ``<component>/mlx/model.safetensors``."""
    return component_dir / "mlx" / MLX_WEIGHTS_NAME


def mlx_weights_ready(bundle_root: Path) -> bool:
    """Bundle layout plus MLX caches from download-time conversion."""
    try:
        paths = resolve_heartmula_bundle(bundle_root)
    except FileNotFoundError:
        return False
    return (
        mlx_weights_path(paths.mula_torch).is_file()
        and mlx_weights_path(paths.codec_torch).is_file()
    )


def bundle_is_ready(bundle_root: Path) -> bool:
    return mlx_weights_ready(bundle_root)


def load_gen_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
