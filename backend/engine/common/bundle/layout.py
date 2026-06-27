"""Diffusers-style bundle directory layout helpers (fail loud)."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def t5_encoder_bundle_paths(bundle_root: Path | None) -> tuple[str, str]:
    """Resolve T5 **weights** dir and **tokenizer** dir from an installed bundle.

    Flux.1: CLIP in ``text_encoder/``; T5-xxl in ``text_encoder_2/`` + ``tokenizer_2/``.
    Single-T5 bundles (video, etc.): ``text_encoder/`` + ``tokenizer/``.
    """
    if bundle_root is None:
        raise RuntimeError(
            "T5 text encoding requires an installed model bundle (registry versions.local_path). "
            "Refusing implicit Hugging Face hub download (google/t5-v1_1-xxl)."
        )
    te2 = bundle_root / "text_encoder_2"
    te1 = bundle_root / "text_encoder"
    enc_dir: Path | None = None
    tok_candidates: list[Path] = []

    if te2.is_dir() and any(te2.iterdir()):
        enc_dir = te2
        tok_candidates = [bundle_root / "tokenizer_2", te2 / "tokenizer"]
    elif te1.is_dir() and any(te1.iterdir()):
        enc_dir = te1
        tok_candidates = [bundle_root / "tokenizer", te1 / "tokenizer"]

    if enc_dir is None:
        raise RuntimeError(
            f"T5 text encoder directory missing: expected ``{te2}`` or ``{te1}``. "
            "Re-install or sync the model bundle."
        )

    tok_dir: Path | None = None
    for c in tok_candidates:
        if c.is_dir() and any(c.iterdir()):
            tok_dir = c
            break
    if tok_dir is None and (bundle_root / "tokenizer_config.json").is_file():
        tok_dir = bundle_root
    if tok_dir is None:
        raise RuntimeError(
            f"T5 tokenizer not found under {bundle_root}. Tried: "
            + ", ".join(str(c) for c in tok_candidates)
            + ". Re-install the upstream tokenizer assets."
        )

    return str(enc_dir), str(tok_dir)


def assert_media_bundle_ready(
    bundle_root: Path | None,
    *,
    family: str,
    model_id: str,
    registry_entry: Any | None = None,
    project_root: Path | None = None,
) -> None:
    """Fail loud when required bundle components are missing (image/video pipelines)."""
    if bundle_root is None:
        raise RuntimeError(
            f"Model {model_id!r} has no installed bundle; cannot run family={family!r}. "
            "Install the model via the registry or sync versions.local_path."
        )
    from backend.core.bundle_manifest import assert_bundle_ready_for_family

    assert_bundle_ready_for_family(
        bundle_root,
        family=family,
        model_id=model_id,
        registry_entry=registry_entry,
        project_root=project_root,
    )
