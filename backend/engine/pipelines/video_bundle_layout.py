"""Video bundle directory layout — diffusers tree vs MLX-forge / dgrauet flat releases."""
from __future__ import annotations

import re
from pathlib import Path


def _is_dir_nonempty(d: Path) -> bool:
    return d.is_dir() and any(d.iterdir())


def is_ltx_mlx_forge_flat_bundle(bundle_root: Path) -> bool:
    """True when bundle looks like ``mlx-forge`` / HuggingFace ``dgrauet/ltx-2.3-mlx*`` (no ``transformer/`` subdir)."""
    if not bundle_root.is_dir():
        return False
    if (bundle_root / "transformer").is_dir():
        return False
    return any(
        p.is_file() and p.name.startswith("transformer-") and p.suffix == ".safetensors"
        for p in bundle_root.iterdir()
    )


def ltx_flat_transformer_weight_files(bundle_root: Path, model_id: str) -> list[Path]:
    """Pick transformer shard(s) for an LTX flat bundle (single-file DiT checkpoints)."""
    mid = (model_id or "").lower()
    candidates: list[str] = []
    if "distilled" in mid:
        candidates = [
            "transformer-distilled-1.1.safetensors",
            "transformer-distilled.safetensors",
            "transformer.safetensors",
        ]
    elif "dev" in mid:
        candidates = ["transformer-dev.safetensors", "transformer.safetensors"]
    else:
        candidates = [
            "transformer-distilled-1.1.safetensors",
            "transformer-distilled.safetensors",
            "transformer-dev.safetensors",
            "transformer.safetensors",
        ]
    out: list[Path] = []
    for name in candidates:
        p = bundle_root / name
        if p.is_file():
            out.append(p)
    return out


def resolve_video_transformer_weight_sources(
    bundle_root: Path | None,
    family: str,
    model_id: str,
) -> tuple[Path | None, list[Path]]:
    """Resolve directory used for affine-quant metadata + explicit ``*.safetensors`` paths to load.

    Returns:
        ``(tensor_root_dir, shard_paths)`` where ``tensor_root_dir`` is passed to
        ``read_bundle_affine_bits_if_quantized`` (must be a directory containing shards).
    """
    if bundle_root is None or not bundle_root.is_dir():
        return None, []

    tp = bundle_root / "transformer"
    if _is_dir_nonempty(tp):
        shards = sorted(tp.glob("*.safetensors"))
        return (tp, shards) if shards else (tp, [])

    if family == "ltx" and is_ltx_mlx_forge_flat_bundle(bundle_root):
        shards = ltx_flat_transformer_weight_files(bundle_root, model_id)
        return (bundle_root, shards) if shards else (bundle_root, [])

    return None, []


def ltx_flat_vae_decoder_file(bundle_root: Path) -> Path | None:
    p = bundle_root / "vae_decoder.safetensors"
    return p if p.is_file() else None


def looks_like_mlx_forge_ltx_transformer_keys(weights: dict) -> bool:
    """Detect mlx-forge ``sanitize_transformer_key`` naming (needs restore before ``remap_ltx_weights``)."""
    for k in weights:
        if ".ff.proj_in." in k or ".ff.proj_out." in k:
            return True
        if ".audio_ff.proj_in." in k or ".audio_ff.proj_out." in k:
            return True
    return False


def max_remapped_ltx_block_index(remapped: dict) -> int:
    """Largest ``N`` in ``blocks.N.*`` keys, or ``-1`` if none."""
    mx = -1
    for k in remapped:
        m = re.match(r"^blocks\.(\d+)\.", k)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx
