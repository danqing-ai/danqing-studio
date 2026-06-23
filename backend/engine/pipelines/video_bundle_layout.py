"""Video bundle directory layout — diffusers tree vs MLX-forge / dgrauet flat releases."""
from __future__ import annotations

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

    layout_handlers = {
        "ltx": _ltx_flat_transformer_sources,
        "wan": _wan_flat_transformer_sources,
    }
    handler = layout_handlers.get(family or "")
    if handler is not None:
        result = handler(bundle_root, model_id)
        if result is not None:
            return result

    return None, []


def _ltx_flat_transformer_sources(
    bundle_root: Path, model_id: str
) -> tuple[Path, list[Path]] | None:
    if not is_ltx_mlx_forge_flat_bundle(bundle_root):
        return None
    shards = ltx_flat_transformer_weight_files(bundle_root, model_id)
    return (bundle_root, shards) if shards else (bundle_root, [])


def _wan_flat_transformer_sources(
    bundle_root: Path, model_id: str
) -> tuple[Path, list[Path]] | None:
    del model_id
    shards = wan_flat_transformer_shards(bundle_root)
    if shards:
        return (bundle_root, shards)
    return None


def wan_is_moe_bundle(bundle_root: Path) -> bool:
    """True when bundle has Wan 14B high/low noise expert directories."""
    if not bundle_root.is_dir():
        return False
    return (
        (bundle_root / "high_noise_model").is_dir()
        and (bundle_root / "low_noise_model").is_dir()
    )


def wan_moe_expert_shards(bundle_root: Path, expert: str) -> list[Path]:
    """Return safetensors for one MoE expert (``high`` or ``low``)."""
    if expert not in {"high", "low"}:
        raise RuntimeError(f"wan_moe_expert_shards: expert must be 'high' or 'low', got {expert!r}")
    sub = "high_noise_model" if expert == "high" else "low_noise_model"
    expert_dir = bundle_root / sub
    if not expert_dir.is_dir():
        return []
    shards = sorted(expert_dir.glob("*.safetensors"))
    if shards:
        return shards
    pth = sorted(expert_dir.glob("*.pth"))
    if pth:
        return pth
    return sorted(expert_dir.glob("diffusion_pytorch_model*.safetensors"))


def wan_flat_transformer_shards(bundle_root: Path) -> list[Path]:
    """Original Wan bundle: ``diffusion_pytorch_model*.safetensors`` at bundle root."""
    if not bundle_root.is_dir():
        return []
    if wan_is_moe_bundle(bundle_root):
        return []
    shards = sorted(bundle_root.glob("diffusion_pytorch_model*.safetensors"))
    if shards:
        return shards
    pth = sorted(bundle_root.glob("diffusion_pytorch_model*.pth"))
    if pth:
        return pth
    turbo = sorted(bundle_root.glob("TurboWan*.pth"))
    if turbo:
        return turbo
    tp = bundle_root / "transformer"
    if _is_dir_nonempty(tp):
        inner = sorted(tp.glob("*.safetensors"))
        if inner:
            return inner
    return []


def ltx_flat_vae_decoder_file(bundle_root: Path) -> Path | None:
    p = bundle_root / "vae_decoder.safetensors"
    return p if p.is_file() else None


def hunyuan_vae_dir(bundle_root: Path) -> Path | None:
    p = bundle_root / "vae"
    return p if p.is_dir() else None


def resolve_hunyuan_sr_bundle(bundle_root: Path | None) -> Path | None:
    """SR weights may live in sibling ``*-1080p-sr`` bundle; caller passes explicit path via registry."""
    if bundle_root is None:
        return None
    sr = bundle_root.parent / "hunyuan-video-1.5-1080p-sr"
    return sr if sr.is_dir() else None
