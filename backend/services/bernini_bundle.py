"""Assemble ByteDance Bernini-R diffusers bundles into Wan-family MoE layout."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

# ``bernini_variant`` → expert subdirs under downloaded diffusers tree.
_BERNINI_VARIANTS: dict[str, dict[str, object]] = {
    "bernini_r_14b": {
        "moe": True,
        "high_subdir": "transformer",
        "low_subdir": "transformer_2",
    },
    "bernini_r_1.3b": {
        "moe": False,
        "single_subdir": "transformer",
    },
}


def _collect_shards(src_dir: Path) -> list[Path]:
    if not src_dir.is_dir():
        return []
    shards = sorted(src_dir.glob("*.safetensors"))
    if shards:
        return shards
    return sorted(src_dir.glob("diffusion_pytorch_model*.safetensors"))


def _stage_shards(shards: list[Path], dest_dir: Path, *, config_src: Path | None = None) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if config_src is not None and (config_src / "config.json").is_file():
        cfg_dest = dest_dir / "config.json"
        if not cfg_dest.is_file():
            shutil.copy2(config_src / "config.json", cfg_dest)
    if len(shards) == 1:
        dest = dest_dir / shards[0].name
        if dest.is_file():
            return
        shutil.move(str(shards[0]), str(dest))
        return
    for shard in shards:
        dest = dest_dir / shard.name
        if dest.is_file():
            continue
        shutil.move(str(shard), str(dest))


def assemble_bernini_bundle(bundle_root: Path, variant: str) -> None:
    """Hoist Bernini-R DiT shards into Wan MoE dirs (14B) or flat transformer/ (1.3B)."""
    root = Path(bundle_root)
    if not variant:
        raise RuntimeError("bernini_variant is required for Bernini bundle assembly.")
    spec = _BERNINI_VARIANTS.get(str(variant))
    if spec is None:
        known = ", ".join(sorted(_BERNINI_VARIANTS))
        raise RuntimeError(f"Unknown bernini_variant {variant!r}. Supported: {known}.")

    if spec.get("moe"):
        high_dir = root / str(spec["high_subdir"])
        low_dir = root / str(spec["low_subdir"])
        high_shards = _collect_shards(high_dir)
        low_shards = _collect_shards(low_dir)
        if not high_shards or not low_shards:
            raise RuntimeError(
                f"Bernini MoE bundle missing transformer shards under {root} "
                f"(high={len(high_shards)}, low={len(low_shards)})."
            )
        _stage_shards(high_shards, root / "high_noise_model", config_src=high_dir)
        _stage_shards(low_shards, root / "low_noise_model", config_src=low_dir)
        dual = True
    else:
        src_dir = root / str(spec["single_subdir"])
        shards = _collect_shards(src_dir)
        if not shards:
            raise RuntimeError(f"Bernini bundle missing transformer/ weights under {root}.")
        flat = root / "transformer"
        if flat.exists() and flat.is_dir():
            shutil.rmtree(flat)
        _stage_shards(shards, flat, config_src=src_dir)
        dual = False

    index_path = root / "model_index.json"
    payload: dict[str, object] = {}
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "_bernini_variant": variant,
            "_danqing_bundle_source": "bernini_r",
            "dual_model": dual,
        }
    )
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
