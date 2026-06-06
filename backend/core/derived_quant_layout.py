"""Registry-driven layout for local MLX derived quantization (int4/int8).

Model ``family`` (from registry) selects a default layout; derived versions may
override via ``quantization.layout``. No media-type branching.
"""
from __future__ import annotations

import fnmatch
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from backend.engine.pipelines.video_bundle_layout import wan_flat_transformer_shards


@dataclass(frozen=True)
class DerivedQuantPlan:
    """Where to read dense weights, write quantized shards, and copy the rest."""

    load_paths: tuple[Path, ...]
    output_dir: Path
    output_shard_prefix: str
    single_output_file: Path | None
    copy_subdirs: tuple[str, ...]
    copy_root_except_globs: tuple[str, ...]
    copy_entire_bundle_except_inputs: bool = False


_FAMILY_DEFAULT_LAYOUT: dict[str, str] = {
    "flux1": "diffusers_transformer",
    "flux2": "diffusers_transformer",
    "z_image": "diffusers_transformer",
    "qwen_image": "diffusers_transformer",
    "fibo": "diffusers_transformer",
    "seedvr2": "diffusers_transformer",
    "hunyuan": "diffusers_transformer",
    "ltx": "diffusers_transformer",
    "longcat": "diffusers_transformer",
    "wan": "wan_dit_shards",
    "diffrhythm": "dit_single_file",
    "ace_step": "dit_single_file",
}


def resolve_derived_quant_layout(
    *,
    family: str,
    from_root: Path,
    to_root: Path,
    to_ver_config: dict[str, Any],
) -> DerivedQuantPlan:
    quant = to_ver_config.get("quantization") or {}
    layout_name = str(quant.get("layout") or _FAMILY_DEFAULT_LAYOUT.get(family) or "diffusers_transformer")

    if layout_name == "wan_dit_shards":
        return _plan_wan_dit_shards(from_root, to_root)
    if layout_name == "dit_single_file":
        return _plan_dit_single_file(from_root, to_root, family)
    if layout_name == "diffusers_transformer":
        return _plan_diffusers_transformer(from_root, to_root)

    raise RuntimeError(
        f"Unknown derived quantization layout {layout_name!r} for family={family!r}. "
        "Set quantization.layout on the derived version or register a family default."
    )


def copy_non_quantized_bundle(from_root: Path, to_root: Path, plan: DerivedQuantPlan) -> None:
    """Copy companion artifacts after quantized weights are written."""
    to_root.mkdir(parents=True, exist_ok=True)

    if plan.copy_entire_bundle_except_inputs:
        _copy_tree_excluding(from_root, to_root, plan.load_paths)
        return

    for subdir in plan.copy_subdirs:
        src = from_root / subdir
        if not src.is_dir():
            continue
        dst = to_root / subdir
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    if not plan.copy_root_except_globs:
        return

    skip_dirs = set(plan.copy_subdirs)
    skip_dirs.add("transformer")

    for item in sorted(from_root.iterdir()):
        if item.is_dir():
            if item.name in skip_dirs:
                continue
            dst = to_root / item.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(item, dst)
            continue
        if _matches_any(item.name, plan.copy_root_except_globs):
            continue
        if item.suffix == ".safetensors" and item.resolve() in {p.resolve() for p in plan.load_paths}:
            continue
        shutil.copy2(item, to_root / item.name)


def _copy_tree_excluding(from_root: Path, to_root: Path, exclude_files: tuple[Path, ...]) -> None:
    excluded = {p.resolve() for p in exclude_files}
    for src in sorted(from_root.rglob("*")):
        if not src.is_file():
            continue
        if src.resolve() in excluded:
            continue
        rel = src.relative_to(from_root)
        dest = to_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def _plan_diffusers_transformer(from_root: Path, to_root: Path) -> DerivedQuantPlan:
    transformer_dir = from_root / "transformer"
    if transformer_dir.is_dir():
        load_paths = tuple(sorted(transformer_dir.glob("*.safetensors")))
    else:
        load_paths = tuple(sorted(from_root.glob("*.safetensors")))

    if not load_paths:
        raise RuntimeError(f"No safetensors weights found under {from_root}")

    return DerivedQuantPlan(
        load_paths=load_paths,
        output_dir=to_root / "transformer",
        output_shard_prefix="model",
        single_output_file=None,
        copy_subdirs=("vae", "text_encoder", "tokenizer", "text_encoder_2", "tokenizer_2"),
        copy_root_except_globs=(),
    )


def _plan_wan_dit_shards(from_root: Path, to_root: Path) -> DerivedQuantPlan:
    load_paths = tuple(wan_flat_transformer_shards(from_root))
    if not load_paths:
        raise RuntimeError(
            f"No Wan DiT safetensors (diffusion_pytorch_model*.safetensors) under {from_root}"
        )

    return DerivedQuantPlan(
        load_paths=load_paths,
        output_dir=to_root / "transformer",
        output_shard_prefix="model",
        single_output_file=None,
        copy_subdirs=(),
        copy_root_except_globs=("diffusion_pytorch_model*.safetensors",),
    )


def _plan_dit_single_file(from_root: Path, to_root: Path, family: str) -> DerivedQuantPlan:
    if family == "diffrhythm":
        from backend.engine.families.diffrhythm.generation import resolve_dit_bundle

        dit_root = resolve_dit_bundle(from_root)
    elif family == "ace_step":
        from backend.engine.families.ace_step.generation import resolve_dit_bundle

        dit_root = resolve_dit_bundle(from_root)
    else:
        dit_root = from_root if (from_root / "model.safetensors").is_file() else from_root

    src_file = dit_root / "model.safetensors"
    if not src_file.is_file():
        raise RuntimeError(f"No DiT checkpoint model.safetensors under {from_root}")

    rel_dir = dit_root.relative_to(from_root) if dit_root != from_root else Path(".")
    out_dir = to_root / rel_dir
    out_file = out_dir / "model.safetensors"

    return DerivedQuantPlan(
        load_paths=(src_file,),
        output_dir=out_dir,
        output_shard_prefix="model",
        single_output_file=out_file,
        copy_subdirs=(),
        copy_root_except_globs=(),
        copy_entire_bundle_except_inputs=True,
    )
