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

from backend.engine.pipelines.video_bundle_layout import (
    wan_flat_transformer_shards,
    wan_is_moe_bundle,
    wan_moe_expert_shards,
)


@dataclass(frozen=True)
class ComponentQuantPlan:
    """Optional TE/VAE (or second text encoder) derived quantization target."""

    subdir: str
    load_paths: tuple[Path, ...]
    output_dir: Path
    bits: int
    output_shard_prefix: str = "model"
    single_output_file: Path | None = None


@dataclass(frozen=True)
class ExpertQuantPlan:
    """Additional DiT shard group (e.g. Wan 14B low-noise MoE expert)."""

    subdir: str
    load_paths: tuple[Path, ...]
    output_dir: Path
    output_shard_prefix: str = "model"
    single_output_file: Path | None = None


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
    component_targets: tuple[ComponentQuantPlan, ...] = ()
    expert_quant_targets: tuple[ExpertQuantPlan, ...] = ()
    exclude_subdirs: tuple[str, ...] = ()


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
        plan = _plan_wan_dit_shards(from_root, to_root)
    elif layout_name == "dit_single_file":
        plan = _plan_dit_single_file(from_root, to_root, family)
    elif layout_name == "diffusers_transformer":
        plan = _plan_diffusers_transformer(from_root, to_root)
    else:
        raise RuntimeError(
            f"Unknown derived quantization layout {layout_name!r} for family={family!r}. "
            "Set quantization.layout on the derived version or register a family default."
        )

    return _attach_component_quant_targets(from_root, to_root, to_ver_config, plan)


def copy_component_companion_files(src_dir: Path, dst_dir: Path) -> None:
    """Copy config/tokenizer sidecars; quantized safetensors are written separately."""
    if not src_dir.is_dir():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in sorted(src_dir.iterdir()):
        if item.is_dir():
            companion_dst = dst_dir / item.name
            if companion_dst.exists():
                shutil.rmtree(companion_dst)
            shutil.copytree(item, companion_dst)
            continue
        if item.suffix == ".safetensors" or item.name == "model.safetensors.index.json":
            continue
        shutil.copy2(item, dst_dir / item.name)


def copy_non_quantized_bundle(from_root: Path, to_root: Path, plan: DerivedQuantPlan) -> None:
    """Copy companion artifacts after quantized weights are written."""
    to_root.mkdir(parents=True, exist_ok=True)

    if plan.copy_entire_bundle_except_inputs:
        _copy_tree_excluding(
            from_root,
            to_root,
            plan.load_paths,
            exclude_subdirs=plan.exclude_subdirs,
        )
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
    skip_dirs.update(plan.exclude_subdirs)

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


def _component_quant_bits(quant: dict[str, Any], component: str) -> int | None:
    block = quant.get(component)
    if not isinstance(block, dict):
        return None
    bits = block.get("bits")
    return int(bits) if bits in (4, 8) else None


def _safetensors_paths_under(dir_path: Path) -> tuple[Path, ...]:
    if not dir_path.is_dir():
        return ()
    single = dir_path / "model.safetensors"
    if single.is_file():
        return (single,)
    shards = tuple(sorted(dir_path.glob("*.safetensors")))
    return shards


def _attach_component_quant_targets(
    from_root: Path,
    to_root: Path,
    to_ver_config: dict[str, Any],
    plan: DerivedQuantPlan,
) -> DerivedQuantPlan:
    quant = to_ver_config.get("quantization") or {}
    targets: list[ComponentQuantPlan] = []
    for component in ("text_encoder", "text_encoder_2", "vae"):
        bits = _component_quant_bits(quant, component)
        if bits is None:
            continue
        src_dir = from_root / component
        load_paths = _safetensors_paths_under(src_dir)
        if not load_paths:
            raise RuntimeError(
                f"Registry declares {component} {bits}-bit derived quantization, "
                f"but no safetensors weights were found under {src_dir}."
            )
        single_file = load_paths[0] if len(load_paths) == 1 and load_paths[0].name == "model.safetensors" else None
        targets.append(
            ComponentQuantPlan(
                subdir=component,
                load_paths=load_paths,
                output_dir=to_root / component,
                bits=bits,
                single_output_file=(to_root / component / "model.safetensors") if single_file else None,
            )
        )

    if not targets:
        return plan

    quant_subdirs = {t.subdir for t in targets}
    expert_subdirs = {t.subdir for t in plan.expert_quant_targets}
    merged_exclude = tuple(sorted(set(plan.exclude_subdirs) | quant_subdirs | expert_subdirs))
    return DerivedQuantPlan(
        load_paths=plan.load_paths,
        output_dir=plan.output_dir,
        output_shard_prefix=plan.output_shard_prefix,
        single_output_file=plan.single_output_file,
        copy_subdirs=tuple(s for s in plan.copy_subdirs if s not in quant_subdirs),
        copy_root_except_globs=plan.copy_root_except_globs,
        copy_entire_bundle_except_inputs=plan.copy_entire_bundle_except_inputs,
        component_targets=tuple(targets),
        expert_quant_targets=plan.expert_quant_targets,
        exclude_subdirs=merged_exclude,
    )


def _copy_tree_excluding(
    from_root: Path,
    to_root: Path,
    exclude_files: tuple[Path, ...],
    *,
    exclude_subdirs: tuple[str, ...] = (),
) -> None:
    excluded = {p.resolve() for p in exclude_files}
    excluded_prefixes = tuple(from_root / name for name in exclude_subdirs)
    for src in sorted(from_root.rglob("*")):
        if not src.is_file():
            continue
        if src.resolve() in excluded:
            continue
        if any(src.is_relative_to(prefix) for prefix in excluded_prefixes if prefix.exists()):
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
    if wan_is_moe_bundle(from_root):
        return _plan_wan_moe_dit_shards(from_root, to_root)

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


def _plan_wan_moe_dit_shards(from_root: Path, to_root: Path) -> DerivedQuantPlan:
    high_shards = tuple(wan_moe_expert_shards(from_root, "high"))
    low_shards = tuple(wan_moe_expert_shards(from_root, "low"))
    if not high_shards or not low_shards:
        raise RuntimeError(
            f"Wan 14B MoE bundle under {from_root} is missing high_noise_model/ or "
            "low_noise_model/ safetensors shards."
        )

    return DerivedQuantPlan(
        load_paths=high_shards,
        output_dir=to_root / "high_noise_model",
        output_shard_prefix="model",
        single_output_file=None,
        copy_subdirs=(),
        copy_root_except_globs=(),
        copy_entire_bundle_except_inputs=True,
        expert_quant_targets=(
            ExpertQuantPlan(
                subdir="low_noise_model",
                load_paths=low_shards,
                output_dir=to_root / "low_noise_model",
            ),
        ),
        exclude_subdirs=("high_noise_model", "low_noise_model"),
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
