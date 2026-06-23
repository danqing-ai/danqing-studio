"""Wan video LoRA merge (Lightning MoE high/low + generic adapters)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import mlx.core as mx

from backend.core.contracts import parse_model_version
from backend.engine.common.bundle.lora_mlx import (
    adapter_id_weight,
    load_lora_flat_weights,
    merge_lora_adapters_common,
    orient_lora_pair,
    read_lora_config,
)
from backend.engine.contracts.pipeline_registry import local_bundle_root
from backend.engine.families.wan.lora_weights import remap_wan_lora_keys, wan_lora_param_key
from backend.engine.runtime._base import RuntimeContext

WAN_LIGHTNING_LORA_IDS = frozenset(
    {
        "wan2.2-i2v-lightning-lora",
        "wan2.2-t2v-lightning-lora",
    }
)

ExpertSide = Literal["high", "low"]


def is_wan_lightning_lora_id(lora_id: str) -> bool:
    mid, _ = parse_model_version(lora_id)
    return mid in WAN_LIGHTNING_LORA_IDS


def adapters_include_wan_lightning(adapters: Sequence[Any], registry: Any) -> bool:
    for item in adapters or ():
        lora_id, _ = adapter_id_weight(item)
        if is_wan_lightning_lora_id(lora_id):
            return True
        mid, _ = parse_model_version(lora_id)
        entry = registry.get(mid) if registry is not None else None
        if entry is not None:
            params = (entry.raw or {}).get("parameters") or {}
            if params.get("wan_lightning_distill"):
                return True
    return False


def resolve_wan_lora_weight_path(bundle_root: Path, side: ExpertSide | None) -> Path:
    """Resolve Lightning MoE LoRA shard or a flat LoRA file under ``bundle_root``."""
    root = Path(bundle_root)
    if not root.is_dir():
        raise RuntimeError(f"Wan LoRA bundle is not a directory: {root}")

    if side is not None:
        sub = "high_noise_model" if side == "high" else "low_noise_model"
        candidates: list[Path] = [
            root / sub / "diffusion_pytorch_model.safetensors",
            root / f"{sub}.safetensors",
        ]
        for nested in root.rglob("*"):
            if nested.is_dir() and nested.name == sub:
                candidates.append(nested / "diffusion_pytorch_model.safetensors")
                candidates.extend(sorted(nested.glob("*.safetensors")))
        for cand in candidates:
            if cand.is_file():
                return cand
            if cand.is_dir():
                shards = sorted(cand.glob("*.safetensors"))
                if shards:
                    return shards[0]
        raise RuntimeError(
            f"Wan Lightning LoRA missing {sub} weights under {root}. "
            "Expected high_noise_model/ and low_noise_model/ safetensors."
        )

    if root.is_file() and root.suffix == ".safetensors":
        return root
    flat = sorted(root.rglob("*.safetensors"))
    if not flat:
        raise RuntimeError(f"No .safetensors LoRA weights under {root}")
    if len(flat) == 1:
        return flat[0]
    raise RuntimeError(
        f"Wan LoRA bundle {root} has multiple safetensors; use a Lightning MoE layout "
        "or a single-file LoRA directory."
    )


def _apply_wan_lora_groups(
    inner: Any,
    *,
    groups: dict[str, tuple[Any, Any, float]],
    lora_id: str,
    strength: float,
    ctx: RuntimeContext,
    config_alpha: float | None,
) -> int:
    if not hasattr(inner, "_param_map"):
        raise RuntimeError("Wan LoRA merge requires TransformerBase with ``_param_map``.")
    param_map = inner._param_map
    applied = 0
    for module_name, (down, up, alpha) in groups.items():
        if module_name.endswith(".delta"):
            wkey = wan_lora_param_key(module_name[: -len(".delta")])
            if wkey not in param_map:
                continue
            param = param_map[wkey]
            scaled = float(strength) * down.astype(mx.float32)
            param[:] = (param.astype(mx.float32) + scaled).astype(param.dtype)
            applied += 1
            continue
        wkey = wan_lora_param_key(module_name)
        if wkey not in param_map:
            continue
        param = param_map[wkey]
        out_d, in_d = int(param.shape[0]), int(param.shape[1])
        d_orient, u_orient, rank = orient_lora_pair(
            down, up, out_d=out_d, in_d=in_d, lora_id=lora_id, wkey=wkey
        )
        if rank <= 0:
            raise RuntimeError(f"Wan LoRA {lora_id!r}: invalid rank for {wkey}.")
        eff_alpha = float(config_alpha) if config_alpha is not None else float(alpha)
        scale = (eff_alpha / float(rank)) * float(strength)
        delta = mx.matmul(u_orient.astype(mx.float32), d_orient.astype(mx.float32))
        param[:] = (param.astype(mx.float32) + scale * delta).astype(param.dtype)
        applied += 1
    return applied


def merge_wan_lora_into_expert(
    expert: Any,
    *,
    side: ExpertSide | None,
    lora_id: str,
    strength: float,
    bundle_root: Path,
    ctx: RuntimeContext,
    on_log: Callable[[str, str], None] | None = None,
) -> None:
    shard = resolve_wan_lora_weight_path(bundle_root, side)
    weights = load_lora_flat_weights(shard, ctx)
    lora_config = read_lora_config(shard if shard.is_file() else shard.parent)
    config_alpha = lora_config.get("lora_alpha", lora_config.get("alpha"))
    if config_alpha is not None:
        config_alpha = float(config_alpha)

    inner = getattr(expert, "_inner", expert)
    groups = remap_wan_lora_keys(weights)
    applied = _apply_wan_lora_groups(
        inner,
        groups=groups,
        lora_id=lora_id,
        strength=strength,
        ctx=ctx,
        config_alpha=config_alpha,
    )
    if applied == 0:
        raise RuntimeError(
            f"Wan LoRA {lora_id!r}: no tensors matched expert side={side!r} "
            f"(groups={len(groups)})."
        )
    if on_log:
        on_log(
            "info",
            f"wan lora merged source={lora_id} side={side or 'flat'} "
            f"strength={strength} tensors={applied}",
        )
    ctx.eval(*[p for _, p in inner.parameters()])
    merged_ids = list(getattr(expert, "_dq_wan_lora_merged_ids", []) or [])
    if lora_id not in merged_ids:
        merged_ids.append(lora_id)
    setattr(expert, "_dq_wan_lora_merged_ids", merged_ids)


def merge_wan_lora_adapters(
    model: Any,
    adapters: Sequence[Any],
    *,
    base_model_id: str,
    project_root: Path,
    registry: Any,
    ctx: RuntimeContext,
    on_log: Callable[[str, str], None] | None = None,
) -> None:
    if not adapters:
        return

    moe_apply = getattr(model, "apply_lora_adapters", None)
    if callable(moe_apply):
        moe_apply(
            adapters=adapters,
            base_model_id=base_model_id,
            project_root=project_root,
            registry=registry,
            ctx=ctx,
            on_log=on_log,
        )
        return

    merge_lora_adapters_common(
        model=getattr(model, "_inner", model),
        adapters=adapters,
        base_model_id=base_model_id,
        project_root=project_root,
        registry=registry,
        ctx=ctx,
        family_name="Wan",
        remap_groups=remap_wan_lora_keys,
        param_key_for_module=wan_lora_param_key,
        on_log=on_log,
    )


def resolve_wan_lora_bundle(
    lora_id: str,
    *,
    base_model_id: str,
    project_root: Path,
    registry: Any,
) -> tuple[str, Path, bool]:
    """Return ``(mid, bundle_path, is_lightning_moe)`` for one adapter id."""
    mid, ver = parse_model_version(lora_id)
    entry = registry.require(mid)
    raw = entry.raw or {}
    if str(raw.get("category") or "") != "loras":
        raise RuntimeError(f"Adapter {lora_id!r} is not a registry LoRA (category={raw.get('category')!r}).")
    declared_base = str(raw.get("base_model") or "").strip()
    base_key = base_model_id.split(":", 1)[0].strip()
    if declared_base and declared_base.split(":", 1)[0] != base_key:
        raise RuntimeError(
            f"LoRA {mid!r} is scoped to base_model={declared_base!r}, "
            f"but the video request uses {base_key!r}."
        )
    bundle = local_bundle_root(project_root, entry, ver or None)
    lightning = is_wan_lightning_lora_id(lora_id) or bool(
        ((raw.get("parameters") or {}).get("wan_lightning_distill"))
    )
    return mid, bundle, lightning
