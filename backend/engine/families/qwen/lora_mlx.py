"""Qwen Image — MLX-only LoRA merge into loaded DiT linear weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import mlx.core as mx

from backend.core.contracts import parse_model_version
from backend.engine.common.pipeline_registry import local_bundle_root
from backend.engine.families.qwen.weights_mlx import remap_qwen_lora_keys
from backend.engine.runtime._base import RuntimeContext


def _adapter_id_weight(item: Any) -> tuple[str, float]:
    if isinstance(item, dict):
        return str(item["id"]), float(item.get("weight", 1.0))
    return str(item.id), float(item.weight)


def _load_lora_flat_weights(bundle: Path, ctx: RuntimeContext) -> dict[str, Any]:
    if bundle.is_file():
        if bundle.suffix != ".safetensors":
            raise RuntimeError(
                f"LoRA path must be a .safetensors file or a directory containing weights: {bundle}"
            )
        return dict(ctx.load_weights(str(bundle)))
    if bundle.is_dir():
        files = sorted(bundle.rglob("*.safetensors"))
        if not files:
            raise RuntimeError(f"No .safetensors files under LoRA directory {bundle}")
        out: dict[str, Any] = {}
        for fp in files:
            out.update(ctx.load_weights(str(fp)))
        return out
    raise RuntimeError(f"LoRA path is not a file or directory: {bundle}")


def _orient_lora_pair(
    down: mx.array,
    up: mx.array,
    *,
    out_d: int,
    in_d: int,
    lora_id: str,
    wkey: str,
) -> tuple[mx.array, mx.array, int]:
    """Return (down, up) with shapes [rank, in_d] and [out_d, rank]."""
    d, u = down, up

    if d.shape[1] == in_d and u.shape[0] == out_d and u.shape[1] == d.shape[0]:
        rank = int(d.shape[0])
        return d, u, rank
    if d.shape[0] == in_d and u.shape[0] == out_d and u.shape[1] == d.shape[1]:
        rank = int(d.shape[1])
        return mx.transpose(d, (1, 0)), u, rank
    if d.shape[1] == in_d and u.shape[1] == out_d and u.shape[0] == d.shape[0]:
        rank = int(d.shape[0])
        return d, mx.transpose(u, (1, 0)), rank
    if d.shape[0] == in_d and u.shape[1] == out_d and u.shape[0] == d.shape[1]:
        rank = int(d.shape[1])
        return mx.transpose(d, (1, 0)), mx.transpose(u, (1, 0)), rank

    raise RuntimeError(
        f"LoRA {lora_id!r} tensor shape mismatch for {wkey}: model ({out_d}, {in_d}), "
        f"lora_down {tuple(d.shape)}, lora_up {tuple(u.shape)}."
    )


def merge_qwen_image_lora_adapters(
    model: Any,
    adapters: Sequence[Any],
    *,
    base_model_id: str,
    project_root: Path,
    registry: Any,
    ctx: RuntimeContext,
    on_log: Callable[[str, str], None] | None = None,
) -> None:
    """Apply one or more registry LoRA bundles in order (weights merged in-place on ``model``)."""
    if not adapters:
        return
    if not hasattr(model, "_param_map"):
        raise RuntimeError("Qwen Image LoRA merge requires a QwenImageTransformer with ``_param_map``.")

    for item in adapters:
        lora_id, strength = _adapter_id_weight(item)
        mid, ver = parse_model_version(lora_id)
        try:
            entry = registry.require(mid)
        except KeyError as e:
            raise RuntimeError(
                f"Unknown LoRA adapter {lora_id!r}: add it to config/models_registry.json (category=loras) "
                "and install weights to the declared local_path."
            ) from e

        raw = entry.raw or {}
        if str(raw.get("category") or "") != "loras":
            raise RuntimeError(
                f"Adapter id {lora_id!r} refers to registry model {mid!r}, which is not a LoRA "
                f"(category={raw.get('category')!r})."
            )

        declared_base = str(raw.get("base_model") or "").strip()
        declared_key = declared_base.split(":", 1)[0].strip() if declared_base else ""
        request_key = base_model_id.split(":", 1)[0].strip() if base_model_id else ""
        if declared_key and declared_key != request_key:
            raise RuntimeError(
                f"LoRA {mid!r} is scoped to base_model={declared_base!r}, "
                f"but the image request uses {base_model_id!r}."
            )
        bundle = local_bundle_root(project_root, entry, ver or None)
        if bundle is None:
            raise RuntimeError(
                f"LoRA {lora_id!r} is not installed on disk (missing registry versions.local_path "
                f"for version {ver or 'default'})."
            )
        log_label = mid

        weights = _load_lora_flat_weights(bundle, ctx)
        groups = remap_qwen_lora_keys(weights)
        if not groups:
            raise RuntimeError(
                f"LoRA {lora_id!r}: after key remap no (lora_down, lora_up) pairs were found — "
                "unsupported layout or empty file."
            )

        applied = 0
        for module_name, (down, up, alpha) in groups.items():
            wkey = f"dit.{module_name}.weight"
            if wkey not in model._param_map:
                continue
            param = model._param_map[wkey]
            out_d, in_d = int(param.shape[0]), int(param.shape[1])
            d_orient, u_orient, rank = _orient_lora_pair(
                down, up, out_d=out_d, in_d=in_d, lora_id=lora_id, wkey=wkey
            )
            if rank <= 0:
                raise RuntimeError(f"LoRA {lora_id!r}: invalid rank for {wkey}.")
            scale = (float(alpha) / float(rank)) * float(strength)
            delta = mx.matmul(u_orient.astype(mx.float32), d_orient.astype(mx.float32))
            updated = param.astype(mx.float32) + (scale * delta)
            new_p = updated.astype(param.dtype)
            param[:] = new_p
            applied += 1

        if applied == 0:
            raise RuntimeError(
                f"LoRA {lora_id!r}: remap produced {len(groups)} module group(s), but none matched "
                "this Qwen Image transformer (wrong base_model / key layout)."
            )
        if on_log:
            on_log("info", f"lora merged source={log_label!s} strength={strength} tensors={applied}")

    ctx.eval(*[t for _, t in model.parameters()])
