"""Registry-driven LoRA adapter metadata (UI picklist, compose overrides, engine flags)."""
from __future__ import annotations

from typing import Any, Sequence

from backend.core.contracts import parse_model_version


def _entry_params(entry: Any) -> dict[str, Any]:
    params = getattr(entry, "parameters", None)
    if isinstance(params, dict):
        return params
    raw = getattr(entry, "raw", None) or {}
    legacy = raw.get("parameters")
    return legacy if isinstance(legacy, dict) else {}


def _catalog_lora_meta(entry: Any) -> dict[str, Any]:
    raw = getattr(entry, "raw", None) or {}
    metadata: dict[str, Any] = {}
    catalog = raw.get("catalog") if isinstance(raw.get("catalog"), dict) else {}
    if isinstance(catalog.get("metadata"), dict):
        metadata = catalog["metadata"]
    elif isinstance(raw.get("metadata"), dict):
        metadata = raw["metadata"]
    else:
        top = getattr(entry, "metadata", None)
        if isinstance(top, dict):
            metadata = top
    lora = metadata.get("lora")
    return lora if isinstance(lora, dict) else {}


def lora_video_step_distill(entry: Any) -> bool:
    """LoRA requests step-distill video schedule when merged (Lightning, etc.)."""
    params = _entry_params(entry)
    return bool(params.get("video_step_distill") or params.get("wan_lightning_distill"))


def lora_moe_shards(entry: Any) -> bool:
    """LoRA bundle uses separate high/low noise MoE safetensors."""
    params = _entry_params(entry)
    return bool(params.get("lora_moe_shards") or params.get("wan_lightning_distill"))


def lora_compose_overrides(entry: Any) -> dict[str, Any]:
    """Composer param overrides when this LoRA is selected (steps, guidance, shift, …)."""
    params = _entry_params(entry)
    raw = params.get("compose_overrides")
    if not isinstance(raw, dict) or not raw:
        return {}
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if value is not None:
            out[str(key)] = value
    return out


def lora_adapter_tags(entry: Any) -> list[str]:
    meta = _catalog_lora_meta(entry)
    tags = meta.get("tags")
    if not isinstance(tags, list):
        return []
    return [str(t).strip() for t in tags if str(t).strip()]


def lora_adapter_hint_key(entry: Any) -> str | None:
    meta = _catalog_lora_meta(entry)
    hint = meta.get("hint_key")
    text = str(hint).strip() if hint is not None else ""
    return text or None


def lora_base_model_key(entry: Any) -> str:
    raw = getattr(entry, "raw", None) or {}
    declared = getattr(entry, "base_model", None) or raw.get("base_model") or ""
    return str(declared).split(":", 1)[0].strip()


def lora_base_compatible(model_base_key: str, lora_entry: Any) -> bool:
    """Default LoRA ↔ base matching: exact registry ``base_model`` id."""
    lora_key = lora_base_model_key(lora_entry)
    model_key = (model_base_key or "").split(":", 1)[0].strip()
    if not model_key or not lora_key:
        return False
    return lora_key == model_key


def lora_adapter_picklist_row(
    entry: Any,
    *,
    lora_id: str,
    name: str,
    base_model: str,
    source: str = "registry",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "kind": "lora",
        "id": lora_id,
        "name": name,
        "base_model": base_model,
        "source": source,
        "tags": lora_adapter_tags(entry),
        "compose_overrides": lora_compose_overrides(entry),
    }
    hint = lora_adapter_hint_key(entry)
    if hint:
        row["hint_key"] = hint
    return row


def adapters_include_video_step_distill(adapters: Sequence[Any], registry: Any) -> bool:
    from backend.engine.common.bundle.lora_mlx import adapter_id_weight

    for item in adapters or ():
        lora_id, _ = adapter_id_weight(item)
        mid, _ = parse_model_version(lora_id)
        entry = registry.get(mid) if registry is not None else None
        if entry is not None and lora_video_step_distill(entry):
            return True
    return False
