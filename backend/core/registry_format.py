"""models_registry.json v2 解析（供 ModelRegistry / Settings）。"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Tuple


def parse_i18n_string_pair(
    name_val: Any,
    desc_val: Any,
    key: str,
) -> Tuple[str, str, str, str]:
    """v2 注册表：name/description 为 {zh,en}；否则仅作占位字符串。返回 (name_zh, name_en, desc_zh, desc_en)。"""
    if isinstance(name_val, dict) and "zh" in name_val:
        nz = str(name_val.get("zh") or key)
        ne = str(name_val.get("en") or nz)
    else:
        nz = str(name_val or key)
        ne = nz
    if isinstance(desc_val, dict) and "zh" in desc_val:
        dz = str(desc_val.get("zh") or "")
        de = str(desc_val.get("en") or dz)
    else:
        dz = str(desc_val or "")
        de = dz
    return nz, ne, dz, de


def api_action_frozenset(actions: Any, *, media: str) -> FrozenSet[str]:
    """调度器 / IImageEngine.supports 使用的 API 级动作名。"""
    if not isinstance(actions, dict):
        return frozenset()
    if media == "video":
        s: set[str] = set()
        if actions.get("create") is not None:
            s.add("generate")
        if actions.get("animate") is not None:
            s.add("edit")
        return frozenset(s)
    s = set()
    if actions.get("create") is not None:
        s.add("generate")
    if any(actions.get(k) is not None for k in ("rewrite", "retouch", "extend")):
        s.add("edit")
    if actions.get("upscale") is not None:
        s.add("upscale")
    return frozenset(s)


def media_from_record(raw: Dict[str, Any]) -> str:
    m = raw.get("media")
    if m in ("image", "video"):
        return str(m)
    cat = str(raw.get("category") or "")
    return "video" if cat == "video_models" else "image"


def typed_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """为缺少 type 的 parameter 条目补上简单 schema（幂等）。"""
    out: Dict[str, Any] = {}
    for key, spec in parameters.items():
        if not isinstance(spec, dict):
            out[key] = spec
            continue
        if "type" in spec:
            out[key] = dict(spec)
            continue
        if isinstance(spec.get("default"), bool) or key.endswith("_support"):
            out[key] = {**spec, "type": "bool"}
            continue
        if "options" in spec and isinstance(spec["options"], list):
            out[key] = {**spec, "type": "enum"}
            continue
        if "min" in spec and "max" in spec:
            d = spec.get("default")
            t = "int" if isinstance(d, int) and not isinstance(d, bool) else "float"
            out[key] = {**spec, "type": t}
            continue
        out[key] = {**spec, "type": "object"}
    return out
