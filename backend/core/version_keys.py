"""Canonical models_registry version keys for MLX / CUDA weight tiers.

Convention (machine ``versions.*`` keys only; display names stay in i18n):

- **Derived** (``source_type: derived``): ``int4``, ``int8``
- **Prequantized MLX download** (``source_type: prequantized``): ``mlx-q4``, ``mlx-q8`` only
- **MLX full bundle** (``source_type: full``): ``mlx-bf16``; optional ``mlx-q4`` / ``mlx-q8`` when upstream ships quantized full bundles
- **Suffix variants**: ``mlx-q4-dmd``, ``mlx-bf16-dmd``, …
- **CUDA / dual-platform full** (``source_type: full``): ``fp16``, ``bf16``, ``fp8`` (dtype — not ``original`` / ``quant``)
- **Same dtype, different resolution**: split into separate ``model_id`` rows (each with a single ``fp16`` / ``bf16`` version)

``local_path`` for prequantized MLX tiers uses ``{stem}-mlx-q*`` (legacy ``*-mlx-community-*bit`` names are forbidden).

Legacy keys (``mlx-4bit``, ``community-4bit``, ``mlx-int4`` derived, bare ``mlx`` for LTX q8,
``original``, ``quant``) resolve through :func:`canonical_version_key` / :func:`resolve_registry_version_key`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Static legacy → canonical (context-free renames).
_STATIC_ALIASES: dict[str, str] = {
    "mlx-3bit": "mlx-q4",
    "mlx-4bit": "mlx-q4",
    "mlx-5bit": "mlx-q4",
    "mlx-6bit": "mlx-q4",
    "mlx-8bit": "mlx-q8",
    "mlx-q3": "mlx-q4",
    "mlx-q5": "mlx-q4",
    "mlx-q6": "mlx-q4",
    "community-3bit": "mlx-q4",
    "community-4bit": "mlx-q4",
    "community-5bit": "mlx-q4",
    "community-6bit": "mlx-q4",
    "community-8bit": "mlx-q8",
    "mlx": "mlx-q8",
}

_ALLOWED_MLX_Q_BITS = frozenset({4, 8})

_MLX_Q_RE = re.compile(r"^mlx-q(\d+)(?:-[\w-]+)?$")
_MLX_BF16_RE = re.compile(r"^mlx-bf16(?:-[\w-]+)?$")

_FORBIDDEN_VAGUE_VERSION_KEYS = frozenset({"original", "quant"})

# Legacy API aliases → first matching key in registry ``versions`` (order matters).
_LEGACY_VERSION_RESOLVE_ORDER: tuple[str, ...] = (
    "fp16",
    "bf16",
    "fp8",
    "encoders",
    "int8",
)

_LEGACY_LOCAL_SUFFIXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"-mlx-community-(?:3|4|5|6|8)bit$"),
    re.compile(r"-community-(?:3|4|5|6|8)bit$"),
    re.compile(r"-themindstudio-4bit$"),
    re.compile(r"-mlx-int4$"),
    re.compile(r"-mlx-int8$"),
    re.compile(r"-original$"),
    re.compile(r"-distill-original$"),
    re.compile(r"-turbo-quant$"),
)

# Legacy avatar dirs used bf16/q4/q8 instead of mlx-bf16/mlx-q4/mlx-q8 in the suffix.
_LEGACY_AVATAR_DIR_NAMES: dict[str, str] = {
    "longcat-avatar-1.5-bf16-dmd": "longcat-avatar-1.5-mlx-bf16-dmd",
    "longcat-avatar-1.5-q8-dmd": "longcat-avatar-1.5-mlx-q8-dmd",
    "longcat-avatar-1.5-q4-dmd": "longcat-avatar-1.5-mlx-q4-dmd",
}

_SHARED_LOCAL_PATH_PREFIXES: tuple[str, ...] = (
    "models/Text/",
    "models/LLM/qwen3-vl-8b-instruct",
)


def local_path_dir_stem(local_path: str) -> str:
    """Basename of *local_path* with legacy MLX quant directory suffix removed."""
    name = Path(local_path).name
    for pat in _LEGACY_LOCAL_SUFFIXES:
        m = pat.search(name)
        if m:
            return name[: m.start()]
    return name


def canonical_local_path(local_path: str, version_key: str) -> str:
    """Return canonical install dir; only rewrites legacy MLX quant directory names."""
    text = str(local_path or "").strip()
    if not text:
        return text
    p = Path(text)
    if not p.parts or p.parts[0] != "models":
        return text.replace("\\", "/")
    normalized = text.replace("\\", "/")
    vk = str(version_key or "").strip()
    if vk and (p.name.endswith(f"-{vk}") or p.name == vk):
        return normalized
    if not is_legacy_quant_local_path(text):
        return normalized
    stem = local_path_dir_stem(text)
    return str(p.with_name(f"{stem}-{vk}")).replace("\\", "/")


def is_legacy_quant_local_path(local_path: str) -> bool:
    name = Path(local_path).name
    return any(pat.search(name) for pat in _LEGACY_LOCAL_SUFFIXES)


def is_canonical_local_path(local_path: str, version_key: str) -> bool:
    if not local_path or not version_key:
        return True
    p = Path(local_path)
    return p.name == f"{local_path_dir_stem(local_path)}-{version_key}"


def is_shared_bundle_local_path(local_path: str) -> bool:
    """True for cross-model text/LLM dependency install roots (no version suffix)."""
    normalized = str(local_path or "").replace("\\", "/")
    return any(normalized.startswith(prefix) for prefix in _SHARED_LOCAL_PATH_PREFIXES)


def canonical_primary_local_path(local_path: str, version_key: str) -> str:
    """Rewrite primary install dir to ``{stem}-{version_key}`` when needed."""
    text = str(local_path or "").strip()
    if not text or not version_key:
        return text.replace("\\", "/")

    legacy = canonical_local_path(text, version_key)
    if legacy != text.replace("\\", "/"):
        return legacy

    p = Path(text)
    normalized = text.replace("\\", "/")
    vk = str(version_key).strip()
    name = p.name

    if name in _LEGACY_AVATAR_DIR_NAMES:
        return str(p.with_name(_LEGACY_AVATAR_DIR_NAMES[name])).replace("\\", "/")
    if name.endswith(f"-{vk}"):
        return normalized
    if is_shared_bundle_local_path(text):
        return normalized

    return str(p.with_name(f"{name}-{vk}")).replace("\\", "/")


def is_forbidden_vague_version_key(version_key: str | None) -> bool:
    return str(version_key or "").strip().lower() in _FORBIDDEN_VAGUE_VERSION_KEYS


def resolve_full_bundle_version_key(versions: dict[str, Any]) -> str | None:
    """Return the registry key for the full-weight bundle (T5 fallback, shared deps)."""
    if not versions:
        return None
    for candidate in _LEGACY_VERSION_RESOLVE_ORDER:
        if candidate in versions and isinstance(versions[candidate], dict):
            return candidate
    for vk, vinfo in versions.items():
        if not isinstance(vinfo, dict):
            continue
        if str(vinfo.get("source_type") or "").lower() == "full" and not is_forbidden_vague_version_key(vk):
            return str(vk)
    return None


def canonical_version_key(
    version_key: str | None,
    *,
    version_entry: dict[str, Any] | None = None,
) -> str | None:
    """Map legacy version keys to the canonical registry key when possible."""
    if not version_key or not str(version_key).strip():
        return version_key
    key = str(version_key).strip()
    lower = key.lower()

    static = _STATIC_ALIASES.get(lower)
    if static:
        return static

    if lower in ("mlx-int4", "mlx-int8"):
        source_type = str((version_entry or {}).get("source_type") or "")
        if source_type == "derived":
            return "int4" if lower == "mlx-int4" else "int8"
        bits = _quant_bits(version_entry)
        if bits in _ALLOWED_MLX_Q_BITS:
            return f"mlx-q{bits}"
        if bits in (3, 5, 6):
            return "mlx-q4"
        return "mlx-q4" if lower == "mlx-int4" else "mlx-q8"

    if lower == "original":
        return "fp16"
    if lower == "quant":
        return "int8"
    if lower in ("distill-4step", "sr-1080p", "distill-sparse"):
        return "bf16"
    if lower in ("fp16-720p", "fp16-480p", "720p", "480p"):
        return "fp16"

    return key


def resolve_registry_version_key(
    versions: dict[str, Any],
    version_key: str | None,
) -> str | None:
    """Resolve *version_key* against registry ``versions`` (canonical + legacy alias)."""
    if not versions:
        return version_key
    if not version_key:
        for vk, vinfo in versions.items():
            if isinstance(vinfo, dict) and vinfo.get("default"):
                return str(vk)
        return version_key

    key = str(version_key).strip()
    if key in versions and isinstance(versions[key], dict):
        return key

    if is_forbidden_vague_version_key(key):
        resolved = resolve_full_bundle_version_key(versions)
        if resolved:
            return resolved

    canonical = canonical_version_key(key, version_entry=versions.get(key) if isinstance(versions.get(key), dict) else None)
    if canonical and canonical in versions and isinstance(versions[canonical], dict):
        return canonical

    for vk, vinfo in versions.items():
        if not isinstance(vinfo, dict):
            continue
        if canonical_version_key(vk, version_entry=vinfo) == canonical:
            return str(vk)

    return key


def is_valid_mlx_quant_version_key(version_key: str, *, source_type: str = "") -> bool:
    """True when *version_key* follows MLX quant naming rules for its source type."""
    key = str(version_key or "").strip().lower()
    if not key:
        return False
    st = str(source_type or "").strip().lower()
    if st == "derived":
        return key in ("int4", "int8")
    if _MLX_BF16_RE.match(key):
        return True
    q_match = _MLX_Q_RE.match(key)
    if q_match:
        return int(q_match.group(1)) in _ALLOWED_MLX_Q_BITS
    if st in ("prequantized", "full") and key.startswith("mlx-"):
        return False
    return True


def _quant_bits(version_entry: dict[str, Any] | None) -> int | None:
    if not version_entry:
        return None
    quant = version_entry.get("quantization")
    if not isinstance(quant, dict):
        return None
    bits = quant.get("bits")
    return int(bits) if isinstance(bits, int) else None


def is_quantized_registry_version(
    version_key: str,
    version_entry: dict[str, Any] | None = None,
) -> bool:
    """True when a registry version row denotes quantized weights (not dense fp16/bf16)."""
    vinfo = version_entry or {}
    st = str(vinfo.get("source_type") or "").lower()
    if st in ("derived", "prequantized"):
        return True
    key = str(version_key or "").strip().lower()
    if key in ("int4", "int8"):
        return True
    if _MLX_Q_RE.match(key):
        return True
    if _quant_bits(vinfo) is not None:
        return True
    return False
