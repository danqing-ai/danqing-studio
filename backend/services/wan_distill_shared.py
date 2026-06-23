"""Link Wan shared encoders into distill / turbo / Bernini bundles; write DiT ``config.json``."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

WAN_SHARED_MODEL_ID = "wan-2.2-14b-shared"

_T5_PTH = "models_t5_umt5-xxl-enc-bf16.pth"
_VAE_PTH = "Wan2.1_VAE.pth"

# Official Wan-AI bundle layout (T2V / I2V 14B share the same encoder files).
_SHARED_FILES = (_T5_PTH, _VAE_PTH, "configuration.json")
_SHARED_DIRS = ("google",)

_WAN_TRANSFORMER_CONFIGS: dict[str, dict[str, Any]] = {
    "t2v_14b": {
        "_class_name": "WanModel",
        "dim": 5120,
        "eps": 1e-06,
        "ffn_dim": 13824,
        "freq_dim": 256,
        "in_dim": 16,
        "model_type": "t2v",
        "num_heads": 40,
        "num_layers": 40,
        "out_dim": 16,
        "text_len": 512,
        "expand_timesteps": False,
    },
    "i2v_14b": {
        "_class_name": "WanModel",
        "dim": 5120,
        "eps": 1e-06,
        "ffn_dim": 13824,
        "freq_dim": 256,
        "in_dim": 36,
        "model_type": "i2v",
        "num_heads": 40,
        "num_layers": 40,
        "out_dim": 16,
        "text_len": 512,
        "expand_timesteps": True,
    },
    "t2v_1.3b": {
        "_class_name": "WanModel",
        "dim": 1536,
        "eps": 1e-06,
        "ffn_dim": 8960,
        "freq_dim": 256,
        "in_dim": 16,
        "model_type": "t2v",
        "num_heads": 12,
        "num_layers": 30,
        "out_dim": 16,
        "text_len": 512,
        "expand_timesteps": False,
    },
}

_WAN_VARIANT_CONFIG_KEY: dict[str, str] = {
    "t2v_fp8": "t2v_14b",
    "i2v_720p": "i2v_14b",
    "i2v_fp8": "i2v_14b",
    "turbo_i2v_720p": "i2v_14b",
    "turbo_i2v_720p_quant": "i2v_14b",
    "turbo_t2v_480p_14b": "t2v_14b",
    "turbo_t2v_480p_14b_quant": "t2v_14b",
    "turbo_t2v_720p_14b": "t2v_14b",
    "turbo_t2v_720p_14b_quant": "t2v_14b",
    "turbo_t2v_480p_1.3b": "t2v_1.3b",
    "turbo_t2v_480p_1.3b_quant": "t2v_1.3b",
}


def _config_key_for_variant(variant: str) -> str:
    key = _WAN_VARIANT_CONFIG_KEY.get(str(variant))
    if key is None:
        known = ", ".join(sorted(_WAN_VARIANT_CONFIG_KEY))
        raise RuntimeError(f"Unknown wan_distill_variant {variant!r} (known: {known}).")
    return key


def _shared_encoders_root(resolve_local_path: Callable[[str, str], Path]) -> Path:
    root = Path(resolve_local_path(WAN_SHARED_MODEL_ID, "encoders"))
    if not root.is_dir():
        raise RuntimeError(
            f"Install {WAN_SHARED_MODEL_ID!r} version encoders first (missing {root})."
        )
    t5 = root / _T5_PTH
    if not t5.is_file() or t5.stat().st_size < 1024 ** 3:
        raise RuntimeError(f"Wan shared encoders incomplete: missing or invalid {_T5_PTH} under {root}.")
    for name in _SHARED_FILES:
        if name == _T5_PTH:
            continue
        if not (root / name).is_file():
            raise RuntimeError(f"Wan shared encoders incomplete: missing {name} under {root}.")
    for name in _SHARED_DIRS:
        if not (root / name).is_dir():
            raise RuntimeError(f"Wan shared encoders incomplete: missing {name}/ under {root}.")
    return root


def _link(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    dest.symlink_to(src, target_is_directory=src.is_dir())


def _write_transformer_config(bundle_root: Path, config_key: str) -> None:
    payload = _WAN_TRANSFORMER_CONFIGS[config_key]
    (Path(bundle_root) / "config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def link_wan_distill_shared_assets(
    *,
    distill_root: Path,
    variant: str,
    resolve_local_path: Callable[[str, str], Path],
) -> None:
    """Symlink UMT5/VAE/tokenizer from ``wan-2.2-14b-shared``; write bundle ``config.json``."""
    root = Path(distill_root)
    if not root.is_dir():
        raise RuntimeError(f"Wan distill bundle root not found: {root}")

    shared = _shared_encoders_root(resolve_local_path)
    config_key = _config_key_for_variant(variant)

    for name in _SHARED_FILES:
        _link(shared / name, root / name)
    for name in _SHARED_DIRS:
        _link(shared / name, root / name)

    _write_transformer_config(root, config_key)
