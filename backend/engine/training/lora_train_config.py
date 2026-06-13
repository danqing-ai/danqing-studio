"""Load LoRA training overrides from JSON/YAML config files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_lora_train_config_file(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise RuntimeError(f"LoRA training config not found: {p}")
    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as e:
            raise RuntimeError(
                "YAML training config requires PyYAML (`pip install pyyaml`) or use a .json config"
            ) from e
        data = yaml.safe_load(text)
    else:
        raise RuntimeError(f"Unsupported training config format {suffix!r}; use .json or .yaml")
    if not isinstance(data, dict):
        raise RuntimeError(f"Training config root must be an object (got {type(data).__name__})")
    return data


def merge_config_into_request_dict(base: dict[str, Any], config_path: Path | str) -> dict[str, Any]:
    """CLI/API helper: file values fill gaps; explicit base keys win."""
    file_cfg = load_lora_train_config_file(config_path)
    return {**file_cfg, **base}
