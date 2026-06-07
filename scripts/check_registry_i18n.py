#!/usr/bin/env python3
"""Lint models_registry.json user-facing i18n labels."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "default_config" / "models_registry.json"

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
REPO_HINT_RE = re.compile(
    r"MLX Community|MindCraft|themindstudio|社区预量化|Provider:|repo_id|"
    r"(?:魔搭|ModelScope|Hugging Face|\bHF\b)\s+\S+/\S+|"
    r"\b[a-z0-9][a-z0-9._-]*/[a-z0-9._-]*\d[a-z0-9._/-]*|"
    r"\b[A-Za-z0-9._-]+\.[A-Za-z0-9._-]+/[A-Za-z0-9]",
    re.IGNORECASE,
)
PLATFORM_HINT_RE = re.compile(r"魔搭|ModelScope|Hugging Face|\bHF\b", re.IGNORECASE)
COMMERCIAL_HINT_RE = re.compile(
    r"(?<![不])可商用|MIT 许可|commercial use allowed",
    re.IGNORECASE,
)


def _is_bilingual_pair(val: Any) -> bool:
    return (
        isinstance(val, dict)
        and isinstance(val.get("zh"), str)
        and val["zh"].strip() != ""
        and isinstance(val.get("en"), str)
        and val["en"].strip() != ""
    )


def _lint_user_text(text: str, ctx: str, failures: list[str]) -> None:
    if COMMERCIAL_HINT_RE.search(text):
        failures.append(f"{ctx} must not duplicate commercial_use_allowed badge text")
    if PLATFORM_HINT_RE.search(text):
        failures.append(f"{ctx} must not mention download platform (use source badge)")
    if REPO_HINT_RE.search(text):
        failures.append(f"{ctx} must not expose repo/community/provider details")


def lint_registry(data: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    models = data.get("models") or {}
    if not isinstance(models, dict):
        return ["models must be an object"]

    for model_id, model_entry in models.items():
        ctx = f"models.{model_id}"
        if not isinstance(model_entry, dict):
            failures.append(f"{ctx}: entry must be an object")
            continue

        name = model_entry.get("name")
        if not _is_bilingual_pair(name):
            failures.append(f"{ctx}.name must be {{zh, en}} with non-empty strings")
        elif isinstance(name, dict):
            for lang in ("zh", "en"):
                _lint_user_text(str(name[lang]), f"{ctx}.name.{lang}", failures)

        desc = model_entry.get("description")
        if not _is_bilingual_pair(desc):
            failures.append(f"{ctx}.description must be {{zh, en}} with non-empty strings")
        elif isinstance(desc, dict):
            en = str(desc.get("en") or "")
            if CJK_RE.search(en):
                failures.append(f"{ctx}.description.en must not contain CJK characters")
            for lang in ("zh", "en"):
                _lint_user_text(str(desc.get(lang) or ""), f"{ctx}.description.{lang}", failures)

        versions = model_entry.get("versions") or {}
        if not isinstance(versions, dict):
            continue
        for version_key, version_entry in versions.items():
            vctx = f"{ctx}.versions.{version_key}"
            if not isinstance(version_entry, dict):
                continue
            vn = version_entry.get("name")
            if isinstance(vn, str):
                failures.append(f"{vctx}.name must be bilingual {{zh, en}}, not a string")
                continue
            if not _is_bilingual_pair(vn):
                failures.append(f"{vctx}.name must be {{zh, en}} with non-empty strings")
                continue
            for lang in ("zh", "en"):
                _lint_user_text(str(vn.get(lang) or ""), f"{vctx}.name.{lang}", failures)

    return failures


def main() -> int:
    path = REGISTRY_PATH
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    data = json.loads(path.read_text(encoding="utf-8"))
    failures = lint_registry(data)
    if failures:
        print(f"Registry i18n check failed ({len(failures)} errors):")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("Registry i18n check OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
