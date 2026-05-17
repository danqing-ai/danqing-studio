#!/usr/bin/env python3
"""Sync Tauri desktop version from a release tag or env (tauri.conf.json + Cargo.toml)."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
TAURI_CONF = PROJECT_ROOT / "desktop" / "src-tauri" / "tauri.conf.json"
CARGO_TOML = PROJECT_ROOT / "desktop" / "src-tauri" / "Cargo.toml"

_VERSION_RE = re.compile(
    r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
_CARGO_VERSION_RE = re.compile(r'^(version\s*=\s*")[^"]+(")', re.MULTILINE)


def normalize_version(raw: str) -> str:
    version = raw.strip()
    if version.startswith("v") or version.startswith("V"):
        version = version[1:]
    if not _VERSION_RE.match(version):
        raise SystemExit(
            f"Invalid semver for desktop bundle: {raw!r}\n"
            "Expected forms like 1.2.3 or 1.2.3-beta.1"
        )
    return version


def resolve_version(explicit: str | None) -> str:
    if explicit:
        return normalize_version(explicit)
    env = os.environ.get("DANQING_DESKTOP_VERSION", "").strip()
    if env:
        return normalize_version(env)
    try:
        tag = subprocess.check_output(
            ["git", "describe", "--exact-match", "--tags", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise SystemExit(
            "No desktop version given. Pass VERSION, set DANQING_DESKTOP_VERSION, "
            "or run from an exact git tag (v*)."
        ) from None
    return normalize_version(tag)


def set_desktop_version(version: str) -> None:
    conf = json.loads(TAURI_CONF.read_text(encoding="utf-8"))
    conf["version"] = version
    TAURI_CONF.write_text(
        json.dumps(conf, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    cargo = CARGO_TOML.read_text(encoding="utf-8")
    updated, count = _CARGO_VERSION_RE.subn(rf"\g<1>{version}\2", cargo, count=1)
    if count != 1:
        raise SystemExit(f"Could not update package version in {CARGO_TOML}")
    CARGO_TOML.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "version",
        nargs="?",
        help="Release version (with or without leading v). "
        "Defaults to DANQING_DESKTOP_VERSION or exact git tag on HEAD.",
    )
    args = parser.parse_args()
    version = resolve_version(args.version)
    set_desktop_version(version)
    print(f"Desktop version -> {version}")
    print(f"  {TAURI_CONF.relative_to(PROJECT_ROOT)}")
    print(f"  {CARGO_TOML.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
