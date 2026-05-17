#!/usr/bin/env python3
"""Detect stale Element-era class names and dq-ui selector mismatches."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend" / "src"
DQ_UI = (ROOT / ".." / "dq-ui" / "packages").resolve()

# (pattern, hint) — applied to Studio styles + dq-ui source
STYLE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\.dq-select-dropdown\b"), "use `.dq-select__content` / `.dq-select__option`"),
    (re.compile(r"\.dq-dropdown-menu__item\b"), "use `.dq-dropdown-item`"),
    (re.compile(r"\.dq-input-number__increase\b"), "use `.dq-input-number__btn`"),
    (re.compile(r"\.dq-input-number__decrease\b"), "use `.dq-input-number__btn`"),
    (re.compile(r"\.dq-tag[^{]*\.is-plain\b"), "use `.dq-tag--plain` (DqTag effect=plain)"),
]

VUE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'\bv-if="motion"\b'), 'use `v-if="motion"` → `v-if="divided"` on DqDropdownItem'),
    (re.compile(r'\bv-if=\'motion\'\b'), "use `v-if='motion'` → `v-if='divided'`"),
]


def _scan_file(path: Path, label: str, rules: list[tuple[re.Pattern[str], str]], failures: list[str]) -> None:
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for pattern, hint in rules:
            if pattern.search(line):
                try:
                    rel = path.relative_to(ROOT).as_posix()
                except ValueError:
                    rel = path.as_posix()
                failures.append(f"{label}/{rel}:{i}: {hint}")


def main() -> int:
    failures: list[str] = []

    style_paths: list[Path] = []
    studio_styles = FRONTEND / "styles"
    if studio_styles.is_dir():
        style_paths.extend(studio_styles.glob("*.css"))
    if DQ_UI.is_dir():
        for pkg in ("ui", "shell", "tokens"):
            pkg_src = DQ_UI / pkg / "src"
            if pkg_src.is_dir():
                style_paths.extend(pkg_src.rglob("*.css"))

    for path in style_paths:
        if "node_modules" in path.parts:
            continue
        _scan_file(path, "styles", STYLE_RULES, failures)

    vue_roots = [FRONTEND]
    if DQ_UI.is_dir():
        vue_roots.append(DQ_UI / "ui" / "src")
        vue_roots.append(DQ_UI / "shell" / "src")

    for root in vue_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.vue"):
            if "node_modules" in path.parts:
                continue
            _scan_file(path, "vue", VUE_RULES, failures)

    if failures:
        print(f"UI compat check failed ({len(failures)}):")
        for f in failures[:50]:
            print(f"  - {f}")
        if len(failures) > 50:
            print(f"  … and {len(failures) - 50} more")
        return 1
    print("UI compat check OK (Studio + dq-ui)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
