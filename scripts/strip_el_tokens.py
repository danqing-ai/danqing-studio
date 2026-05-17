#!/usr/bin/env python3
"""Remove all --el-* CSS declarations and empty rules; migrate native token block."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STYLES = ROOT / "frontend" / "src" / "styles"
DQ_UI_STYLE = (ROOT / ".." / "dq-ui" / "packages" / "ui" / "src" / "style.css").resolve()
SKIP: set[str] = set()

TOKEN_RENAME = {
    "--el-text-color-primary": "--dq-label-primary",
    "--el-text-color-regular": "--dq-label-secondary",
    "--el-text-color-secondary": "--dq-label-tertiary",
    "--el-text-color-placeholder": "--dq-label-quaternary",
    "--el-text-color-disabled": "--dq-label-disabled",
    "--el-bg-color-page": "--dq-bg-page",
    "--el-bg-color": "--dq-bg-base",
    "--el-bg-color-overlay": "--dq-bg-elevated",
    "--el-fill-color-blank": "--dq-fill-control",
    "--el-fill-color": "--dq-fill-secondary",
    "--el-fill-color-light": "--dq-fill-tertiary",
    "--el-fill-color-lighter": "--dq-fill-quaternary",
    "--el-border-color": "--dq-border",
    "--el-border-color-light": "--dq-border",
    "--el-border-color-lighter": "--dq-border-subtle",
    "--el-color-primary": "--dq-accent",
    "--el-color-success": "--dq-success",
    "--el-color-warning": "--dq-warning",
    "--el-color-danger": "--dq-danger",
    "--el-color-info": "--dq-info",
}


def strip_el_lines(text: str) -> str:
    lines = []
    for line in text.splitlines(keepends=True):
        if re.match(r"\s*--el-", line):
            continue
        lines.append(line)
    return "".join(lines)


def remove_empty_rules(text: str) -> str:
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        m = re.search(r"([^{}]+)\{", text[i:], re.DOTALL)
        if not m:
            out.append(text[i:])
            break
        sel_start = i + m.start()
        out.append(text[i:sel_start])
        brace = i + m.end() - 1
        depth = 0
        j = brace
        while j < n:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    j += 1
                    break
            j += 1
        body = text[brace + 1 : j - 1]
        if body.strip():
            out.append(text[sel_start:j])
        i = j
    result = "".join(out)
    return re.sub(r"\n{3,}", "\n\n", result)


def rename_tokens(text: str) -> str:
    for old, new in TOKEN_RENAME.items():
        text = text.replace(old, new)
    return text


def _strip_var_el_fallbacks(text: str) -> str:
    return re.sub(r",\s*var\(--el-[^,]+,\s*([^)]+)\)", r", \1", text)


def main() -> int:
    bridge = STYLES / "dq-legacy-bridge.css"
    if bridge.exists():
        bridge.unlink()
        print("deleted dq-legacy-bridge.css")

    if DQ_UI_STYLE.is_file():
        orig = DQ_UI_STYLE.read_text(encoding="utf-8")
        text = _strip_var_el_fallbacks(orig)
        if text != orig:
            DQ_UI_STYLE.write_text(text)
            print(f"updated {DQ_UI_STYLE.name} (dq-ui)")

    for path in sorted(STYLES.glob("*.css")):
        if path.name in SKIP:
            continue
        orig = path.read_text(encoding="utf-8")
        text = strip_el_lines(orig)
        if path.name == "theme-apple-native.css":
            text = rename_tokens(text)
        text = remove_empty_rules(text)
        if text != orig:
            path.write_text(text)
            print(f"updated {path.name}")

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
