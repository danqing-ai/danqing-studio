#!/usr/bin/env python3
"""One-shot / idempotent: migrate Studio theme CSS from Element Plus selectors & tokens to dq-ui."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STYLE_DIR = ROOT / "frontend" / "src" / "styles"
SKIP = {"dq-legacy-bridge.css"}

VAR_MAP = {
    "var(--el-color-primary)": "var(--dq-accent)",
    "var(--el-color-primary-light-3)": "var(--dq-accent-muted)",
    "var(--el-color-success)": "var(--dq-success)",
    "var(--el-color-warning)": "var(--dq-warning)",
    "var(--el-color-danger)": "var(--dq-danger)",
    "var(--el-color-info)": "var(--dq-info)",
    "var(--el-color-white)": "var(--dq-color-white)",
    "var(--el-bg-color-page)": "var(--dq-bg-page)",
    "var(--el-bg-color)": "var(--dq-bg-base)",
    "var(--el-bg-color-overlay)": "var(--dq-bg-elevated)",
    "var(--el-fill-color-blank)": "var(--dq-fill-control)",
    "var(--el-fill-color-light)": "var(--dq-fill-tertiary)",
    "var(--el-fill-color-lighter)": "var(--dq-fill-quaternary)",
    "var(--el-fill-color)": "var(--dq-fill-secondary)",
    "var(--el-fill-color-dark)": "var(--dq-fill-dim)",
    "var(--el-fill-color-darker)": "var(--dq-fill-dimmer)",
    "var(--el-text-color-primary)": "var(--dq-label-primary)",
    "var(--el-text-color-regular)": "var(--dq-label-secondary)",
    "var(--el-text-color-secondary)": "var(--dq-label-tertiary)",
    "var(--el-text-color-placeholder)": "var(--dq-label-quaternary)",
    "var(--el-text-color-disabled)": "var(--dq-label-disabled)",
    "var(--el-border-color)": "var(--dq-border)",
    "var(--el-border-color-lighter)": "var(--dq-border-subtle)",
    "var(--el-border-color-light)": "var(--dq-border-subtle)",
    "var(--el-border-color-dark)": "var(--dq-border-strong)",
    "var(--el-border-color-darker)": "var(--dq-border-strong)",
    "var(--el-font-family)": "var(--dq-font-sans)",
    "var(--el-border-radius-base)": "var(--dq-radius-control)",
    "var(--el-border-radius-small)": "var(--dq-radius-control-sm)",
    "var(--el-border-radius-round)": "var(--dq-radius-pill)",
    "var(--el-box-shadow-lighter)": "var(--dq-shadow-sm)",
    "var(--el-box-shadow-light)": "var(--dq-shadow-md)",
    "var(--el-box-shadow)": "var(--dq-shadow-lg)",
    "var(--el-mask-color)": "var(--dq-mask)",
    "var(--el-menu-hover-bg-color)": "var(--dq-fill-tertiary)",
    "var(--el-color-primary-dark-2)": "#0077ed",
    "var(--el-color-primary-light-9)": "var(--dq-accent-tint)",
    "var(--el-color-success-light-9)": "color-mix(in srgb, var(--dq-success) 12%, transparent)",
    "var(--el-color-success-light-5)": "var(--dq-success)",
    "var(--el-color-success-light-7)": "var(--dq-success)",
    "var(--el-color-warning-light-9)": "color-mix(in srgb, var(--dq-warning) 12%, transparent)",
    "var(--el-color-warning-light-7)": "var(--dq-warning)",
    "var(--el-color-black)": "#000000",
    "var(--el-border-color-extra-light)": "var(--dq-border-subtle)",
    "var(--el-card-bg-color)": "var(--dq-bg-elevated)",
}

SELECTOR_REPLACEMENTS = [
    (re.compile(r":is\(\s*\.dq-([\w-]+)\s*,\s*\.el-[\w-]+\s*\)"), r".dq-\1"),
    (re.compile(r":is\(\s*\.el-[\w-]+\s*,\s*\.dq-([\w-]+)\s*\)"), r".dq-\1"),
    (re.compile(r"\.el-button\b"), ".dq-btn"),
    (re.compile(r"\.el-input-number\b"), ".dq-input-number"),
    (re.compile(r"\.el-input\b"), ".dq-input"),
    (re.compile(r"\.el-select\b"), ".dq-select"),
    (re.compile(r"\.el-slider\b"), ".dq-slider"),
    (re.compile(r"\.el-switch\b"), ".dq-switch"),
    (re.compile(r"\.el-checkbox\b"), ".dq-checkbox"),
    (re.compile(r"\.el-card\b"), ".dq-surface-card"),
    (re.compile(r"\.el-alert\b"), ".dq-alert"),
    (re.compile(r"\.el-collapse\b"), ".dq-collapse"),
    (re.compile(r"\.el-drawer\b"), ".dq-drawer"),
    (re.compile(r"\.el-row\b"), ".dq-row"),
    (re.compile(r"\.el-col\b"), ".dq-col"),
    (re.compile(r"\.el-dropdown\b"), ".dq-dropdown"),
    (re.compile(r"\.el-segmented\b"), ".dq-segmented"),
    (re.compile(r"\.el-icon\b"), ".dq-icon"),
    (re.compile(r"\.el-progress\b"), ".dq-progress"),
    (re.compile(r"\.el-tag\b"), ".dq-tag"),
    (re.compile(r"\.el-divider\b"), ".dq-vdivider"),
]

DEAD_LINE_PATTERNS = [
    re.compile(r"\.el-menu\b"),
    re.compile(r"\.el-tabs"),
    re.compile(r"\.el-table"),
    re.compile(r"\.el-form"),
    re.compile(r"\.el-select__"),
    re.compile(r"\.el-input__"),
    re.compile(r"\.el-textarea\b"),
    re.compile(r"\.el-card__"),
    re.compile(r"\.el-alert__"),
    re.compile(r"\.el-collapse-item__"),
    re.compile(r"\.el-drawer__"),
    re.compile(r"\.el-popper\b"),
    re.compile(r"\.el-dropdown"),
    re.compile(r"\.el-segmented"),
    re.compile(r"\.el-radio"),
    re.compile(r"\.el-divider"),
    re.compile(r"settings-ep-tabs"),
    re.compile(r"settings-ep-media-segmented"),
]



def strip_dead_rules(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    skip_depth = 0
    brace_depth = 0
    for line in lines:
        stripped = line.strip()
        if skip_depth > 0:
            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0:
                skip_depth = 0
                brace_depth = 0
            continue
        if stripped and not stripped.startswith("/*") and not stripped.startswith("@") and "{" in line:
            if any(p.search(line) for p in DEAD_LINE_PATTERNS):
                skip_depth = 1
                brace_depth = line.count("{") - line.count("}")
                if brace_depth <= 0 and "}" in line:
                    skip_depth = 0
                continue
        out.append(line)
    return "".join(out)


def migrate_text(text: str) -> str:
    for old, new in sorted(VAR_MAP.items(), key=lambda x: -len(x[0])):
        text = text.replace(old, new)
    for pattern, repl in SELECTOR_REPLACEMENTS:
        text = pattern.sub(repl, text)
    text = strip_dead_rules(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def main() -> int:
    changed = 0
    for path in sorted(STYLE_DIR.glob("*.css")):
        if path.name in SKIP:
            continue
        original = path.read_text(encoding="utf-8")
        updated = migrate_text(original)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed += 1
            print(f"updated {path.name}")
    print(f"done ({changed} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
