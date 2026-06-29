#!/usr/bin/env python3
"""CLI alias for fixed chapter-parse benchmark cases."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.chapter_parse_benchmark import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--case" not in argv and "--project-id" not in argv and "--script-file" not in argv:
        argv = ["--case", "all", *argv]
    raise SystemExit(main(argv))
