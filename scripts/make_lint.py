#!/usr/bin/env python3
"""Makefile ``lint`` target — py_compile smoke for engine + benchmark helpers."""
from __future__ import annotations

import os
import py_compile
import sys


def main() -> int:
    fail = 0
    for root, _dirs, files in os.walk("backend/engine"):
        if "__pycache__" in root:
            continue
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f)
            try:
                py_compile.compile(path, doraise=True)
            except py_compile.PyCompileError:
                print("FAIL:", path)
                fail = 1
    extra = (
        "backend/main.py",
        "tests/benchmark/cases.py",
        "tests/benchmark/runner.py",
        "tests/benchmark/compare.py",
        "tests/benchmark/sanity.py",
        "tests/benchmark/test_exit_exempt.py",
    )
    for f in extra:
        try:
            py_compile.compile(f, doraise=True)
        except py_compile.PyCompileError:
            print("FAIL:", f)
            fail = 1
    return fail


if __name__ == "__main__":
    sys.exit(main())
