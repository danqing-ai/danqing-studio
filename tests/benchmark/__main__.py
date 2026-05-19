"""``python -m tests.benchmark mflux|sanity [--all|--case ID]``"""
from __future__ import annotations

import argparse
import sys

from tests.benchmark.cases import (
    iter_mflux_cases,
    list_cases,
    list_sanity_cases,
    list_skipped_mflux_cases,
    resolve_benchmark_data_root,
)
from tests.benchmark.run import run_mflux, run_sanity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DanQing benchmark (mflux PSNR | output sanity)")
    sub = parser.add_subparsers(dest="suite", required=True)

    for name, list_fn in (("mflux", list_cases), ("sanity", list_sanity_cases)):
        p = sub.add_parser(name, help="mflux: PSNR vs reference CLI" if name == "mflux" else "sanity: reject flat outputs")
        p.add_argument("--all", action="store_true", help="Run all registered cases")
        p.add_argument("--case", metavar="ID", default="", help="Single case id")
        p.add_argument("--list", action="store_true", help="List case ids and exit")
        p.add_argument(
            "--list-runnable",
            action="store_true",
            help="(mflux) List cases with local fp16 bundle under workspace pointer",
        )
        p.add_argument("--output-dir", default="tests/benchmark/outputs")

    args = parser.parse_args(argv)
    if getattr(args, "list_runnable", False):
        if args.suite != "mflux":
            parser.error("--list-runnable is only for the mflux suite")
        print(resolve_benchmark_data_root())
        print("\n".join(c.id for c in iter_mflux_cases()))
        skipped = list_skipped_mflux_cases()
        if skipped:
            print("\n# skipped (no bundle)", file=sys.stderr)
            for cid, reason in skipped:
                print(f"{cid}\t{reason}", file=sys.stderr)
        return 0
    if args.list:
        ids = list_cases() if args.suite == "mflux" else list_sanity_cases()
        print("\n".join(ids))
        return 0
    if args.all and args.case:
        parser.error("use either --all or --case, not both")
    if not args.all and not args.case:
        parser.error("specify --all or --case <id>")

    if args.suite == "mflux":
        return run_mflux("" if args.all else args.case, output_dir=args.output_dir)
    return run_sanity("" if args.all else args.case, output_dir=args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
