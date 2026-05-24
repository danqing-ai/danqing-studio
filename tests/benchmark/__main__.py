"""``python -m tests.benchmark mflux|sanity|mlx-video|diffusers [--all|--case ID]``"""
from __future__ import annotations

import argparse
import sys

from tests.benchmark.cases import (
    iter_external_ref_cases_by_backend,
    iter_mflux_cases,
    list_cases,
    list_external_ref_cases_by_backend,
    list_sanity_cases,
    list_skipped_external_ref_cases_by_backend,
    list_skipped_mflux_cases,
    resolve_benchmark_data_root,
)
from tests.benchmark.run import run_diffusers, run_mflux, run_mlx_video, run_sanity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DanQing benchmark (mflux | sanity | mlx-video | diffusers)")
    sub = parser.add_subparsers(dest="suite", required=True)

    suite_defs = (
        ("mflux", list_cases, "mflux: PSNR vs reference CLI"),
        ("sanity", list_sanity_cases, "sanity: reject flat/noise outputs"),
        ("mlx-video", lambda: list_external_ref_cases_by_backend("mlx_video"), "mlx-video: parity vs mlx-video reference"),
        ("diffusers", lambda: list_external_ref_cases_by_backend("diffusers"), "diffusers: parity vs diffusers reference"),
    )
    for name, list_fn, help_text in suite_defs:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--all", action="store_true", help="Run all registered cases")
        p.add_argument("--case", metavar="ID", default="", help="Single case id")
        p.add_argument("--list", action="store_true", help="List case ids and exit")
        p.add_argument(
            "--list-runnable",
            action="store_true",
            help="(mflux/mlx-video/diffusers) List runnable cases with local bundles",
        )
        p.add_argument("--output-dir", default="tests/benchmark/outputs")

    args = parser.parse_args(argv)
    if getattr(args, "list_runnable", False):
        if args.suite not in ("mflux", "mlx-video", "diffusers"):
            parser.error("--list-runnable is only for mflux/mlx-video/diffusers suites")
        print(resolve_benchmark_data_root())
        if args.suite == "mflux":
            print("\n".join(c.id for c in iter_mflux_cases()))
            skipped = list_skipped_mflux_cases()
        elif args.suite == "mlx-video":
            print("\n".join(c.id for c in iter_external_ref_cases_by_backend("mlx_video")))
            skipped = list_skipped_external_ref_cases_by_backend("mlx_video")
        elif args.suite == "diffusers":
            print("\n".join(c.id for c in iter_external_ref_cases_by_backend("diffusers")))
            skipped = list_skipped_external_ref_cases_by_backend("diffusers")
        else:
            print("\n".join(c.id for c in iter_external_ref_cases_by_backend("diffusers")))
            skipped = list_skipped_external_ref_cases_by_backend("diffusers")
        if skipped:
            print("\n# skipped (no bundle)", file=sys.stderr)
            for cid, reason in skipped:
                print(f"{cid}\t{reason}", file=sys.stderr)
        return 0
    if args.list:
        if args.suite == "mflux":
            ids = list_cases()
        elif args.suite == "sanity":
            ids = list_sanity_cases()
        elif args.suite == "mlx-video":
            ids = list_external_ref_cases_by_backend("mlx_video")
        else:
            ids = list_external_ref_cases_by_backend("diffusers")
        print("\n".join(ids))
        return 0
    if args.all and args.case:
        parser.error("use either --all or --case, not both")
    if not args.all and not args.case:
        parser.error("specify --all or --case <id>")

    if args.suite == "mflux":
        return run_mflux("" if args.all else args.case, output_dir=args.output_dir)
    if args.suite == "sanity":
        return run_sanity("" if args.all else args.case, output_dir=args.output_dir)
    if args.suite == "mlx-video":
        return run_mlx_video("" if args.all else args.case, output_dir=args.output_dir)
    if args.suite == "diffusers":
        return run_diffusers("" if args.all else args.case, output_dir=args.output_dir)
    parser.error(f"Unknown suite: {args.suite}")


if __name__ == "__main__":
    raise SystemExit(main())
