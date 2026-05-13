"""python -m tests.benchmark.run 入口"""
from tests.benchmark.runner import (
    list_cases,
    list_sanity_cases,
    run_benchmark,
    run_sanity_benchmark,
)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DanQing Benchmark Runner")
    parser.add_argument("--all", action="store_true", help="Run all mflux comparison cases")
    parser.add_argument("--case", type=str, default="", help="Single case id (mflux bench or sanity)")
    parser.add_argument("--sanity", action="store_true",
                        help="Output sanity checks for models without mflux reference")
    parser.add_argument(
        "--ltx-video",
        action="store_true",
        help="LTX video: DanQing danqing-video-generate vs ltx-2-mlx CLI (bench venv)",
    )
    parser.add_argument(
        "--ltx-video-case",
        type=str,
        default="",
        metavar="ID",
        help="Run a single LTX video case (default: all when --ltx-video)",
    )
    parser.add_argument(
        "--ltx-ref-only",
        action="store_true",
        help="With --ltx-video: only run ltx-2-mlx reference mp4, skip DanQing",
    )
    parser.add_argument("--output-dir", type=str, default="tests/benchmark/outputs")
    parser.add_argument("--ref-only", action="store_true",
                        help="Only generate mflux reference images, skip DanQing")
    args = parser.parse_args()

    if args.ltx_video:
        if args.all or args.sanity or args.ref_only:
            parser.error("--ltx-video cannot be combined with --all, --sanity, or --ref-only")
        if args.case:
            parser.error("For LTX video suite use --ltx-video-case <id>, not --case")
        from tests.benchmark.ltx_video_runner import run_ltx_video_benchmark

        raise SystemExit(
            run_ltx_video_benchmark(
                args.ltx_video_case,
                run_ours=not args.ltx_ref_only,
                run_ref=True,
                output_dir=args.output_dir,
            )
        )

    if args.sanity:
        raise SystemExit(
            run_sanity_benchmark(case_id=args.case, output_dir=args.output_dir)
        )
    raise SystemExit(
        run_benchmark(
            case_id=args.case,
            run_all=args.all,
            run_ours=not args.ref_only,
            output_dir=args.output_dir,
        )
    )
