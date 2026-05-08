"""python -m tests.benchmark.run 入口"""
from tests.benchmark.runner import run_benchmark, list_cases

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DanQing Benchmark Runner")
    parser.add_argument("--all", action="store_true", help="Run all cases")
    parser.add_argument("--case", type=str, help="Run a single case by ID")
    parser.add_argument("--output-dir", type=str, default="tests/benchmark/outputs")
    parser.add_argument("--ref-only", action="store_true",
                        help="Only generate mflux reference images, skip DanQing")
    args = parser.parse_args()
    run_benchmark(
        case_id=args.case,
        run_all=args.all,
        run_ours=not args.ref_only,
        output_dir=args.output_dir,
    )
