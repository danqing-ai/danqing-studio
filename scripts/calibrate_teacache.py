#!/usr/bin/env python3
"""
TeaCache calibration — probe rel_l1 during generation, replay thresholds offline.

Usage:
    # Record rel_l1 trace (sets DANQING_TEACACHE_PROBE=1 internally)
    python scripts/calibrate_teacache.py run \\
        --model flux1-dev --prompt "a mountain" --steps 28 \\
        --output /tmp/out.png --trace /tmp/flux1_teacache_trace.json

    # Replay trace → suggested threshold + skip-rate sweep
    python scripts/calibrate_teacache.py fit --trace /tmp/flux1_teacache_trace.json

    # Fit new polynomial coefficients (identity proxy; validate on real outputs before committing)
    python scripts/calibrate_teacache.py fit --trace /tmp/trace.json --fit-coefficients
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_CLI_FILE = Path(__file__).resolve()
PROJECT_ROOT = _CLI_FILE.parent.parent
if sys.platform != "win32":
    _venv_root = PROJECT_ROOT / ".venv"
    _venv_py = _venv_root / "bin" / "python3"
    if _venv_py.is_file() and Path(sys.prefix).resolve() != _venv_root.resolve():
        os.execv(str(_venv_py), [str(_venv_py), str(_CLI_FILE), *sys.argv[1:]])
sys.path.insert(0, str(PROJECT_ROOT))

from backend.engine.common.ops.step_cache import StepCacheSession
from backend.engine.common.ops.teacache_calibrate import (
    build_calibration_report,
    load_trace_json,
    print_calibration_report,
    resolve_family_from_model,
    write_trace_json,
)


def _cmd_run(args: argparse.Namespace) -> int:
    os.environ["DANQING_TEACACHE_PROBE"] = "1"
    project_root = Path(args.project_root).resolve()
    family = resolve_family_from_model(args.model, project_root=project_root)

    from backend.core.model_registry import ModelRegistry
    from backend.utils.path_utils import PathResolver

    reg = ModelRegistry.load(PathResolver(project_root).get_models_registry_path())
    entry = reg.require(args.model)
    media = entry.media

    trace_path = Path(args.trace).resolve()
    num_steps = int(args.steps)

    if media == "video":
        from backend.cli.video_cli import generate

        generate(
            model=args.model,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            size=args.size,
            num_frames=args.num_frames,
            fps=args.fps,
            steps=num_steps,
            guidance=args.guidance,
            shift=args.shift,
            seed=args.seed,
            output=args.output,
            project_root=project_root,
        )
    elif media == "image":
        from backend.cli.image_cli import generate

        generate(
            model=args.model,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            size=args.size,
            steps=num_steps,
            guidance=args.guidance,
            seed=args.seed,
            output=args.output,
            project_root=project_root,
        )
    else:
        raise RuntimeError(f"Model {args.model!r} media={media!r} is not image/video")

    rel_l1 = StepCacheSession.consume_probe_trace()
    if not rel_l1:
        raise RuntimeError(
            "Probe trace empty — model family may not support TeaCache probe "
            f"(family={family!r}, steps={num_steps})"
        )

    write_trace_json(
        trace_path,
        family=family,
        num_steps=num_steps,
        rel_l1=rel_l1,
        model=args.model,
        prompt=args.prompt,
    )
    print(f"[calibrate] wrote trace ({len(rel_l1)} rel_l1 samples) -> {trace_path}")

    report = build_calibration_report(
        rel_l1,
        family=family,
        num_steps=num_steps,
        target_skip_rate=float(args.target_skip_rate),
    )
    print_calibration_report(report)
    return 0


def _cmd_fit(args: argparse.Namespace) -> int:
    data = load_trace_json(Path(args.trace).resolve())
    family = str(args.family or data.get("family") or "")
    if not family:
        raise RuntimeError("Trace JSON missing family; pass --family")
    num_steps = int(args.steps or data.get("num_steps") or 0)
    if num_steps <= 0:
        raise RuntimeError("Trace JSON missing num_steps; pass --steps")

    rel_l1 = [float(x) for x in data["rel_l1"]]
    report = build_calibration_report(
        rel_l1,
        family=family,
        num_steps=num_steps,
        target_skip_rate=float(args.target_skip_rate),
        fit_coefficients=bool(args.fit_coefficients),
    )
    print_calibration_report(report, fit_coefficients=bool(args.fit_coefficients))

    if args.write_report:
        out = Path(args.write_report).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[calibrate] report -> {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="TeaCache calibration (probe + replay)")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run probe generation and write rel_l1 trace JSON")
    run_p.add_argument("--model", required=True)
    run_p.add_argument("--prompt", required=True)
    run_p.add_argument("--negative-prompt", default="")
    run_p.add_argument("--size", default="1024x1024")
    run_p.add_argument("--steps", type=int, required=True)
    run_p.add_argument("--guidance", type=float, default=None)
    run_p.add_argument("--seed", type=int, default=None)
    run_p.add_argument("--num-frames", type=int, default=None)
    run_p.add_argument("--fps", type=int, default=None)
    run_p.add_argument("--shift", type=float, default=None)
    run_p.add_argument("--output", required=True, help="Generation output path (image or video)")
    run_p.add_argument("--trace", required=True, help="Where to write rel_l1 trace JSON")
    run_p.add_argument("--target-skip-rate", type=float, default=0.35)
    run_p.set_defaults(func=_cmd_run)

    fit_p = sub.add_parser("fit", help="Replay trace JSON and suggest TeaCache threshold")
    fit_p.add_argument("--trace", required=True)
    fit_p.add_argument("--family", default="", help="Override family in trace JSON")
    fit_p.add_argument("--steps", type=int, default=0, help="Override num_steps in trace JSON")
    fit_p.add_argument("--target-skip-rate", type=float, default=0.35)
    fit_p.add_argument("--fit-coefficients", action="store_true")
    fit_p.add_argument("--write-report", default="", help="Optional JSON report path")
    fit_p.set_defaults(func=_cmd_fit)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
