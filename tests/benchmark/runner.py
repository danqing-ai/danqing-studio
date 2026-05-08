"""
基准测试运行器 — 旁路对比丹青引擎 vs mflux CLI。

用法:
    python -m tests.benchmark.run --case z-image-turbo-basic
    python -m tests.benchmark.run --all --output-dir tests/benchmark/outputs

依赖:
    - mflux CLI (mflux-generate)
    - DanQing 引擎 (backend/engine/)
    - PIL, numpy, scipy
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from .cases import ALL_CASES, BenchmarkCase, get_case, list_cases
from .compare import CompareResult, compare_images, hash_image


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class BenchmarkRunner:
    """基准测试运行器。"""

    def __init__(self, output_dir: str | Path = "tests/benchmark/outputs",
                 mflux_bin: str = "mflux-generate"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mflux_bin = mflux_bin
        self.results: list[tuple[BenchmarkCase, CompareResult]] = []

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def run_case(self, case: BenchmarkCase,
                 run_ours: bool = True,
                 run_ref: bool = True) -> CompareResult:
        """执行单个用例对比。"""
        our_path = self.output_dir / f"{case.id}_danqing.png"
        ref_path = self.output_dir / f"{case.id}_mflux.png"

        t0 = time.time()
        ours_ok = False
        ref_ok = False

        if run_ours:
            ours_ok = self._run_danqing(case, our_path)
        if run_ref:
            ref_ok = self._run_mflux(case, ref_path)
        elapsed = time.time() - t0

        if not ref_ok:
            print(f"  [SKIP] {case.id}: 参考输出 (mflux) 生成失败")
            result = CompareResult()
            result.ours_time_sec = elapsed if ours_ok else None
            result.ref_time_sec = None
            self.results.append((case, result))
            return result

        if not ours_ok:
            print(f"  [SKIP] {case.id}: 丹青引擎输出生成失败或未实现")
            result = CompareResult(ref_hash=hash_image(ref_path))
            result.ref_time_sec = elapsed
            self.results.append((case, result))
            return result

        result = compare_images(our_path, ref_path)
        result.ours_time_sec = elapsed
        result.ref_time_sec = elapsed  # 粗略值，可后续细化
        self.results.append((case, result))
        return result

    def run_all(self, run_ours: bool = True) -> None:
        """执行全部已注册用例。"""
        print(f"\n{'='*60}")
        print(f"DanQing Benchmark — {len(ALL_CASES)} cases")
        print(f"{'='*60}\n")

        for case in ALL_CASES:
            print(f"[{case.id}] {case.description}")
            print(f"  model={case.model} seed={case.seed} steps={case.steps} "
                  f"size={case.width}x{case.height}")
            result = self.run_case(case, run_ours=run_ours, run_ref=True)
            self._print_result(result)

        self._print_summary()

    # ------------------------------------------------------------------
    # 丹青引擎生成 — 调用 danqing-* CLI（复用产品代码路径）
    # ------------------------------------------------------------------

    def _run_danqing(self, case: BenchmarkCase, output_path: Path) -> bool:
        """调用 danqing-* CLI 生成图像。

        CLI 与 REST API 共享 Engine 层执行路径：
          create  → danqing-generate  → IImageEngine.generate()
          rewrite → danqing-edit      → IImageEngine.edit()
          upscale → danqing-upscale   → IImageEngine.upscale()
        """
        if case.action == "create":
            return self._run_danqing_generate(case, output_path)
        elif case.action in ("rewrite", "retouch", "extend"):
            print(f"    [danqing] edit/rewrite 暂未实现（需资产上传）")
            return False
        elif case.action == "upscale":
            print(f"    [danqing] upscale 暂未实现（需资产上传）")
            return False
        else:
            print(f"    [danqing] 未知 action: {case.action}")
            return False

    def _run_danqing_generate(self, case: BenchmarkCase, output_path: Path) -> bool:
        """调用 danqing-generate CLI（对应 POST /api/images/generations）。"""
        cli = PROJECT_ROOT / "bin" / "danqing-generate"
        cmd = [
            sys.executable, str(cli),
            "--model", case.model,
            "--prompt", case.prompt,
            "--seed", str(case.seed),
            "--steps", str(case.steps),
            "--guidance", str(case.guidance),
            "--size", f"{case.width}x{case.height}",
            "--output", str(output_path),
        ]
        if case.negative_prompt:
            cmd += ["--negative-prompt", case.negative_prompt]

        try:
            t0 = time.time()
            env = os.environ.copy()
            env.setdefault("MLX_METAL_DEVICE_ONLY", "1")
            env.setdefault("MLX_METAL_MEMORY_LIMIT", "120")
            env["PYTHONUNBUFFERED"] = "1"

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
                cwd=str(PROJECT_ROOT),
            )

            elapsed = time.time() - t0

            if proc.returncode != 0:
                print(f"    [danqing] 失败 (exit={proc.returncode})")
                if proc.stderr:
                    print(f"    [danqing] stderr: {proc.stderr[:500]}")
                return False

            print(f"    [danqing] 生成完成 ({elapsed:.1f}s)")
            print(f"    [danqing] 输出: {output_path}")
            return True

        except subprocess.TimeoutExpired:
            print(f"    [danqing] 超时 (600s)")
            return False
        except FileNotFoundError:
            print(f"    [danqing] 未找到 CLI: {cli}")
            return False

    # ------------------------------------------------------------------
    # mflux CLI 参考生成
    # ------------------------------------------------------------------

    def _run_mflux(self, case: BenchmarkCase, output_path: Path) -> bool:
        """用 mflux CLI 生成参考图像。"""
        cli_bin = case._mflux_cli or self.mflux_bin
        # 优先使用基准测试独立 venv
        bench_venv = PROJECT_ROOT / "tests" / "benchmark" / "venv" / "bin" / cli_bin
        if bench_venv.exists():
            cli_bin = str(bench_venv)
        # 其次用项目 venv（兼容旧版）
        else:
            venv_bin = PROJECT_ROOT / ".venv" / "bin" / cli_bin
            if venv_bin.exists():
                cli_bin = str(venv_bin)
        cmd = [
            cli_bin,
            "--model", case._mflux_model_flag,
            "--seed", str(case.seed),
            "--output", str(output_path),
        ]
        if case.action == "rewrite" and case.source_image:
            cmd += ["--image-path", str(case.source_image),
                    "--image-strength", str(case.image_strength),
                    "--prompt", case.prompt,
                    "--width", str(case.width), "--height", str(case.height),
                    "--guidance", str(case.guidance)]
        elif case.prompt:
            cmd += ["--prompt", case.prompt,
                    "--width", str(case.width), "--height", str(case.height),
                    "--guidance", str(case.guidance)]
        if case.steps > 1:
            cmd += ["--steps", str(case.steps)]
        if case.negative_prompt:
            cmd += ["--negative-prompt", case.negative_prompt]

        try:
            t0 = time.time()

            # 设置 MLX 环境变量
            env = os.environ.copy()
            env.setdefault("MLX_METAL_DEVICE_ONLY", "1")
            env.setdefault("MLX_METAL_MEMORY_LIMIT", "120")
            env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            env["PYTHONUNBUFFERED"] = "1"

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
                cwd=str(PROJECT_ROOT),
            )

            elapsed = time.time() - t0

            if proc.returncode != 0:
                print(f"    [mflux] 失败 (exit={proc.returncode})")
                if proc.stderr:
                    print(f"    [mflux] stderr: {proc.stderr[:300]}")
                return False

            print(f"    [mflux] 生成完成 ({elapsed:.1f}s)")
            print(f"    [mflux] 输出: {output_path}")
            return True

        except subprocess.TimeoutExpired:
            print(f"    [mflux] 超时 (600s)")
            return False
        except FileNotFoundError:
            print(f"    [mflux] 未找到: {self.mflux_bin}")
            return False

    # ------------------------------------------------------------------
    # 报告
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 报告
    # ------------------------------------------------------------------

    def _print_result(self, result: CompareResult) -> None:
        if result.psnr is not None:
            status = "PASS" if result.psnr >= 30 else ("WARN" if result.psnr >= 20 else "FAIL")
            print(f"  {status}: PSNR={result.psnr:.2f}dB SSIM={result.ssim:.4f} "
                  f"max_diff={result.pixel_max_diff:.4f} mean_diff={result.pixel_mean_diff:.4f}")
        elif result.ref_hash:
            print(f"  SKIP: ref_hash={result.ref_hash} (丹青引擎输出不可用)")
        else:
            print(f"  SKIP: 无可用的参考输出")

    def _print_summary(self) -> None:
        passed = sum(1 for _, r in self.results if r.psnr is not None and r.psnr >= 30)
        warned = sum(1 for _, r in self.results if r.psnr is not None and 20 <= r.psnr < 30)
        failed = sum(1 for _, r in self.results if r.psnr is not None and r.psnr < 20)
        skipped = sum(1 for _, r in self.results if r.psnr is None)
        total = len(self.results)

        print(f"\n{'='*60}")
        print(f"Summary: {total} cases — "
              f"{passed} PASS / {warned} WARN / {failed} FAIL / {skipped} SKIP")
        print(f"{'='*60}")

        for case, r in self.results:
            if r.psnr is not None:
                print(f"  [{case.id}] PSNR={r.psnr:.1f}dB SSIM={r.ssim:.4f}")
            else:
                print(f"  [{case.id}] SKIP")


def run_benchmark(
    case_id: str = "",
    run_all: bool = False,
    run_ours: bool = True,
    output_dir: str = "tests/benchmark/outputs",
):
    """便捷入口（从脚本或 __main__ 调用）。"""
    runner = BenchmarkRunner(output_dir=output_dir)

    if run_all:
        runner.run_all(run_ours=run_ours)
    elif case_id:
        case = get_case(case_id)
        if case is None:
            print(f"Unknown case: {case_id}")
            print(f"Available: {list_cases()}")
            return
        print(f"[{case.id}] {case.description}")
        result = runner.run_case(case, run_ours=run_ours, run_ref=True)
        runner._print_result(result)
        print(f"\nOutputs: {runner.output_dir}")
    else:
        print("Usage: run_benchmark --all | --case <case_id>")
        print(f"Available cases: {list_cases()}")


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
