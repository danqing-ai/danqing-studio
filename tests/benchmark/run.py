"""Benchmark runners: ``mflux`` (PSNR vs reference CLI) and ``sanity`` (output flat-field gate)."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from .cases import (
    ALL_CASES,
    ALL_SANITY_CASES,
    BENCHMARK_EXIT_EXEMPT_MISMATCH_VS_MFLUX,
    BenchmarkCase,
    SanityCase,
    get_case,
    get_sanity_case,
    iter_mflux_cases,
    list_cases,
    list_sanity_cases,
    list_skipped_mflux_cases,
    resolve_benchmark_data_root,
    resolve_fp16_bundle_dir,
)
from .metrics import CompareResult, SanityResult, check_output_image, compare_images, hash_image

# 与参考图对比：PSNR 档位 + SSIM 下限（纯噪声 / 完全跑崩时相对参考图 SSIM 极低）
MIN_PSNR_PASS = 30.0
MIN_PSNR_WARN = 20.0
MIN_SSIM_NOT_CATASTROPHIC = 0.12


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 默认 CLI 超时；FIBO 等首包加载可用 ``SanityCase.timeout_sec`` 覆盖
DEFAULT_DANQING_CLI_TIMEOUT_SEC = 600
# 失败时打印的 stderr 上限（字符截断）
STDERR_HEAD_CHARS = 4000


def _seedvr2_flat_bundle_ready(model_id: str) -> bool:
    """与 ``job_mlx.validate_seedvr2_bundle`` 一致：扁平目录下两份 safetensors 齐全。"""
    base = model_id.split(":", 1)[0].strip()
    try:
        root = resolve_fp16_bundle_dir(base)
    except KeyError:
        return False
    if not root.is_dir():
        return False
    try:
        from backend.engine.families.seedvr2.job_mlx import expected_seedvr2_weight_files
    except Exception:
        return False
    for name in expected_seedvr2_weight_files(base):
        if not (root / name).is_file():
            return False
    return True


def _benchmark_source_path(case: BenchmarkCase) -> Path:
    """rewrite / upscale 源图路径（相对路径相对项目根）。"""
    p = Path(case.source_image)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def ensure_benchmark_source_image() -> Path:
    """rewrite / upscale 用源图；不存在时写一张 256² 占位 PNG（无需 mflux ``make bench-src``）。"""
    p = PROJECT_ROOT / "tests" / "benchmark" / "outputs" / "rewrite_src.png"
    if p.is_file():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        import numpy as np
    except ImportError as e:
        raise RuntimeError(
            "Benchmark needs PIL+numpy to create tests/benchmark/outputs/rewrite_src.png"
        ) from e
    rng = np.random.default_rng(1)
    arr = (rng.random((256, 256, 3)) * 220 + 16).astype("uint8")
    Image.fromarray(arr).save(p)
    return p


class BenchmarkRunner:
    """基准测试运行器。"""

    def __init__(self, output_dir: str | Path = "tests/benchmark/outputs",
                 mflux_bin: str = "mflux-generate"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mflux_bin = mflux_bin
        self.results: list[tuple[BenchmarkCase, CompareResult]] = []

    @staticmethod
    def grade(result: CompareResult) -> str:
        """SKIP | FAIL | WARN | PASS — FAIL 含明显噪声级不一致（PSNR 或 SSIM 过低）。"""
        if result.psnr is None:
            return "SKIP"
        ssim = result.ssim if result.ssim is not None else 0.0
        if result.psnr < MIN_PSNR_WARN or ssim < MIN_SSIM_NOT_CATASTROPHIC:
            return "FAIL"
        if result.psnr >= MIN_PSNR_PASS:
            return "PASS"
        return "WARN"

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def run_case(self, case: BenchmarkCase,
                 run_ours: bool = True,
                 run_ref: bool = True) -> CompareResult:
        """执行单个用例对比。"""
        base = case.model.split(":", 1)[0].strip()
        try:
            bundle = resolve_fp16_bundle_dir(base)
        except KeyError:
            bundle = None
        if bundle is None or not bundle.is_dir():
            print(f"  [SKIP] {case.id}: fp16 bundle not installed ({base})")
            result = CompareResult()
            self.results.append((case, result))
            return result
        if case.action in ("rewrite", "upscale") or case.source_image:
            ensure_benchmark_source_image()
        our_path = self.output_dir / f"{case.id}_danqing.png"
        ref_path = self.output_dir / f"{case.id}_mflux.png"

        ours_ok = False
        ref_ok = False
        ours_time = None
        ref_time = None

        if run_ours:
            t0 = time.time()
            ours_ok = self._run_danqing(case, our_path)
            ours_time = time.time() - t0

        if run_ref:
            t0 = time.time()
            ref_ok = self._run_mflux(case, ref_path)
            ref_time = time.time() - t0

        if not ref_ok:
            print(f"  [SKIP] {case.id}: 参考输出 (mflux) 生成失败")
            result = CompareResult()
            result.ours_time_sec = ours_time
            result.ref_time_sec = None
            self.results.append((case, result))
            return result

        if not ours_ok:
            print(f"  [SKIP] {case.id}: 丹青引擎输出生成失败或未实现")
            result = CompareResult(ref_hash=hash_image(ref_path))
            result.ref_time_sec = ref_time
            self.results.append((case, result))
            return result

        result = compare_images(our_path, ref_path)
        result.ours_time_sec = ours_time
        result.ref_time_sec = ref_time
        self.results.append((case, result))
        return result

    def run_all(self, run_ours: bool = True) -> None:
        """执行全部已注册用例（跳过 workspace 中未安装的 fp16 bundle）。"""
        ensure_benchmark_source_image()
        cases = iter_mflux_cases()
        skipped = list_skipped_mflux_cases()
        print(f"\n{'='*60}")
        print(f"DanQing Benchmark — {len(cases)} runnable / {len(ALL_CASES)} registered")
        print(f"Data root: {resolve_benchmark_data_root()}")
        if skipped:
            print(f"Skipped (no local bundle): {len(skipped)}")
        print(f"{'='*60}\n")
        for case_id, reason in skipped:
            print(f"  [SKIP] {case_id}: {reason}")

        for case in cases:
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
            return self._run_danqing_edit(case, output_path)
        elif case.action == "upscale":
            return self._run_danqing_upscale(case, output_path)
        else:
            print(f"    [danqing] 未知 action: {case.action}")
            return False

    def _run_danqing_generate(self, case: BenchmarkCase, output_path: Path, *, timeout_sec: int | None = None) -> bool:
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
        if getattr(case, "scheduler", None):
            cmd += ["--scheduler", str(case.scheduler)]

        try:
            t0 = time.time()
            env = os.environ.copy()
            env.setdefault("MLX_METAL_DEVICE_ONLY", "1")
            env.setdefault("MLX_METAL_MEMORY_LIMIT", "120")
            env["PYTHONUNBUFFERED"] = "1"
            tout = int(timeout_sec) if timeout_sec is not None else DEFAULT_DANQING_CLI_TIMEOUT_SEC

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=tout,
                env=env,
                cwd=str(PROJECT_ROOT),
            )

            elapsed = time.time() - t0

            if proc.returncode != 0:
                print(f"    [danqing] 失败 (exit={proc.returncode})")
                if proc.stderr:
                    print(f"    [danqing] stderr: {proc.stderr[:STDERR_HEAD_CHARS]}")
                return False

            print(f"    [danqing] 生成完成 ({elapsed:.1f}s)")
            print(f"    [danqing] 输出: {output_path}")
            return True

        except subprocess.TimeoutExpired:
            print(f"    [danqing] 超时 ({tout}s)")
            return False
        except FileNotFoundError:
            print(f"    [danqing] 未找到 CLI: {cli}")
            return False

    def _run_danqing_upscale(self, case: BenchmarkCase, output_path: Path, *, timeout_sec: int | None = None) -> bool:
        """调用 danqing-upscale CLI（对应 POST /api/images/upscales）。"""
        ensure_benchmark_source_image()
        cli = PROJECT_ROOT / "bin" / "danqing-upscale"
        src = _benchmark_source_path(case)
        if not src.is_file():
            print(f"    [danqing] 源图不存在: {src}")
            return False
        scale = int(getattr(case, "upscale_scale", 2) or 2)
        if scale not in (2, 4):
            scale = 2
        cmd = [
            sys.executable, str(cli),
            "--model", case.model,
            "--source-image", str(src),
            "--scale-factor", str(scale),
            "--seed", str(case.seed),
            "--output", str(output_path),
        ]
        try:
            t0 = time.time()
            env = os.environ.copy()
            env.setdefault("MLX_METAL_DEVICE_ONLY", "1")
            env.setdefault("MLX_METAL_MEMORY_LIMIT", "120")
            env["PYTHONUNBUFFERED"] = "1"
            tout = int(timeout_sec) if timeout_sec is not None else DEFAULT_DANQING_CLI_TIMEOUT_SEC
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=tout, env=env, cwd=str(PROJECT_ROOT),
            )
            elapsed = time.time() - t0
            if proc.returncode != 0:
                print(f"    [danqing] 失败 (exit={proc.returncode})")
                if proc.stderr:
                    print(f"    [danqing] stderr: {proc.stderr[:STDERR_HEAD_CHARS]}")
                return False
            print(f"    [danqing] 超分完成 ({elapsed:.1f}s)")
            print(f"    [danqing] 输出: {output_path}")
            return True
        except subprocess.TimeoutExpired:
            print(f"    [danqing] 超时 ({tout}s)")
            return False
        except FileNotFoundError:
            print(f"    [danqing] 未找到 CLI: {cli}")
            return False

    def _run_danqing_edit(self, case: BenchmarkCase, output_path: Path, *, timeout_sec: int | None = None) -> bool:
        """调用 danqing-edit CLI（对应 POST /api/images/edits）。"""
        ensure_benchmark_source_image()
        cli = PROJECT_ROOT / "bin" / "danqing-edit"
        cmd = [
            sys.executable, str(cli),
            "--model", case.model,
            "--operation", "rewrite",
            "--source-image", str(_benchmark_source_path(case)),
            "--prompt", case.prompt,
            "--seed", str(case.seed),
            "--source-fidelity", str(case.image_strength),
            "--guidance", str(case.guidance),
            "--output", str(output_path),
        ]
        if case.steps > 1:
            cmd += ["--steps", str(case.steps)]
        if getattr(case, "scheduler", None):
            cmd += ["--scheduler", str(case.scheduler)]
        try:
            t0 = time.time()
            env = os.environ.copy()
            env.setdefault("MLX_METAL_DEVICE_ONLY", "1")
            env.setdefault("MLX_METAL_MEMORY_LIMIT", "120")
            env["PYTHONUNBUFFERED"] = "1"
            tout = int(timeout_sec) if timeout_sec is not None else DEFAULT_DANQING_CLI_TIMEOUT_SEC
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=tout, env=env, cwd=str(PROJECT_ROOT))
            if proc.returncode != 0:
                print(f"    [danqing] 失败 (exit={proc.returncode})")
                if proc.stderr:
                    print(f"    [danqing] stderr: {proc.stderr[:STDERR_HEAD_CHARS]}")
                return False
            elapsed = time.time() - t0
            print(f"    [danqing] 生成完成 ({elapsed:.1f}s)")
            print(f"    [danqing] 输出: {output_path}")
            return True
        except subprocess.TimeoutExpired:
            print(f"    [danqing] 超时 ({tout}s)")
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
        if case.action == "upscale" and case.source_image:
            src = _benchmark_source_path(case)
            scale = int(getattr(case, "upscale_scale", 2) or 2)
            if scale not in (2, 4):
                scale = 2
            cmd += [
                "--image-path",
                str(src),
                "--resolution",
                f"{scale}x",
            ]
        elif case.action == "rewrite" and case.source_image:
            cmd += ["--image-path", str(_benchmark_source_path(case)),
                    "--image-strength", str(case.image_strength),
                    "--prompt", case.prompt,
                    "--width", str(case.width), "--height", str(case.height),
                    "--guidance", str(case.guidance)]
        elif case.prompt:
            cmd += ["--prompt", case.prompt,
                    "--width", str(case.width), "--height", str(case.height),
                    "--guidance", str(case.guidance)]
        if getattr(case, "scheduler", None):
            cmd += ["--scheduler", str(case.scheduler)]
        if case.steps > 1:
            cmd += ["--steps", str(case.steps)]
        if case.negative_prompt:
            cmd += ["--negative-prompt", case.negative_prompt]
        if case._mflux_extra_args:
            cmd += case._mflux_extra_args

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

    def _print_result(self, result: CompareResult) -> None:
        time_info = ""
        if result.ours_time_sec is not None or result.ref_time_sec is not None:
            parts = []
            if result.ours_time_sec is not None:
                parts.append(f"ours={result.ours_time_sec:.1f}s")
            if result.ref_time_sec is not None:
                parts.append(f"ref={result.ref_time_sec:.1f}s")
            time_info = " (" + ", ".join(parts) + ")"

        if result.psnr is not None:
            status = BenchmarkRunner.grade(result)
            print(f"  {status}: PSNR={result.psnr:.2f}dB SSIM={result.ssim:.4f} "
                  f"max_diff={result.pixel_max_diff:.4f} mean_diff={result.pixel_mean_diff:.4f}{time_info}")
        elif result.ref_hash:
            print(f"  SKIP: ref_hash={result.ref_hash} (丹青引擎输出不可用){time_info}")
        else:
            print(f"  SKIP: 无可用的参考输出{time_info}")

    def _print_summary(self) -> None:
        passed = sum(1 for _, r in self.results if BenchmarkRunner.grade(r) == "PASS")
        warned = sum(1 for _, r in self.results if BenchmarkRunner.grade(r) == "WARN")
        failed = sum(1 for _, r in self.results if BenchmarkRunner.grade(r) == "FAIL")
        exempt_fail = sum(
            1
            for c, r in self.results
            if BenchmarkRunner.grade(r) == "FAIL" and c.id in BENCHMARK_EXIT_EXEMPT_MISMATCH_VS_MFLUX
        )
        skipped = sum(1 for _, r in self.results if BenchmarkRunner.grade(r) == "SKIP")
        total = len(self.results)

        print(f"\n{'='*60}")
        print(f"Summary: {total} cases — "
              f"{passed} PASS / {warned} WARN / {failed} FAIL / {skipped} SKIP")
        if exempt_fail:
            ids = ", ".join(sorted(BENCHMARK_EXIT_EXEMPT_MISMATCH_VS_MFLUX))
            print(
                f"KNOWN GAP (exempt from --all exit code): {exempt_fail} case(s) "
                f"in {{{ids}}} — rewrite vs mflux PSNR not yet aligned"
            )
        print(f"{'='*60}")

        for case, r in self.results:
            time_str = ""
            if r.ours_time_sec is not None or r.ref_time_sec is not None:
                parts = []
                if r.ours_time_sec is not None:
                    parts.append(f"ours={r.ours_time_sec:.1f}s")
                if r.ref_time_sec is not None:
                    parts.append(f"ref={r.ref_time_sec:.1f}s")
                time_str = " (" + ", ".join(parts) + ")"

            if r.psnr is not None:
                g = BenchmarkRunner.grade(r)
                print(f"  [{case.id}] {g} PSNR={r.psnr:.1f}dB SSIM={r.ssim:.4f}{time_str}")
            else:
                print(f"  [{case.id}] SKIP{time_str}")


class SanitySuiteRunner(BenchmarkRunner):
    """无 mflux 参考：仅生成 + 像素健全性（见 ``sanity.check_output_image``）。"""

    def __init__(self, output_dir: str | Path = "tests/benchmark/outputs"):
        super().__init__(output_dir=output_dir)
        self.sanity_results: list[tuple[SanityCase, SanityResult]] = []

    def run_one_sanity(self, case: SanityCase) -> SanityResult:
        out_path = self.output_dir / f"{case.id}_sanity.png"
        bc = case.as_benchmark_case()
        ensure_benchmark_source_image()
        tout = case.timeout_sec if case.timeout_sec is not None else DEFAULT_DANQING_CLI_TIMEOUT_SEC

        if bc.action == "upscale" and str(bc.model).startswith("seedvr2"):
            if not _seedvr2_flat_bundle_ready(bc.model):
                print(f"    [SKIP] 本地权重不完整（缺 {bc.model} 扁平 bundle），与 make bench-seedvr2-mflux 一致")
                res = SanityResult(
                    ok=True,
                    reason="skip_missing_seedvr2_weights",
                    mean_luma=0.0,
                    std_luma=0.0,
                    entropy_bits=0.0,
                    laplacian_var=0.0,
                    skipped=True,
                )
                self.sanity_results.append((case, res))
                print(f"  SKIP: {res.reason}")
                return res

        t0 = time.time()
        if bc.action == "rewrite":
            ok = self._run_danqing_edit(bc, out_path, timeout_sec=tout)
        elif bc.action == "upscale":
            ok = self._run_danqing_upscale(bc, out_path, timeout_sec=tout)
        else:
            ok = self._run_danqing_generate(bc, out_path, timeout_sec=tout)
        elapsed = time.time() - t0
        if not ok:
            res = SanityResult(
                ok=False,
                reason="danqing_cli_failed",
                mean_luma=0.0,
                std_luma=0.0,
                entropy_bits=0.0,
                laplacian_var=0.0,
            )
            self.sanity_results.append((case, res))
            print(f"  FAIL: {res.reason} ({elapsed:.1f}s)")
            return res
        res = check_output_image(out_path)
        self.sanity_results.append((case, res))
        status = "PASS" if res.ok else "FAIL"
        detail = res.reason if not res.ok else (
            f"std={res.std_luma:.4f} entropy_bits={res.entropy_bits:.2f} lap_var={res.laplacian_var:.2f}"
        )
        print(f"  {status}: {detail} ({elapsed:.1f}s)")
        print(f"    output: {out_path}")
        return res

    def run_all_sanity_cases(self) -> None:
        print(f"\n{'='*60}")
        print(f"DanQing Output Sanity — {len(ALL_SANITY_CASES)} cases (no mflux reference)")
        print(f"{'='*60}\n")
        for case in ALL_SANITY_CASES:
            print(f"[{case.id}] {case.description}")
            print(f"  model={case.model} seed={case.seed} steps={case.steps} "
                  f"size={case.width}x{case.height} guidance={case.guidance}")
            self.run_one_sanity(case)
        self._print_sanity_summary()

    def _print_sanity_summary(self) -> None:
        ok_n = sum(1 for _, r in self.sanity_results if r.ok and not r.skipped)
        skip_n = sum(1 for _, r in self.sanity_results if r.skipped)
        bad_n = sum(1 for _, r in self.sanity_results if not r.ok)
        total = len(self.sanity_results)
        print(f"\n{'='*60}")
        print(f"Sanity summary: {total} cases — {ok_n} PASS / {skip_n} SKIP / {bad_n} FAIL")
        print(f"{'='*60}")
        for case, r in self.sanity_results:
            if r.skipped:
                print(f"  [{case.id}] SKIP ({r.reason})")
            else:
                tag = "PASS" if r.ok else "FAIL"
                extra = f" ({r.reason})" if not r.ok else ""
                print(f"  [{case.id}] {tag}{extra}")


def run_sanity(case_id: str = "", output_dir: str = "tests/benchmark/outputs") -> int:
    """Sanity suite: ``case_id`` empty runs all ``ALL_SANITY_CASES``. Exit 1 on any FAIL."""
    runner = SanitySuiteRunner(output_dir=output_dir)
    if case_id:
        sc = get_sanity_case(case_id)
        if sc is None:
            print(f"Unknown sanity case: {case_id}")
            print(f"Available: {list_sanity_cases()}")
            return 2
        print(f"[{sc.id}] {sc.description}")
        r = runner.run_one_sanity(sc)
        print(f"\nOutputs: {runner.output_dir}")
        return 0 if r.ok else 1
    runner.run_all_sanity_cases()
    failed = sum(1 for _, r in runner.sanity_results if not r.ok)
    return 1 if failed else 0


def run_mflux(case_id: str = "", output_dir: str = "tests/benchmark/outputs") -> int:
    """Mflux PSNR suite: ``case_id`` empty runs all ``ALL_CASES``. Exit 1 on non-exempt FAIL."""
    runner = BenchmarkRunner(output_dir=output_dir)
    if not case_id:
        runner.run_all(run_ours=True)
        failed = sum(
            1
            for c, r in runner.results
            if BenchmarkRunner.grade(r) == "FAIL" and c.id not in BENCHMARK_EXIT_EXEMPT_MISMATCH_VS_MFLUX
        )
        return 1 if failed else 0
    case = get_case(case_id)
    if case is None:
        print(f"Unknown mflux case: {case_id}")
        print(f"Available: {list_cases()}")
        return 2
    print(f"[{case.id}] {case.description}")
    result = runner.run_case(case, run_ours=True, run_ref=True)
    runner._print_result(result)
    print(f"\nOutputs: {runner.output_dir}")
    grade = BenchmarkRunner.grade(result)
    if grade == "FAIL" and case.id in BENCHMARK_EXIT_EXEMPT_MISMATCH_VS_MFLUX:
        print("  (exit 0: exempt — see BENCHMARK_EXIT_EXEMPT_MISMATCH_VS_MFLUX in cases.py)")
        return 0
    return 1 if grade == "FAIL" else 0
