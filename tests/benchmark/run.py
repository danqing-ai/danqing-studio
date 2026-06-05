"""Benchmark runners: ``mflux`` (PSNR vs reference CLI) and ``sanity`` (output flat-field gate)."""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .cases import (
    ALL_CASES,
    ALL_SANITY_CASES,
    BENCHMARK_EXIT_EXEMPT_MISMATCH_VS_MFLUX,
    BenchmarkCase,
    ExternalRefCase,
    SRC_IMAGE,
    SanityCase,
    ace_step_bundle_installed,
    cuda_runtime_available,
    heartmula_bundle_installed,
    mlx_runtime_available,
    get_external_ref_case,
    get_case,
    get_sanity_case,
    iter_external_ref_cases,
    iter_external_ref_cases_by_backend,
    iter_mflux_cases,
    list_cases,
    list_external_ref_cases,
    list_external_ref_cases_by_backend,
    list_skipped_external_ref_cases_by_backend,
    list_sanity_cases,
    list_skipped_external_ref_cases,
    list_skipped_mflux_cases,
    resolve_benchmark_data_root,
    resolve_fp16_bundle_dir,
    wan_video_bundle_installed,
    WAN_VIDEO_BUNDLE,
)
from .metrics import (
    CompareResult,
    SanityResult,
    check_output_image,
    check_output_audio_with_thresholds,
    check_output_image_with_thresholds,
    check_output_video_with_thresholds,
    compare_images,
    compare_videos,
    hash_image,
)
from .heartmula_codec_parity import (
    resolve_codec_parity_manifest,
    run_codec_parity_check,
)

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
    """与 ``stem.validate_seedvr2_bundle`` 一致：扁平目录下两份 safetensors 齐全。"""
    base = model_id.split(":", 1)[0].strip()
    try:
        root = resolve_fp16_bundle_dir(base)
    except KeyError:
        return False
    if not root.is_dir():
        return False
    try:
        from backend.engine.families.seedvr2.stem import expected_seedvr2_weight_files
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
            result.product_ok = None
            result.product_reason = "danqing_generate_failed"
            self.results.append((case, result))
            return result

        product = check_output_image(our_path)
        result = compare_images(our_path, ref_path)
        result.product_ok = product.ok
        result.product_reason = product.reason if not product.ok else "ok"
        result.ours_time_sec = ours_time
        result.ref_time_sec = ref_time
        self.results.append((case, result))
        return result

    @staticmethod
    def grade_product(result: CompareResult) -> str:
        """PRODUCT_SKIP | PRODUCT_FAIL | PRODUCT_PASS — 丹青成片是否可用（非 PSNR）。"""
        if result.product_ok is None:
            return "PRODUCT_SKIP"
        return "PRODUCT_PASS" if result.product_ok else "PRODUCT_FAIL"

    @staticmethod
    def grade_parity(result: CompareResult) -> str:
        """PSNR 档位（与 mflux 参考图的像素级一致性）。"""
        return BenchmarkRunner.grade(result)

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
            "--guidance", str(case.guidance),
            "--output", str(output_path),
        ]
        if case.steps > 1:
            cmd += ["--steps", str(case.steps)]
        if getattr(case, "scheduler", None):
            cmd += ["--scheduler", str(case.scheduler)]
        if not getattr(case, "_mflux_omit_image_strength", False):
            cmd += ["--source-fidelity", str(case.image_strength)]
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
            src = str(_benchmark_source_path(case))
            if getattr(case, "_mflux_use_image_paths", False):
                cmd += ["--image-paths", src]
            else:
                cmd += ["--image-path", src]
            cmd += [
                "--prompt",
                case.prompt,
                "--guidance",
                str(case.guidance),
            ]
            if not getattr(case, "_mflux_omit_output_size", False):
                cmd += [
                    "--width",
                    str(case.width),
                    "--height",
                    str(case.height),
                ]
            if not getattr(case, "_mflux_omit_image_strength", False):
                cmd += ["--image-strength", str(case.image_strength)]
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
            parity = BenchmarkRunner.grade_parity(result)
            product = BenchmarkRunner.grade_product(result)
            prod_note = ""
            if product == "PRODUCT_FAIL":
                prod_note = f" product=FAIL({result.product_reason})"
            elif product == "PRODUCT_PASS":
                prod_note = " product=PASS"
            print(
                f"  {parity}: PSNR={result.psnr:.2f}dB SSIM={result.ssim:.4f} "
                f"max_diff={result.pixel_max_diff:.4f} mean_diff={result.pixel_mean_diff:.4f}"
                f"{prod_note}{time_info}"
            )
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

        product_fail = sum(
            1 for _, r in self.results if BenchmarkRunner.grade_product(r) == "PRODUCT_FAIL"
        )
        print(f"\n{'='*60}")
        print(
            f"Summary: {total} cases — "
            f"{passed} PASS / {warned} WARN / {failed} FAIL / {skipped} SKIP"
        )
        print(f"Product gate: {product_fail} PRODUCT_FAIL (exit code uses this, not PSNR alone)")
        if exempt_fail:
            ids = ", ".join(sorted(BENCHMARK_EXIT_EXEMPT_MISMATCH_VS_MFLUX))
            print(
                f"Parity gap (PSNR FAIL, product OK): {exempt_fail} case(s) "
                f"in {{{ids}}} — pixel match vs mflux still open"
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


class ExternalReferenceRunner(BenchmarkRunner):
    """Open-source reference parity (mlx-video / diffusers / custom command)."""

    def __init__(self, output_dir: str | Path = "tests/benchmark/outputs"):
        super().__init__(output_dir=output_dir)
        self.ref_results: list[tuple[ExternalRefCase, CompareResult]] = []

    def run_case(self, case: ExternalRefCase, *, run_ours: bool = True, run_ref: bool = True) -> CompareResult:
        media = (case.media or "image").strip().lower()
        ext = "mp4" if media == "video" else "png"
        ours_path = self.output_dir / f"{case.id}_danqing.{ext}"
        ref_path = self.output_dir / f"{case.id}_ref.{ext}"
        if case.action in ("rewrite", "upscale") or case.source_image:
            ensure_benchmark_source_image()

        ours_ok = False
        ref_ok = False
        ours_time = None
        ref_time = None
        if run_ours:
            t0 = time.time()
            ours_ok = self._run_danqing_external(case, ours_path)
            ours_time = time.time() - t0
        if run_ref:
            t0 = time.time()
            ref_ok = self._run_external_reference(case, ref_path)
            ref_time = time.time() - t0

        if not ref_ok:
            print(f"  [SKIP] {case.id}: 参考输出({case.ref_backend})生成失败")
            res = CompareResult(ours_time_sec=ours_time, ref_time_sec=ref_time)
            self.ref_results.append((case, res))
            return res
        if not ours_ok:
            print(f"  [SKIP] {case.id}: 丹青引擎输出生成失败")
            res = CompareResult(ref_hash=hash_image(ref_path), ref_time_sec=ref_time)
            res.product_ok = False
            res.product_reason = "danqing_generate_failed"
            self.ref_results.append((case, res))
            return res

        if media == "video":
            product = check_output_video_with_thresholds(ours_path, thresholds=None)
            res = compare_videos(ours_path, ref_path)
        else:
            product = check_output_image(ours_path)
            res = compare_images(ours_path, ref_path)
        res.product_ok = product.ok
        res.product_reason = product.reason if not product.ok else "ok"
        res.ours_time_sec = ours_time
        res.ref_time_sec = ref_time
        self.ref_results.append((case, res))
        return res

    def run_all(self, run_ours: bool = True, *, backend_filter: str = "") -> None:
        if backend_filter:
            cases = iter_external_ref_cases_by_backend(backend_filter)
            skipped = list_skipped_external_ref_cases_by_backend(backend_filter)
        else:
            cases = iter_external_ref_cases()
            skipped = list_skipped_external_ref_cases()
        total_registered = (
            len(list_external_ref_cases_by_backend(backend_filter))
            if backend_filter
            else len(list_external_ref_cases())
        )
        print(f"\n{'='*60}")
        print(
            f"DanQing OSS Reference Parity — {len(cases)} runnable / "
            f"{total_registered} registered"
        )
        print(f"Data root: {resolve_benchmark_data_root()}")
        if skipped:
            print(f"Skipped (no local bundle): {len(skipped)}")
        print(f"{'='*60}\n")
        for cid, reason in skipped:
            print(f"  [SKIP] {cid}: {reason}")

        for case in cases:
            print(f"[{case.id}] {case.description}")
            if case.media == "video":
                print(
                    f"  media=video model={case.model} seed={case.seed} steps={case.steps} "
                    f"frames={case.video_num_frames} size={case.video_size} ref={case.ref_backend}"
                )
            else:
                print(
                    f"  media=image model={case.model} seed={case.seed} steps={case.steps} "
                    f"size={case.width}x{case.height} ref={case.ref_backend}"
                )
            result = self.run_case(case, run_ours=run_ours, run_ref=True)
            self._print_result(result)
        self._print_summary()

    def _run_danqing_external(self, case: ExternalRefCase, output_path: Path) -> bool:
        if case.media == "video":
            return self._run_danqing_video_generate(case, output_path, timeout_sec=case.timeout_sec)
        bc = BenchmarkCase(
            id=case.id,
            model=case.model,
            action=case.action,
            prompt=case.prompt,
            seed=case.seed,
            width=case.width,
            height=case.height,
            steps=case.steps,
            guidance=case.guidance,
            negative_prompt=case.negative_prompt,
            source_image=case.source_image,
            image_strength=case.image_strength,
            _mflux_model_flag="__EXTERNAL_REF__",
        )
        return self._run_danqing(bc, output_path)

    def _run_danqing_video_generate(
        self, case: ExternalRefCase, output_path: Path, *, timeout_sec: int | None = None
    ) -> bool:
        cli = PROJECT_ROOT / "bin" / "danqing-video-generate"
        cmd = [
            sys.executable,
            str(cli),
            "--model",
            case.model,
            "--prompt",
            case.prompt,
            "--size",
            case.video_size,
            "--num-frames",
            str(int(case.video_num_frames)),
            "--fps",
            str(int(case.video_fps)),
            "--steps",
            str(int(case.steps)),
            "--guidance",
            str(float(case.guidance)),
            "--seed",
            str(case.seed),
            "--output",
            str(output_path),
        ]
        if case.negative_prompt:
            cmd.extend(["--negative-prompt", case.negative_prompt])
        if case.shift is not None:
            cmd.extend(["--shift", str(float(case.shift))])
        tout = int(timeout_sec) if timeout_sec is not None else DEFAULT_DANQING_CLI_TIMEOUT_SEC
        try:
            env = os.environ.copy()
            env.setdefault("MLX_METAL_DEVICE_ONLY", "1")
            env.setdefault("MLX_METAL_MEMORY_LIMIT", "120")
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=tout,
                env=env,
                cwd=str(PROJECT_ROOT),
            )
            if proc.returncode != 0:
                print(f"    [danqing-video] 失败 (exit={proc.returncode})")
                if proc.stderr:
                    print(f"    [danqing-video] stderr: {proc.stderr[:STDERR_HEAD_CHARS]}")
                return False
            print(f"    [danqing-video] 输出: {output_path}")
            return True
        except subprocess.TimeoutExpired:
            print(f"    [danqing-video] 超时 ({tout}s)")
            return False
        except FileNotFoundError:
            print(f"    [danqing-video] 未找到 CLI: {cli}")
            return False

    def _run_external_reference(self, case: ExternalRefCase, output_path: Path) -> bool:
        backend = (case.ref_backend or "").strip().lower()
        if backend == "mlx_video":
            return self._run_ref_mlx_video(case, output_path)
        if backend == "mflux":
            return self._run_ref_mflux(case, output_path)
        if backend == "diffusers":
            return self._run_ref_diffusers(case, output_path)
        if backend == "custom":
            return self._run_ref_custom(case, output_path)
        print(f"    [ref] 未知 backend: {case.ref_backend!r}")
        return False

    def _run_ref_mflux(self, case: ExternalRefCase, output_path: Path) -> bool:
        if case.media != "image":
            print("    [ref-mflux] 目前仅支持 image")
            return False
        bc = BenchmarkCase(
            id=case.id,
            model=case.model,
            action=case.action,
            prompt=case.prompt,
            seed=case.seed,
            width=case.width,
            height=case.height,
            steps=case.steps,
            guidance=case.guidance,
            negative_prompt=case.negative_prompt,
            source_image=case.source_image or SRC_IMAGE,
            image_strength=case.image_strength,
            _mflux_cli=case.ref_mflux_cli or "mflux-generate",
            _mflux_model_flag="",
        )
        return self._run_mflux(bc, output_path)

    def _run_ref_custom(self, case: ExternalRefCase, output_path: Path) -> bool:
        if not case.ref_cmd_template:
            print("    [ref] custom backend 需要 ref_cmd_template")
            return False
        return self._run_ref_template_cmd(case, output_path, default_cmd="")

    def _run_ref_mlx_video(self, case: ExternalRefCase, output_path: Path) -> bool:
        default_cmd = os.getenv("DANQING_BENCH_MLX_VIDEO_CMD", "mlx-video-generate")
        if not case.ref_cmd_template:
            print("    [ref] mlx_video backend 缺少 ref_cmd_template")
            return False
        return self._run_ref_template_cmd(case, output_path, default_cmd=default_cmd)

    def _run_ref_template_cmd(self, case: ExternalRefCase, output_path: Path, *, default_cmd: str) -> bool:
        model_dir = ""
        if case.local_bundle_rel:
            model_dir = str((resolve_benchmark_data_root() / case.local_bundle_rel).resolve())
        shift_s = str(float(case.shift)) if case.shift is not None else ""
        fmt = {
            "cmd": default_cmd,
            "python": sys.executable,
            "model": case.model,
            "ref_model": case.ref_model or case.model,
            "model_dir": model_dir,
            "prompt": case.prompt,
            "seed": str(case.seed),
            "steps": str(int(case.steps)),
            "guidance": str(float(case.guidance)),
            "shift": shift_s,
            "negative_prompt": case.negative_prompt,
            "width": str(int(case.width)),
            "height": str(int(case.height)),
            "video_size": case.video_size,
            "num_frames": str(int(case.video_num_frames)),
            "fps": str(int(case.video_fps)),
            "source_image": str(
                _benchmark_source_path(
                    BenchmarkCase(
                        id=case.id,
                        model=case.model,
                        action=case.action,
                        source_image=case.source_image or SRC_IMAGE,
                        _mflux_model_flag="__EXTERNAL_REF__",
                    )
                )
            ),
            "output": str(output_path),
        }
        try:
            cmd_text = case.ref_cmd_template.format(**fmt).strip()
        except KeyError as e:
            print(f"    [ref] 模板变量缺失: {e}")
            return False
        if not cmd_text:
            print("    [ref] 空命令模板")
            return False
        cmd = shlex.split(cmd_text)
        tout = int(case.timeout_sec or DEFAULT_DANQING_CLI_TIMEOUT_SEC)
        env = os.environ.copy()
        env.setdefault("MLX_METAL_DEVICE_ONLY", "1")
        env.setdefault("MLX_METAL_MEMORY_LIMIT", "120")
        env["PYTHONUNBUFFERED"] = "1"
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=tout,
                env=env,
                cwd=str(PROJECT_ROOT),
            )
            if proc.returncode != 0:
                print(f"    [ref] 命令失败 (exit={proc.returncode})")
                if proc.stderr:
                    print(f"    [ref] stderr: {proc.stderr[:STDERR_HEAD_CHARS]}")
                return False
            print(f"    [ref] 输出: {output_path}")
            return True
        except subprocess.TimeoutExpired:
            print(f"    [ref] 超时 ({tout}s)")
            return False
        except FileNotFoundError:
            print(f"    [ref] 未找到命令: {cmd[0] if cmd else default_cmd}")
            return False

    def _run_ref_diffusers(self, case: ExternalRefCase, output_path: Path) -> bool:
        if case.media != "image":
            print("    [ref] diffusers backend 目前仅支持 image")
            return False
        if case.action != "create":
            print("    [ref] diffusers backend 当前仅支持 create")
            return False
        ref_model = (case.ref_model or "").strip()
        if not ref_model:
            print("    [ref] diffusers backend 缺少 ref_model")
            return False
        lora_path = ""
        if case.ref_lora_rel:
            lora_abs = (resolve_benchmark_data_root() / case.ref_lora_rel).resolve()
            if not lora_abs.exists():
                print(f"    [ref-diffusers] LoRA 不存在: {lora_abs}")
                return False
            lora_path = str(lora_abs)
        script_lines = [
            "import torch",
            "from diffusers import DiffusionPipeline",
            (
                "pipe=DiffusionPipeline.from_pretrained("
                f"{ref_model!r}, torch_dtype=torch.bfloat16 if hasattr(torch,'bfloat16') else torch.float32)"
            ),
            "pipe=pipe.to('cpu')",
        ]
        if lora_path:
            script_lines.append(f"pipe.load_lora_weights({lora_path!r})")
        script_lines.extend(
            [
                (
                    f"img=pipe(prompt={case.prompt!r}, width={int(case.width)}, height={int(case.height)}, "
                    f"num_inference_steps={int(case.steps)}, guidance_scale={float(case.guidance)}).images[0]"
                ),
                f"img.save({str(output_path)!r})",
                "print('saved', " + repr(str(output_path)) + ")",
            ]
        )
        script = "\n".join(script_lines) + "\n"
        cmd = [sys.executable, "-c", script]
        tout = int(case.timeout_sec or 3600)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=tout,
                env=env,
                cwd=str(PROJECT_ROOT),
            )
            if proc.returncode != 0:
                print(f"    [ref-diffusers] 失败 (exit={proc.returncode})")
                if proc.stderr:
                    print(f"    [ref-diffusers] stderr: {proc.stderr[:STDERR_HEAD_CHARS]}")
                return False
            return output_path.is_file()
        except subprocess.TimeoutExpired:
            print(f"    [ref-diffusers] 超时 ({tout}s)")
            return False
        except FileNotFoundError:
            print("    [ref-diffusers] Python 不可用")
            return False

    def _print_summary(self) -> None:
        passed = sum(1 for _, r in self.ref_results if BenchmarkRunner.grade(r) == "PASS")
        warned = sum(1 for _, r in self.ref_results if BenchmarkRunner.grade(r) == "WARN")
        failed = sum(1 for _, r in self.ref_results if BenchmarkRunner.grade(r) == "FAIL")
        skipped = sum(1 for _, r in self.ref_results if BenchmarkRunner.grade(r) == "SKIP")
        total = len(self.ref_results)
        product_fail = sum(
            1 for _, r in self.ref_results if BenchmarkRunner.grade_product(r) == "PRODUCT_FAIL"
        )
        print(f"\n{'='*60}")
        print(
            f"OSSRef summary: {total} cases — "
            f"{passed} PASS / {warned} WARN / {failed} FAIL / {skipped} SKIP"
        )
        print(f"Product gate: {product_fail} PRODUCT_FAIL (exit code uses this)")
        print(f"{'='*60}")
        for case, r in self.ref_results:
            if r.psnr is not None:
                print(f"  [{case.id}] {BenchmarkRunner.grade(r)} PSNR={r.psnr:.1f}dB SSIM={r.ssim:.4f}")
            else:
                print(f"  [{case.id}] SKIP")


class SanitySuiteRunner(BenchmarkRunner):
    """无 mflux 参考：仅生成 + 像素健全性（见 ``sanity.check_output_image``）。"""

    def __init__(self, output_dir: str | Path = "tests/benchmark/outputs"):
        super().__init__(output_dir=output_dir)
        self.sanity_results: list[tuple[SanityCase, SanityResult]] = []

    @staticmethod
    def _score_text(res: SanityResult) -> str:
        if res.score <= 0:
            return ""
        if not res.subscores:
            return f" score={res.score:.1f}"
        i = float(res.subscores.get("integrity", 0.0))
        a = float(res.subscores.get("anti_garbage", 0.0))
        s = float(res.subscores.get("semantic_proxy", 0.0))
        return f" score={res.score:.1f} (I={i:.1f},A={a:.1f},S={s:.1f})"

    @staticmethod
    def _apply_semantic_gate(case: SanityCase, res: SanityResult, out_path: Path, media: str) -> SanityResult:
        if not case.semantic_gate_enabled or res.skipped or not res.ok:
            return res
        try:
            sem_score = score_semantic_alignment(
                media=media if media in ("image", "video", "audio") else "image",
                path=out_path,
                prompt=case.prompt,
                backend=case.semantic_backend,
                model_id=case.semantic_model_id,
            )
        except Exception as e:
            res.ok = False
            res.reason = f"semantic_gate_error:{e}"
            return res

        subs = dict(res.subscores or {})
        integrity = float(subs.get("integrity", 0.0))
        anti_garbage = float(subs.get("anti_garbage", 0.0))
        subs["semantic_proxy"] = float(np.clip(sem_score, 0.0, 100.0))
        subs["semantic_model"] = float(np.clip(sem_score, 0.0, 100.0))
        if integrity > 0.0 or anti_garbage > 0.0:
            res.score = float(np.clip(0.2 * integrity + 0.5 * anti_garbage + 0.3 * subs["semantic_proxy"], 0.0, 100.0))
        else:
            res.score = float(np.clip(0.7 * res.score + 0.3 * subs["semantic_proxy"], 0.0, 100.0))
        res.subscores = subs
        if sem_score < float(case.semantic_min_score):
            res.ok = False
            res.reason = (
                f"semantic_low(score={sem_score:.1f},"
                f"min={float(case.semantic_min_score):.1f},backend={case.semantic_backend or 'auto'})"
            )
        return res

    def _apply_codec_parity_gate(self, case: SanityCase, res: SanityResult) -> SanityResult:
        manifest_rel = (case.codec_parity_manifest or "").strip()
        if not manifest_rel:
            return res
        manifest_path = resolve_codec_parity_manifest(
            manifest_rel,
            project_root=PROJECT_ROOT,
        )
        parity = run_codec_parity_check(
            manifest_path,
            output_dir=self.output_dir,
            case_id=case.id,
            min_si_sdr_db=float(case.codec_parity_min_si_sdr_db),
            min_correlation=float(case.codec_parity_min_correlation),
            warn_si_sdr_db=float(case.codec_parity_warn_si_sdr_db),
        )
        if parity.skipped:
            print(f"    [codec-parity SKIP] {parity.reason}")
            return res
        tag = "PASS" if parity.ok else "FAIL"
        if parity.reason.startswith("codec_parity_warn"):
            tag = "WARN"
        print(
            f"    [codec-parity {tag}] {parity.reason} "
            f"(si_sdr={parity.mean_luma:.2f}dB corr={parity.std_luma:.4f})"
        )
        if not parity.ok:
            return parity
        return res

    @staticmethod
    def _audio_model_base(case: SanityCase) -> str:
        return case.model.split(":", 1)[0].strip()

    def _run_danqing_video_generate(
        self, case: SanityCase, output_path: Path, *, timeout_sec: int | None = None
    ) -> bool:
        cli = PROJECT_ROOT / "bin" / "danqing-video-generate"
        cmd = [
            sys.executable,
            str(cli),
            "--model",
            case.model,
            "--prompt",
            case.prompt,
            "--size",
            case.video_size,
            "--num-frames",
            str(int(case.video_num_frames)),
            "--fps",
            str(int(case.video_fps)),
            "--steps",
            str(int(case.steps)),
            "--guidance",
            str(float(case.guidance)),
            "--seed",
            str(case.seed),
            "--output",
            str(output_path),
        ]
        if case.negative_prompt:
            cmd.extend(["--negative-prompt", case.negative_prompt])
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
                print(f"    [danqing-video] 失败 (exit={proc.returncode})")
                if proc.stderr:
                    print(f"    [danqing-video] stderr: {proc.stderr[:STDERR_HEAD_CHARS]}")
                return False
            print(f"    [danqing-video] 生成完成 ({elapsed:.1f}s)")
            print(f"    [danqing-video] 输出: {output_path}")
            return True
        except subprocess.TimeoutExpired:
            print(f"    [danqing-video] 超时 ({tout}s)")
            return False
        except FileNotFoundError:
            print(f"    [danqing-video] 未找到 CLI: {cli}")
            return False

    def _run_danqing_audio_generate(
        self, case: SanityCase, output_path: Path, *, timeout_sec: int | None = None
    ) -> bool:
        cli = PROJECT_ROOT / "bin" / "danqing-audio-generate"
        base = self._audio_model_base(case)
        is_heartmula = base.startswith("heartmula")
        cmd = [
            sys.executable,
            str(cli),
            "--model",
            case.model,
            "--prompt",
            case.prompt,
            "--duration",
            str(int(case.duration)),
            "--guidance",
            str(float(case.guidance)),
            "--seed",
            str(case.seed),
            "--n",
            "1",
            "--audio-format",
            case.audio_format or "wav",
            "--output",
            str(output_path),
        ]
        if case.instrumental:
            cmd.append("--instrumental")
        cmd.extend(["--lyrics", case.lyrics or ""])
        if not is_heartmula and case.steps is not None:
            cmd.extend(["--steps", str(int(case.steps))])
        if case.temperature is not None:
            cmd.extend(["--temperature", str(float(case.temperature))])
        if case.top_k is not None:
            cmd.extend(["--top-k", str(int(case.top_k))])
        if case.codec_steps is not None:
            cmd.extend(["--codec-steps", str(int(case.codec_steps))])
        if case.codec_guidance is not None:
            cmd.extend(["--codec-guidance", str(float(case.codec_guidance))])
        if case.long_form_temperature is not None:
            cmd.extend(["--long-form-temperature", str(float(case.long_form_temperature))])
        if case.long_form_topk is not None:
            cmd.extend(["--long-form-topk", str(int(case.long_form_topk))])
        return self._run_danqing_audio_cmd(cmd, case, timeout_sec=timeout_sec, label="danqing-audio")

    def _run_danqing_audio_edit(
        self, case: SanityCase, output_path: Path, *, timeout_sec: int | None = None
    ) -> bool:
        from tests.benchmark.cases import ensure_ace_step_cover_source

        cli = PROJECT_ROOT / "bin" / "danqing-audio-edit"
        src = case.source_audio or ""
        if src and not Path(src).is_absolute():
            src = str(PROJECT_ROOT / src)
        src_path = ensure_ace_step_cover_source() if not src else Path(src)
        if not src_path.is_file():
            print(f"    [danqing-audio-edit] 缺少参考音频: {src_path}")
            return False
        cmd = [
            sys.executable,
            str(cli),
            "--model",
            case.model,
            "--operation",
            "cover",
            "--source-audio",
            str(src_path),
            "--prompt",
            case.prompt or "",
            "--source-fidelity",
            str(float(case.source_fidelity)),
            "--seed",
            str(case.seed),
            "--n",
            "1",
            "--audio-format",
            case.audio_format or "wav",
            "--output",
            str(output_path),
        ]
        return self._run_danqing_audio_cmd(cmd, case, timeout_sec=timeout_sec, label="danqing-audio-edit")

    def _run_danqing_audio_cmd(
        self,
        cmd: list,
        case: SanityCase,
        *,
        timeout_sec: int | None,
        label: str,
    ) -> bool:
        base = self._audio_model_base(case)
        is_heartmula = base.startswith("heartmula")
        try:
            t0 = time.time()
            env = os.environ.copy()
            env.setdefault("MLX_METAL_DEVICE_ONLY", "1")
            env.setdefault("MLX_METAL_MEMORY_LIMIT", "120")
            env["PYTHONUNBUFFERED"] = "1"
            if case.id.endswith("-cuda-sanity"):
                env["DANQING_FORCE_AUDIO_BACKEND"] = "cuda"
            if not is_heartmula:
                if not case.ace_step_use_lm:
                    env["ACESTEP_USE_LM"] = "0"
                else:
                    env.setdefault("ACESTEP_USE_LM", "1")
                    env.setdefault("ACESTEP_LM_CODES", "1")
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
                print(f"    [{label}] 失败 (exit={proc.returncode})")
                if proc.stderr:
                    print(f"    [{label}] stderr: {proc.stderr[:STDERR_HEAD_CHARS]}")
                return False
            print(f"    [{label}] 生成完成 ({elapsed:.1f}s)")
            return True
        except subprocess.TimeoutExpired:
            print(f"    [{label}] 超时 ({tout}s)")
            return False
        except FileNotFoundError:
            print(f"    [{label}] 未找到 CLI")
            return False

    def run_one_sanity(self, case: SanityCase) -> SanityResult:
        media = (case.media or "image").strip().lower()
        ext = "mp4" if media == "video" else ("wav" if media == "audio" else "png")
        out_path = self.output_dir / f"{case.id}_sanity.{ext}"
        tout = case.timeout_sec if case.timeout_sec is not None else DEFAULT_DANQING_CLI_TIMEOUT_SEC

        if media == "video":
            if not mlx_runtime_available():
                print("    [SKIP] MLX 不可用，跳过 Wan 视频健全性")
                res = SanityResult(
                    ok=True,
                    reason="skip_no_mlx",
                    mean_luma=0.0,
                    std_luma=0.0,
                    entropy_bits=0.0,
                    laplacian_var=0.0,
                    skipped=True,
                )
                self.sanity_results.append((case, res))
                print(f"  SKIP: {res.reason}")
                return res
            if not wan_video_bundle_installed():
                print(
                    "    [SKIP] Wan bundle 未安装"
                    f"（见 {WAN_VIDEO_BUNDLE}）"
                )
                res = SanityResult(
                    ok=True,
                    reason="skip_missing_wan_bundle",
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
            ok = self._run_danqing_video_generate(case, out_path, timeout_sec=tout)
            elapsed = time.time() - t0
            if not ok:
                res = SanityResult(
                    ok=False,
                    reason="danqing_video_cli_failed",
                    mean_luma=0.0,
                    std_luma=0.0,
                    entropy_bits=0.0,
                    laplacian_var=0.0,
                )
                self.sanity_results.append((case, res))
                print(f"  FAIL: {res.reason} ({elapsed:.1f}s)")
                return res
            res = check_output_video_with_thresholds(
                out_path,
                thresholds=case.video_quality_thresholds or None,
            )
            res = self._apply_semantic_gate(case, res, out_path, media="video")
            self.sanity_results.append((case, res))
            status = "PASS" if res.ok else "FAIL"
            detail = res.reason if not res.ok else (
                f"frame_std={res.std_luma:.4f} mean_luma={res.mean_luma:.3f}"
            )
            detail += self._score_text(res)
            label = "BASELINE" if case.is_timing_baseline else status
            print(f"  {label}: {detail} ({elapsed:.1f}s)")
            if case.is_timing_baseline:
                print(
                    f"    [BASELINE] model={case.model} steps={case.steps} "
                    f"frames={case.video_num_frames} size={case.video_size} "
                    f"total_sec={elapsed:.1f}"
                )
            print(f"    output: {out_path}")
            return res

        if media == "audio":
            if case.id.endswith("-cuda-sanity") and not cuda_runtime_available():
                print("    [SKIP] CUDA 不可用，跳过 ACE-Step CUDA 健全性")
                res = SanityResult(
                    ok=True,
                    reason="skip_no_cuda",
                    mean_luma=0.0,
                    std_luma=0.0,
                    entropy_bits=0.0,
                    laplacian_var=0.0,
                    skipped=True,
                )
                self.sanity_results.append((case, res))
                print(f"  SKIP: {res.reason}")
                return res
            base = self._audio_model_base(case)
            if base.startswith("heartmula"):
                if not mlx_runtime_available():
                    print("    [SKIP] MLX 不可用，跳过 HeartMuLa 健全性")
                    res = SanityResult(
                        ok=True,
                        reason="skip_no_mlx",
                        mean_luma=0.0,
                        std_luma=0.0,
                        entropy_bits=0.0,
                        laplacian_var=0.0,
                        skipped=True,
                    )
                    self.sanity_results.append((case, res))
                    print(f"  SKIP: {res.reason}")
                    return res
                if not heartmula_bundle_installed():
                    print(
                        "    [SKIP] HeartMuLa bundle 未安装"
                        "（见 models/Audio/heartmula-oss-3b-happy-new-year）"
                    )
                    res = SanityResult(
                        ok=True,
                        reason="skip_missing_heartmula_bundle",
                        mean_luma=0.0,
                        std_luma=0.0,
                        entropy_bits=0.0,
                        laplacian_var=0.0,
                        skipped=True,
                    )
                    self.sanity_results.append((case, res))
                    print(f"  SKIP: {res.reason}")
                    return res
            elif base.startswith("ace-step"):
                if not ace_step_bundle_installed():
                    print("    [SKIP] ACE-Step bundle 未安装（见 models/Audio/acestep-v15-xl-sft）")
                    res = SanityResult(
                        ok=True,
                        reason="skip_missing_ace_step_bundle",
                        mean_luma=0.0,
                        std_luma=0.0,
                        entropy_bits=0.0,
                        laplacian_var=0.0,
                        skipped=True,
                    )
                    self.sanity_results.append((case, res))
                    print(f"  SKIP: {res.reason}")
                    return res
            else:
                print(f"    [SKIP] 未知音频模型 {base!r}")
                res = SanityResult(
                    ok=True,
                    reason="skip_unknown_audio_model",
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
            if (case.audio_operation or "create") == "cover":
                ok = self._run_danqing_audio_edit(case, out_path, timeout_sec=tout)
            else:
                ok = self._run_danqing_audio_generate(case, out_path, timeout_sec=tout)
            elapsed = time.time() - t0
            if not ok:
                res = SanityResult(
                    ok=False,
                    reason="danqing_audio_cli_failed",
                    mean_luma=0.0,
                    std_luma=0.0,
                    entropy_bits=0.0,
                    laplacian_var=0.0,
                )
                if base.startswith("heartmula"):
                    res = self._apply_codec_parity_gate(case, res)
                self.sanity_results.append((case, res))
                print(f"  FAIL: {res.reason} ({elapsed:.1f}s)")
                return res
            res = check_output_audio_with_thresholds(
                out_path,
                thresholds=case.audio_quality_thresholds or None,
            )
            res = self._apply_semantic_gate(case, res, out_path, media="audio")
            res = self._apply_codec_parity_gate(case, res)
            self.sanity_results.append((case, res))
            status = "PASS" if res.ok else "FAIL"
            detail = res.reason if not res.ok else (
                f"rms={res.mean_luma:.4f} peak={res.std_luma:.4f}"
            )
            detail += self._score_text(res)
            print(f"  {status}: {detail} ({elapsed:.1f}s)")
            print(f"    output: {out_path}")
            return res

        bc = case.as_benchmark_case()
        ensure_benchmark_source_image()

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
        res = check_output_image_with_thresholds(
            out_path,
            thresholds=case.image_quality_thresholds or None,
        )
        res = self._apply_semantic_gate(case, res, out_path, media="image")
        self.sanity_results.append((case, res))
        status = "PASS" if res.ok else "FAIL"
        detail = res.reason if not res.ok else (
            f"std={res.std_luma:.4f} entropy_bits={res.entropy_bits:.2f} lap_var={res.laplacian_var:.2f}"
        )
        detail += self._score_text(res)
        print(f"  {status}: {detail} ({elapsed:.1f}s)")
        print(f"    output: {out_path}")
        return res

    def run_all_sanity_cases(self) -> None:
        print(f"\n{'='*60}")
        print(f"DanQing Output Sanity — {len(ALL_SANITY_CASES)} cases (no mflux reference)")
        print(f"{'='*60}\n")
        for case in ALL_SANITY_CASES:
            print(f"[{case.id}] {case.description}")
            media = (case.media or "image").strip().lower()
            if media == "video":
                print(
                    f"  model={case.model} seed={case.seed} steps={case.steps} "
                    f"frames={case.video_num_frames} size={case.video_size} "
                    f"baseline={case.is_timing_baseline}"
                )
            elif media == "audio":
                base = self._audio_model_base(case)
                if base.startswith("heartmula"):
                    print(
                        f"  model={case.model} seed={case.seed} duration={case.duration}s "
                        f"guidance={case.guidance} temperature={case.temperature} top_k={case.top_k}"
                    )
                    if case.codec_parity_manifest:
                        print(
                            f"  codec_parity: manifest={case.codec_parity_manifest} "
                            f"min_si_sdr={case.codec_parity_min_si_sdr_db}dB"
                        )
                else:
                    op = case.audio_operation or "create"
                    print(
                        f"  model={case.model} op={op} seed={case.seed} steps={case.steps} "
                        f"duration={case.duration}s guidance={case.guidance} lm={case.ace_step_use_lm}"
                    )
            else:
                print(
                    f"  model={case.model} seed={case.seed} steps={case.steps} "
                    f"size={case.width}x{case.height} guidance={case.guidance}"
                )
            if case.semantic_gate_enabled:
                backend = case.semantic_backend or ("clap" if media == "audio" else "clip")
                model_id = case.semantic_model_id or "(default)"
                print(
                    f"  semantic_gate: enabled backend={backend} "
                    f"min_score={case.semantic_min_score:.1f} model={model_id}"
                )
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
                extra += self._score_text(r)
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
    """Mflux suite: exit 1 when 丹青成片健全性失败；PSNR 仅作 parity 报告。"""
    runner = BenchmarkRunner(output_dir=output_dir)
    if not case_id:
        runner.run_all(run_ours=True)
        product_failed = sum(
            1 for _, r in runner.results if BenchmarkRunner.grade_product(r) == "PRODUCT_FAIL"
        )
        return 1 if product_failed else 0
    case = get_case(case_id)
    if case is None:
        print(f"Unknown mflux case: {case_id}")
        print(f"Available: {list_cases()}")
        return 2
    print(f"[{case.id}] {case.description}")
    result = runner.run_case(case, run_ours=True, run_ref=True)
    runner._print_result(result)
    print(f"\nOutputs: {runner.output_dir}")
    if BenchmarkRunner.grade_product(result) == "PRODUCT_FAIL":
        return 1
    return 0


def run_reference(case_id: str = "", output_dir: str = "tests/benchmark/outputs", *, backend_filter: str = "") -> int:
    """Reference parity suite (mlx-video or diffusers, selected by backend_filter)."""
    runner = ExternalReferenceRunner(output_dir=output_dir)
    if not case_id:
        runner.run_all(run_ours=True, backend_filter=backend_filter)
        product_failed = sum(
            1 for _, r in runner.ref_results if BenchmarkRunner.grade_product(r) == "PRODUCT_FAIL"
        )
        return 1 if product_failed else 0
    case = get_external_ref_case(case_id)
    if case is None:
        ids = list_external_ref_cases_by_backend(backend_filter) if backend_filter else list_external_ref_cases()
        print(f"Unknown reference case: {case_id}")
        print(f"Available: {ids}")
        return 2
    if backend_filter and (case.ref_backend or "").strip().lower() != backend_filter.strip().lower():
        ids = list_external_ref_cases_by_backend(backend_filter)
        print(f"Case {case_id} is not in backend suite {backend_filter!r}")
        print(f"Available: {ids}")
        return 2
    print(f"[{case.id}] {case.description}")
    result = runner.run_case(case, run_ours=True, run_ref=True)
    runner._print_result(result)
    print(f"\nOutputs: {runner.output_dir}")
    if BenchmarkRunner.grade_product(result) == "PRODUCT_FAIL":
        return 1
    return 0


def run_mlx_video(case_id: str = "", output_dir: str = "tests/benchmark/outputs") -> int:
    """mlx-video reference parity suite."""
    return run_reference(case_id=case_id, output_dir=output_dir, backend_filter="mlx_video")


def run_diffusers(case_id: str = "", output_dir: str = "tests/benchmark/outputs") -> int:
    """diffusers reference parity suite."""
    return run_reference(case_id=case_id, output_dir=output_dir, backend_filter="diffusers")
