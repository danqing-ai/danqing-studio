"""基准测试框架"""
from .cases import BenchmarkCase, ALL_CASES, get_case, list_cases
from .compare import CompareResult, compare_images, compute_psnr, compute_ssim, hash_image
from .runner import BenchmarkRunner, run_benchmark

__all__ = [
    "BenchmarkCase", "ALL_CASES", "get_case", "list_cases",
    "CompareResult", "compare_images", "compute_psnr", "compute_ssim", "hash_image",
    "BenchmarkRunner", "run_benchmark",
]
