"""Benchmark: ``mflux`` PSNR suite + ``sanity`` output checks."""
from .cases import ALL_CASES, BenchmarkCase, get_case, list_cases, list_sanity_cases
from .metrics import CompareResult, SanityResult, compare_images, hash_image
from .run import BenchmarkRunner, run_mflux, run_sanity

__all__ = [
    "BenchmarkCase",
    "ALL_CASES",
    "get_case",
    "list_cases",
    "list_sanity_cases",
    "CompareResult",
    "SanityResult",
    "compare_images",
    "hash_image",
    "BenchmarkRunner",
    "run_mflux",
    "run_sanity",
]
