"""Benchmark: ``mflux`` + ``sanity`` + ``mlx-video`` + ``diffusers`` suites."""
from .cases import (
    ALL_CASES,
    ALL_EXTERNAL_REF_CASES,
    BenchmarkCase,
    ExternalRefCase,
    get_case,
    get_external_ref_case,
    list_cases,
    list_external_ref_cases,
    list_sanity_cases,
)
from .metrics import CompareResult, SanityResult, compare_images, compare_videos, hash_image
from .run import BenchmarkRunner, run_diffusers, run_mflux, run_mlx_video, run_sanity

__all__ = [
    "BenchmarkCase",
    "ExternalRefCase",
    "ALL_CASES",
    "ALL_EXTERNAL_REF_CASES",
    "get_case",
    "get_external_ref_case",
    "list_cases",
    "list_external_ref_cases",
    "list_sanity_cases",
    "CompareResult",
    "SanityResult",
    "compare_images",
    "compare_videos",
    "hash_image",
    "BenchmarkRunner",
    "run_mflux",
    "run_mlx_video",
    "run_diffusers",
    "run_sanity",
]
