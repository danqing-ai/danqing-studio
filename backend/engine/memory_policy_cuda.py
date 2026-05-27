"""CUDA cache release hook for memory_policy."""
from __future__ import annotations


def clear_cuda_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
