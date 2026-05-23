"""Resolve PyTorch inference device for text-encoder sidecars (MLX desktop + Mac)."""
from __future__ import annotations


def resolve_torch_inference_device(preference: str = "auto") -> str:
    """Pick ``mps`` on Apple Silicon when available, else ``cpu``.

    ``preference`` may be ``auto``, ``cpu``, ``mps``, or ``cuda``.
    Unknown values fail loud.
    """
    pref = (preference or "auto").strip().lower()
    if pref in ("cpu", "mps", "cuda"):
        return pref
    if pref != "auto":
        raise RuntimeError(
            f"Unknown torch text-encoder device preference: {preference!r} "
            "(expected auto, cpu, mps, or cuda)."
        )
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "PyTorch is required for HunyuanVideo / Qwen2.5-VL text encoding but is not installed."
        ) from e
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
