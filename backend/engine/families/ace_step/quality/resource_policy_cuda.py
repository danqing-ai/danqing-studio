"""ACE-Step CUDA memory detection helpers."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def detect_cuda_memory_gb() -> float | None:
    """Return total CUDA device memory in GB, or ``None`` when unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return float(props.total_memory) / (1024**3)
    except Exception as exc:
        logger.warning("CUDA memory detection failed: %s; falling back to default tier", exc)
    return None
