"""
DiffRhythm 2 CFM/DiT — PyTorch / CUDA (integration pending).

Upstream wraps Llama-NAR DiT inside ``diffrhythm2.cfm.CFM`` for block flow matching.
"""
from __future__ import annotations

import logging
import math
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DiffRhythmDiTCuda:
    """DiffRhythm DiT decoder — PyTorch / CUDA implementation.

    This is a placeholder implementation that delegates to the upstream
    DiffRhythm model when available. The structure mirrors the MLX version
    for weight-loading compatibility.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 32,
        head_dim: int = 64,
        intermediate_size: int = 8192,
        in_channels: int = 64,
        out_channels: int = 64,
        cross_attention_dim: int = 768,
        max_position_embeddings: int = 32768,
        rope_theta: float = 10_000.0,
        qk_norm: bool = True,
        patch_size: int = 1,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self._model: Any = None

    def _ensure_model(self):
        """Lazy-load the PyTorch model."""
        if self._model is not None:
            return
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise RuntimeError("DiffRhythm CUDA requires PyTorch")

        # Placeholder: upstream DiffRhythm would provide the actual model class.
        # For now, we create a minimal compatible structure.
        self._model = _DiffRhythmDiTTorch(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            head_dim=self.head_dim,
            intermediate_size=intermediate_size,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
        )

    def __call__(
        self,
        hidden_states: Any,
        timestep: Any,
        encoder_hidden_states: Any,
        encoder_attention_mask: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        self._ensure_model()
        return self._model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            **kwargs,
        )

    def parameters(self):
        self._ensure_model()
        return list(self._model.named_parameters())


class _DiffRhythmDiTTorch:
    """Minimal PyTorch DiT compatible with DiffRhythm weight layout.

    Full implementation should be replaced with upstream DiffRhythm model
    when integrated.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 32,
        head_dim: int = 64,
        intermediate_size: int = 8192,
        in_channels: int = 64,
        out_channels: int = 64,
    ):
        import torch.nn as nn

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.proj_in = nn.Linear(in_channels, hidden_size, bias=False)
        self.proj_out = nn.Linear(hidden_size, out_channels, bias=False)

        # Store config for parameter enumeration
        self._config = {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "head_dim": head_dim,
            "intermediate_size": intermediate_size,
        }

    def __call__(self, **kwargs: Any) -> Any:
        import torch

        hidden_states = kwargs["hidden_states"]
        # Minimal forward: project in, project out
        x = self.proj_in(hidden_states)
        x = self.proj_out(x)
        return x

    def named_parameters(self):
        import torch.nn as nn

        result = {}
        for name, param in self.proj_in.named_parameters():
            result[f"proj_in.{name}"] = param
        for name, param in self.proj_out.named_parameters():
            result[f"proj_out.{name}"] = param
        return result
