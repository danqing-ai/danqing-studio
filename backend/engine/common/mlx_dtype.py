"""MLX dtype helpers shared by text encoders."""
from __future__ import annotations

from typing import Any

import mlx.core as mx


def cast_floating_mx_tree(obj: Any, dtype: mx.Dtype) -> Any:
    if isinstance(obj, dict):
        return {k: cast_floating_mx_tree(v, dtype) for k, v in obj.items()}
    if isinstance(obj, list):
        return [cast_floating_mx_tree(v, dtype) for v in obj]
    if isinstance(obj, mx.array) and mx.issubdtype(obj.dtype, mx.floating):
        return obj.astype(dtype)
    return obj


def cast_module_parameters(module: Any, dtype: mx.Dtype) -> None:
    """In-place cast of all floating ``nn.Module`` parameters."""
    from mlx.utils import tree_flatten, tree_unflatten

    flat = dict(tree_flatten(module.parameters()))
    casted = cast_floating_mx_tree(flat, dtype)
    module.update(tree_unflatten(list(casted.items())))
    mx.eval(module.parameters())
