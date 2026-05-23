"""MLX dtype helpers shared by text encoders."""
from __future__ import annotations

import importlib
from typing import Any


def _mlx_core() -> Any:
    return importlib.import_module("mlx.core")


def cast_floating_mx_tree(obj: Any, dtype: Any) -> Any:
    mx = _mlx_core()
    if isinstance(obj, dict):
        return {k: cast_floating_mx_tree(v, dtype) for k, v in obj.items()}
    if isinstance(obj, list):
        return [cast_floating_mx_tree(v, dtype) for v in obj]
    obj_mod = getattr(obj.__class__, "__module__", "")
    if obj_mod.startswith("mlx.") and hasattr(obj, "dtype") and mx.issubdtype(obj.dtype, mx.floating):
        return obj.astype(dtype)
    return obj


def cast_module_parameters(
    module: Any, dtype: Any, *, eval_fn: Any | None = None
) -> None:
    """In-place cast of all floating ``nn.Module`` parameters."""
    mx = _mlx_core()
    mx_utils = importlib.import_module("mlx.utils")

    flat = dict(mx_utils.tree_flatten(module.parameters()))
    casted = cast_floating_mx_tree(flat, dtype)
    module.update(mx_utils.tree_unflatten(list(casted.items())))
    if eval_fn is not None:
        eval_fn(module.parameters())
    else:
        _mlx_core().eval(module.parameters())
