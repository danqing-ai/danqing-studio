"""MLX LoRA linear layers for DreamBooth training."""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen ``nn.Linear``."""

    @staticmethod
    def from_base(linear: nn.Linear, r: int = 8, scale: float = 1.0) -> "LoRALinear":
        output_dims, input_dims = linear.weight.shape
        lora_lin = LoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            r=r,
            scale=scale,
        )
        lora_lin.linear = linear
        linear.freeze()
        return lora_lin

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        r: int = 8,
        scale: float = 1.0,
        bias: bool = False,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.scale = scale
        init_scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-init_scale,
            high=init_scale,
            shape=(input_dims, r),
        )
        self.lora_b = mx.zeros(shape=(r, output_dims))

    def __call__(self, x: mx.array) -> mx.array:
        y = self.linear(x)
        z = (x @ self.lora_a) @ self.lora_b
        return y + (self.scale * z).astype(x.dtype)


def _apply_lora_recursive(module: Any, rank: int) -> None:
    if isinstance(module, nn.Linear) and not isinstance(module, LoRALinear):
        return
    if isinstance(module, list):
        for item in module:
            _apply_lora_recursive(item, rank)
        return
    if not hasattr(module, "__dict__"):
        return
    for name, val in list(vars(module).items()):
        if name.startswith("_"):
            continue
        if isinstance(val, nn.Linear) and not isinstance(val, LoRALinear):
            setattr(module, name, LoRALinear.from_base(val, r=rank))
        else:
            _apply_lora_recursive(val, rank)


def _dit_inner(model: Any) -> Any:
    return getattr(model, "_inner", model)


def _walk_dit_nodes(root: Any) -> Any:
    if isinstance(root, (LoRALinear, nn.Linear)):
        yield root
        return
    if isinstance(root, list):
        for item in root:
            yield from _walk_dit_nodes(item)
        return
    if isinstance(root, nn.Module) and not isinstance(root, LoRALinear):
        for child in root.children().values():
            yield from _walk_dit_nodes(child)
        return
    if not hasattr(root, "__dict__"):
        return
    for name, val in vars(root).items():
        if name.startswith("_"):
            continue
        yield from _walk_dit_nodes(val)


def iter_lora_linears(model: Any) -> list[LoRALinear]:
    layers = [node for node in _walk_dit_nodes(_dit_inner(model)) if isinstance(node, LoRALinear)]
    if not layers:
        raise RuntimeError("No LoRA layers found on DiT after apply_lora")
    return layers


def freeze_dit_base_weights(model: Any) -> None:
    """Freeze all base ``nn.Linear`` weights; leave LoRA adapter tensors trainable."""
    for node in _walk_dit_nodes(_dit_inner(model)):
        if isinstance(node, LoRALinear):
            continue
        if isinstance(node, nn.Linear):
            node.freeze()


class DiTLoRATrainModule(nn.Module):
    """Thin ``nn.Module`` wrapper so ``mlx.nn.value_and_grad`` can optimize LoRA only."""

    def __init__(self, dit: Any, lora_layers: list[LoRALinear]):
        super().__init__()
        self._dit = dit
        for idx, layer in enumerate(lora_layers):
            setattr(self, f"lora_{idx}", layer)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._dit(*args, **kwargs)


def build_lora_train_module(model: Any) -> DiTLoRATrainModule:
    return DiTLoRATrainModule(model, iter_lora_linears(model))


def prepare_dit_for_lora_training(model: Any, apply_lora_fn: Any, **lora_kw: Any) -> tuple[Any, DiTLoRATrainModule]:
    apply_lora_fn(model, **lora_kw)
    freeze_dit_base_weights(model)
    train_module = build_lora_train_module(model)
    mx.eval(train_module.parameters())
    return model, train_module


def apply_lora_to_flux1_dit(model: Any, *, rank: int, lora_blocks: int) -> None:
    """Replace Linear layers in the last ``lora_blocks`` joint+single blocks with LoRA."""
    dit = getattr(model, "_inner", model)
    joint = list(dit.transformer_blocks)
    single = list(dit.single_transformer_blocks)
    all_blocks = list(reversed(joint + single))
    n = len(all_blocks) if lora_blocks < 0 else min(lora_blocks, len(all_blocks))
    for block in all_blocks[:n]:
        _apply_lora_recursive(block, rank)
    if hasattr(dit, "_build_param_map"):
        dit._build_param_map()
    if hasattr(model, "_inner") and hasattr(dit, "_param_map"):
        model._param_map = dit._param_map


def apply_lora_to_zimage_dit(model: Any, *, rank: int, lora_blocks: int) -> None:
    """Replace Linear layers in the last ``lora_blocks`` Z-Image DiT blocks with LoRA."""
    dit = getattr(model, "_inner", model)
    all_blocks: list[Any] = []
    for layer in reversed(list(dit.layers)):
        all_blocks.append(layer)
    for layer in reversed(list(dit.noise_refiner)):
        all_blocks.append(layer)
    for layer in reversed(list(dit.context_refiner)):
        all_blocks.append(layer)
    n = len(all_blocks) if lora_blocks < 0 else min(lora_blocks, len(all_blocks))
    for block in all_blocks[:n]:
        _apply_lora_recursive(block, rank)
    if hasattr(dit, "_build_param_map"):
        dit._build_param_map()
    if hasattr(model, "_inner") and hasattr(dit, "_param_map"):
        model._param_map = dit._param_map


def add_grad_trees(left: Any, right: Any) -> Any:
    """Element-wise add for nested MLX grad trees from ``value_and_grad``."""
    from mlx.utils import tree_map

    return tree_map(lambda a, b: a + b, left, right)


def scale_grad_tree(tree: Any, divisor: float) -> Any:
    from mlx.utils import tree_map

    return tree_map(lambda g: g / divisor, tree)


def collect_lora_safetensors(model: Any, *, rank: int) -> dict[str, mx.array]:
    """Export trainable LoRA weights with diffusers-compatible key names."""
    from mlx.utils import tree_flatten

    flat = dict(tree_flatten(model.trainable_parameters()))
    out: dict[str, mx.array] = {}
    for key, tensor in flat.items():
        if ".lora_a" in key:
            base = key.replace(".lora_a", "")
            out[f"{base}.lora_A.weight"] = tensor.T
        elif ".lora_b" in key:
            base = key.replace(".lora_b", "")
            out[f"{base}.lora_B.weight"] = tensor.T
    if not out:
        raise RuntimeError("No LoRA trainable parameters found to save")
    out["lora_rank"] = mx.array([float(rank)])
    return out
