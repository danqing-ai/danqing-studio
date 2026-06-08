"""Flat safetensors keys → nested MLX dict: mapping targets, transforms, and mapper.

Shared by SeedVR2 ``WeightLoader`` and any future bundle-driven MLX loaders. Lives in
``engine.common`` so call sites do not reach through ``seedvr2`` internals for generic
table + apply logic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import mlx.core as mx


@dataclass
class WeightTarget:
    to_pattern: str
    from_pattern: list[str]
    transform: Callable[[mx.array], mx.array] | None = None
    required: bool = True
    max_blocks: int | None = None


class WeightMapping:
    """Optional base for mapping table classes used with ``WeightMapper``."""

    pass


class WeightTransforms:
    @staticmethod
    def reshape_gamma_to_1d(tensor: mx.array) -> mx.array:
        if len(tensor.shape) > 1:
            return mx.reshape(tensor, (tensor.shape[0],))
        return tensor

    @staticmethod
    def transpose_patch_embed(tensor: mx.array) -> mx.array:
        if len(tensor.shape) == 5:
            return tensor.transpose(0, 2, 3, 4, 1)
        return tensor

    @staticmethod
    def transpose_conv3d_weight(tensor: mx.array) -> mx.array:
        if len(tensor.shape) == 5:
            return tensor.transpose(0, 2, 3, 4, 1)
        return tensor

    @staticmethod
    def transpose_conv2d_weight(tensor: mx.array) -> mx.array:
        if len(tensor.shape) == 4:
            return tensor.transpose(0, 2, 3, 1)
        return tensor

    @staticmethod
    def transpose_conv_transpose2d_weight(tensor: mx.array) -> mx.array:
        if len(tensor.shape) == 4:
            return tensor.transpose(1, 2, 3, 0)
        return tensor


class WeightMapper:
    @staticmethod
    def apply_mapping(
        hf_weights: dict[str, mx.array],
        mapping: list[WeightTarget],
        num_blocks: int | None = None,
        num_layers: int | None = None,
    ) -> dict:
        if num_blocks is None:
            num_blocks = WeightMapper._detect_num_blocks(hf_weights)

        if num_layers is None:
            num_layers = WeightMapper._detect_num_layers(hf_weights)

        flat_mapping = WeightMapper._build_flat_mapping(mapping, num_blocks, num_layers)

        mapped_weights: dict = {}

        for hf_key, hf_tensor in hf_weights.items():
            targets = flat_mapping.get(hf_key, [])

            if targets:
                for mlx_path, transform in targets:
                    tensor = hf_tensor
                    if transform:
                        tensor = transform(tensor)

                    WeightMapper._set_nested_value(mapped_weights, mlx_path, tensor)

        return mapped_weights

    @staticmethod
    def _detect_num_blocks(hf_weights: dict[str, mx.array]) -> int:
        block_numbers: set[int] = set()
        for key in hf_weights.keys():
            match = re.search(r"transformer_blocks\.(\d+)\.", key)
            if match:
                block_numbers.add(int(match.group(1)))
                continue
            match = re.search(r"single_transformer_blocks\.(\d+)\.", key)
            if match:
                block_numbers.add(int(match.group(1)))

        if block_numbers:
            return max(block_numbers) + 1
        return 0

    @staticmethod
    def _detect_num_layers(hf_weights: dict[str, mx.array]) -> int:
        layer_numbers: set[int] = set()
        for key in hf_weights.keys():
            match = re.search(r"model\.layers\.(\d+)\.", key)
            if match:
                layer_numbers.add(int(match.group(1)))

        if layer_numbers:
            return max(layer_numbers) + 1
        return 28

    @staticmethod
    def _build_flat_mapping(
        mapping: list[WeightTarget],
        num_blocks: int = 0,
        num_layers: int = 28,
    ) -> dict[str, list[tuple[str, Callable[[mx.array], mx.array] | None]]]:
        flat: dict[str, list[tuple[str, Callable[[mx.array], mx.array] | None]]] = {}

        def add_mapping(
            hf_key: str,
            mlx_path: str,
            transform: Callable[[mx.array], mx.array] | None,
        ) -> None:
            if hf_key not in flat:
                flat[hf_key] = []
            flat[hf_key].append((mlx_path, transform))

        for target in mapping:
            for hf_pattern in target.from_pattern:
                hf_has_block = "{block}" in hf_pattern
                to_has_block = "{block}" in target.to_pattern
                has_i = "{i}" in hf_pattern or "{i}" in target.to_pattern
                has_res = "{res}" in hf_pattern or "{res}" in target.to_pattern
                has_layer = "{layer}" in hf_pattern or "{layer}" in target.to_pattern

                if (hf_has_block or to_has_block) and has_res:
                    max_blocks = num_blocks if num_blocks > 0 else 4
                    for block_num in range(max_blocks):
                        for res in range(3):
                            concrete_hf = hf_pattern.replace("{block}", str(block_num)).replace(
                                "{res}", str(res)
                            )
                            concrete_mlx = target.to_pattern.replace("{block}", str(block_num)).replace(
                                "{res}", str(res)
                            )
                            add_mapping(concrete_hf, concrete_mlx, target.transform)
                elif hf_has_block and to_has_block:
                    if target.max_blocks is not None:
                        max_blocks = target.max_blocks
                    elif "visual.blocks" in hf_pattern or "visual.blocks" in target.to_pattern:
                        max_blocks = 32
                    else:
                        max_blocks = num_blocks if num_blocks > 0 else 4
                    for block_num in range(max_blocks):
                        concrete_hf = hf_pattern.replace("{block}", str(block_num))
                        concrete_mlx = target.to_pattern.replace("{block}", str(block_num))
                        add_mapping(concrete_hf, concrete_mlx, target.transform)
                elif to_has_block and not hf_has_block:
                    if target.max_blocks is not None:
                        max_blocks = target.max_blocks
                    else:
                        max_blocks = num_blocks if num_blocks > 0 else 24
                    for block_num in range(max_blocks):
                        concrete_mlx = target.to_pattern.replace("{block}", str(block_num))
                        add_mapping(hf_pattern, concrete_mlx, target.transform)
                elif has_layer:
                    max_layers = num_layers if num_layers > 0 else 28
                    for layer_num in range(max_layers):
                        concrete_hf = hf_pattern.replace("{layer}", str(layer_num))
                        concrete_mlx = target.to_pattern.replace("{layer}", str(layer_num))
                        add_mapping(concrete_hf, concrete_mlx, target.transform)
                elif has_i:
                    for i in range(2):
                        concrete_hf = hf_pattern.replace("{i}", str(i))
                        concrete_mlx = target.to_pattern.replace("{i}", str(i))
                        add_mapping(concrete_hf, concrete_mlx, target.transform)
                elif has_res:
                    if "up_block" in hf_pattern:
                        for res in range(3):
                            concrete_hf = hf_pattern.replace("{res}", str(res))
                            concrete_mlx = target.to_pattern.replace("{res}", str(res))
                            add_mapping(concrete_hf, concrete_mlx, target.transform)
                else:
                    add_mapping(hf_pattern, target.to_pattern, target.transform)

        return flat

    @staticmethod
    def _set_nested_value(d: dict, path: str, value: mx.array) -> None:
        parts = path.split(".")
        current = d
        i = 0

        while i < len(parts) - 1:
            part = parts[i]

            if i + 1 < len(parts) and parts[i + 1].isdigit():
                if part not in current:
                    current[part] = []
                idx = int(parts[i + 1])
                while len(current[part]) <= idx:
                    current[part].append({})
                current = current[part][idx]
                i += 2
            else:
                if part not in current:
                    current[part] = {}
                current = current[part]
                i += 1

        final_key = parts[-1]
        current[final_key] = value
