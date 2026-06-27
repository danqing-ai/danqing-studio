"""MLX affine quantization helpers for local derived weight conversion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def quantize_linear_weights_dict(weights: dict[str, Any], bits: int) -> dict[str, Any]:
    """Quantize 2D Linear weights in a flat safetensors dict (MLX affine layout)."""
    if bits not in (4, 8):
        raise ValueError(f"Unsupported quantization bits: {bits}")

    import mlx.nn as nn

    quantized: dict[str, Any] = {}
    processed_bias_keys: set[str] = set()

    for key, tensor in weights.items():
        if not key.endswith(".weight"):
            continue
        if tensor.ndim != 2 or tensor.shape[0] <= 1 or tensor.shape[1] <= 1:
            continue
        if any(x in key.lower() for x in ("embed", "vocab", "token")):
            quantized[key] = tensor
            continue

        in_features = int(tensor.shape[1])
        out_features = int(tensor.shape[0])
        bias_key = key.replace(".weight", ".bias")
        has_bias = bias_key in weights

        linear = nn.Linear(in_features, out_features, bias=has_bias)
        linear.weight = tensor
        if has_bias:
            linear.bias = weights[bias_key]
            processed_bias_keys.add(bias_key)

        q_linear = linear.to_quantized(bits=bits)
        base = key[:-7]
        quantized[f"{base}.weight"] = q_linear.weight
        quantized[f"{base}.scales"] = q_linear.scales
        quantized[f"{base}.biases"] = q_linear.biases
        if "bias" in q_linear:
            quantized[f"{base}.bias"] = q_linear.bias

    for key, tensor in weights.items():
        if key in processed_bias_keys or key in quantized:
            continue
        quantized[key] = tensor

    return quantized


def save_quantized_weight_bundle(
    quantized: dict[str, Any],
    *,
    output_dir: Path,
    shard_prefix: str,
    bits: int,
    single_output_file: Path | None = None,
    max_shard_bytes: int = 2 << 30,
) -> int:
    """Write quantized tensors to safetensors (single file or sharded + index)."""
    import mlx.core as mx

    output_dir.mkdir(parents=True, exist_ok=True)
    linear_count = sum(1 for k in quantized if k.endswith(".scales"))

    if single_output_file is not None:
        mx.save_safetensors(
            str(single_output_file),
            quantized,
            metadata={"quantization_level": str(bits)},
        )
        return linear_count

    shards: list[dict[str, Any]] = []
    current_shard: dict[str, Any] = {}
    current_size = 0

    for key, value in quantized.items():
        if current_size + value.nbytes > max_shard_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[key] = value
        current_size += value.nbytes
    if current_shard:
        shards.append(current_shard)

    weight_map: dict[str, str] = {}
    for i, shard in enumerate(shards):
        shard_name = f"{shard_prefix}_{i:05d}.safetensors"
        mx.save_safetensors(
            str(output_dir / shard_name),
            shard,
            metadata={"quantization_level": str(bits)},
        )
        for k in shard.keys():
            weight_map[k] = shard_name

    _write_quantized_index(output_dir, weight_map, bits)
    return linear_count


def _write_quantized_index(output_dir: Path, weight_map: dict[str, str], bits: int) -> None:
    index_data = {
        "metadata": {"quantization_level": str(bits)},
        "weight_map": weight_map,
    }
    with open(output_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)


def _release_mlx_memory() -> None:
    import gc

    gc.collect()
    try:
        import mlx.core as mx

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
    except Exception:
        pass


def quantize_and_save_sharded_inputs(
    load_paths: tuple[Path, ...] | list[Path],
    *,
    bits: int,
    output_dir: Path,
    shard_prefix: str = "model",
    single_output_file: Path | None = None,
) -> int:
    """Quantize large multi-shard checkpoints one input file at a time.

    Wan 14B MoE experts ship ~9GB safetensors per shard; loading all shards into
    one dict can exceed unified memory. This path keeps peak RAM near one shard.
    """
    import mlx.core as mx

    paths = tuple(load_paths)
    if not paths:
        raise RuntimeError("quantize_and_save_sharded_inputs: no input shards")

    if single_output_file is not None and len(paths) == 1:
        weights = dict(mx.load(str(paths[0])))
        if not weights:
            raise RuntimeError(f"No safetensors weights found in {paths[0]!r}")
        quantized = quantize_linear_weights_dict(weights, bits)
        del weights
        count = save_quantized_weight_bundle(
            quantized,
            output_dir=output_dir,
            shard_prefix=shard_prefix,
            bits=bits,
            single_output_file=single_output_file,
        )
        del quantized
        _release_mlx_memory()
        return count

    output_dir.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    linear_count = 0

    for i, shard_path in enumerate(paths):
        weights = dict(mx.load(str(shard_path)))
        if not weights:
            raise RuntimeError(f"No safetensors weights found in {shard_path!r}")
        quantized = quantize_linear_weights_dict(weights, bits)
        del weights

        shard_name = f"{shard_prefix}_{i:05d}.safetensors"
        mx.save_safetensors(
            str(output_dir / shard_name),
            quantized,
            metadata={"quantization_level": str(bits)},
        )
        for key in quantized:
            weight_map[key] = shard_name
        linear_count += sum(1 for k in quantized if k.endswith(".scales"))
        del quantized
        _release_mlx_memory()

    _write_quantized_index(output_dir, weight_map, bits)
    return linear_count
