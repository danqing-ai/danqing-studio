"""Safetensors metadata for MLX affine-quantized linear bundles (e.g. ``quantization_level``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def read_affine_quant_bits_from_transformer_dir(transformer_dir: Path) -> int | None:
    """Read ``quantization_level`` from the first transformer shard that carries it.

    MLX ``save_safetensors`` / in-repo conversion store ``"4"`` or ``"8"`` for affine
    :class:`mlx.nn.QuantizedLinear` bundles. Returns ``None`` if absent (caller may still
    infer from tensor shapes when the dense parameter map is available).
    """
    if not transformer_dir.is_dir():
        return None
    try:
        from safetensors import safe_open
    except ImportError:
        return None

    for sf in sorted(transformer_dir.glob("*.safetensors")):
        try:
            with safe_open(str(sf), framework="np", device="cpu") as f:
                meta = f.metadata()
                if not meta:
                    continue
                raw = meta.get("quantization_level")
                if raw is None:
                    continue
                s = str(raw).strip()
                if s in ("", "None", "null"):
                    continue
                v = int(s, 10)
                if v in (4, 8):
                    return v
                raise RuntimeError(
                    f"Invalid safetensors metadata quantization_level={raw!r} in {sf.name} "
                    f"(expected 4 or 8 for MLX affine linear bundles)."
                )
        except RuntimeError:
            raise
        except Exception:
            continue
    return None


def read_affine_quant_bits_from_safetensors(path: Path) -> int | None:
    """Read ``quantization_level`` from a single safetensors file (e.g. LTX / ACE-Step DiT)."""
    if not path.is_file():
        return None
    try:
        from safetensors import safe_open
    except ImportError:
        return None
    try:
        with safe_open(str(path), framework="np", device="cpu") as f:
            meta = f.metadata()
            if not meta:
                return None
            raw = meta.get("quantization_level")
            if raw is None:
                return None
            s = str(raw).strip()
            if s in ("", "None", "null"):
                return None
            v = int(s, 10)
            if v in (4, 8):
                return v
            raise RuntimeError(
                f"Invalid safetensors metadata quantization_level={raw!r} in {path.name} "
                f"(expected 4 or 8 for MLX affine linear bundles)."
            )
    except RuntimeError:
        raise
    except Exception:
        return None


def read_bundle_affine_bits_if_quantized(
    w: dict[str, Any],
    transformer_dir: Path,
) -> int | None:
    """If ``w`` looks like MLX affine quant (``*.scales``), return shard ``quantization_level`` bits."""
    if not any(k.endswith(".scales") for k in w.keys()):
        return None
    if transformer_dir.is_file():
        return read_affine_quant_bits_from_safetensors(transformer_dir)
    return read_affine_quant_bits_from_transformer_dir(transformer_dir)
