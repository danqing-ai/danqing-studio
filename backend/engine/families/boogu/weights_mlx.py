"""Boogu DiT + bundle weight loading (MLX)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open


def read_safetensors_dir(directory: Path, *, dtype: Any = mx.bfloat16) -> dict[str, mx.array]:
    weights: dict[str, mx.array] = {}
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(str(path), framework="mlx") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key).astype(dtype)
    if not weights:
        raise RuntimeError(f"No safetensors weights found under {directory}")
    return weights


def load_named_into_mlx(model: nn.Module, weights: dict[str, mx.array], *, strict: bool = False) -> None:
    """Load name-matched weights into an MLX module (quantized DiT-safe)."""
    from mlx.utils import tree_flatten, tree_unflatten

    model_keys = {k for k, _ in tree_flatten(model.parameters())}
    flat: list[tuple[str, mx.array]] = []
    for key, value in weights.items():
        if key not in model_keys:
            continue
        flat.append((key, value if isinstance(value, mx.array) else mx.array(value)))
    model.update(tree_unflatten(flat))
    mx.eval(model.parameters())
    if strict:
        missing = sorted(model_keys - {k for k, _ in flat})
        extra = sorted(set(weights) - model_keys)
        if missing or extra:
            raise RuntimeError(
                f"Boogu DiT strict weight load failed: missing={missing[:8]} extra={extra[:8]}"
            )


def load_boogu_dit_mlx(model: nn.Module, transformer_dir: Path, *, dtype: Any = mx.bfloat16) -> None:
    qpath = transformer_dir / "quant_config.json"
    if qpath.is_file():
        qc = json.loads(qpath.read_text())
        wfile = transformer_dir / qc.get("weights_file", "transformer_int4.safetensors")
        if not wfile.is_file():
            raise RuntimeError(f"Boogu quantized weights missing: {wfile}")
        g, b = int(qc["group_size"]), int(qc["bits"])

        def _pred(path: str, m: nn.Module) -> bool:
            return (
                isinstance(m, nn.Linear)
                and (("attn" in path) or ("feed_forward" in path))
                and m.weight.shape[1] % g == 0
            )

        nn.quantize(model, group_size=g, bits=b, class_predicate=_pred)
        load_named_into_mlx(model, dict(mx.load(str(wfile)).items()))
        return

    single = transformer_dir / "diffusion_pytorch_model.safetensors"
    if single.is_file():
        with safe_open(str(single), framework="mlx") as f:
            weights = {k: f.get_tensor(k).astype(dtype) for k in f.keys()}
        load_named_into_mlx(model, weights)
        return

    weights = read_safetensors_dir(transformer_dir, dtype=dtype)
    load_named_into_mlx(model, weights)


def resolve_boogu_bundle_dirs(bundle_root: Path) -> dict[str, Path]:
    """Resolve Boogu bundle layout (full HF bundle or mlx-community DiT-only)."""
    transformer = bundle_root / "transformer"
    vae = bundle_root / "vae"
    scheduler = bundle_root / "scheduler"
    mllm = bundle_root / "mllm"
    processor = bundle_root / "processor"

    core_missing = [name for name, p in (
        ("transformer", transformer),
        ("vae", vae),
        ("scheduler", scheduler),
    ) if not p.is_dir()]
    if core_missing:
        raise RuntimeError(
            f"Boogu-Image bundle at {bundle_root} is missing: {', '.join(core_missing)}. "
            "Re-download the model bundle."
        )

    if not mllm.is_dir():
        ext_mllm = _resolve_external_qwen3_vl_mllm(bundle_root)
        if ext_mllm is None:
            raise RuntimeError(
                f"Boogu-Image bundle at {bundle_root} is missing mllm/. "
                "Install Boogu mlx versions (bundle_repos pulls "
                "mlx-community/Qwen3-VL-8B-Instruct-8bit → "
                "models/LLM/qwen3-vl-8b-instruct-mlx-q8) or use the full Boogu bf16 bundle."
            )
        mllm = ext_mllm

    if not processor.is_dir():
        processor = mllm

    _assert_boogu_qwen3_vl_hidden_size(mllm)

    return {
        "transformer": transformer,
        "vae": vae,
        "scheduler": scheduler,
        "mllm": mllm,
        "processor": processor,
    }


def _qwen3_vl_bundle_usable(root: Path) -> bool:
    if not root.is_dir() or not (root / "config.json").is_file():
        return False
    for path in root.rglob("*.safetensors"):
        try:
            if path.is_file() and path.stat().st_size > 1024:
                return True
        except OSError:
            continue
    return False


def _assert_boogu_qwen3_vl_hidden_size(vlm_dir: Path, *, required: int = 4096) -> None:
    cfg_path = vlm_dir / "config.json"
    if not cfg_path.is_file():
        return
    cfg = json.loads(cfg_path.read_text())
    text_cfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    hidden = int((text_cfg or {}).get("hidden_size") or cfg.get("hidden_size") or 0)
    if hidden and hidden != required:
        raise RuntimeError(
            f"Boogu-Image requires Qwen3-VL-8B (text hidden_size={required}), "
            f"but {vlm_dir} reports hidden_size={hidden}. "
            "Qwen3-VL-4B Instruct is not compatible. Install "
            "mlx-community/Qwen3-VL-8B-Instruct-* to models/LLM/qwen3-vl-8b-instruct-mlx-q8."
        )


def _resolve_external_qwen3_vl_mllm(bundle_root: Path) -> Path | None:
    explicit = str(os.environ.get("DANQING_BOOGU_QWEN3_VL_PATH") or "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if _qwen3_vl_bundle_usable(path):
            return path

    models_dir = bundle_root.parent.parent if bundle_root.parent.name == "Image" else None
    if models_dir is None or models_dir.name != "models":
        return None

    llm_dir = models_dir / "LLM"
    if not llm_dir.is_dir():
        return None

    candidates = sorted(
        p for p in llm_dir.iterdir()
        if p.is_dir() and "qwen3-vl-8b" in p.name.lower() and "processor" not in p.name.lower()
    )
    for path in candidates:
        if _qwen3_vl_bundle_usable(path):
            return path
    return None
