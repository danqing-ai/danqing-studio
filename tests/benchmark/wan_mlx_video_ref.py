"""mlx-video reference runner for Wan2.2 TI2V-5B benchmark parity (same seed PSNR).

Converts the official ModelScope/HuggingFace bundle to mlx-video layout once
(``tests/benchmark/.cache/wan22_mlx_ref/<bundle-name>/``), then calls
``mlx_video.models.wan_2.generate.generate_video`` with aligned geometry
(480×704 etc.) and UniPC + shift=5 to match DanQing Studio defaults.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _parse_size(size: str) -> tuple[int, int]:
    parts = str(size).lower().replace(" ", "").split("x")
    if len(parts) != 2:
        raise ValueError(f"invalid size {size!r}, expected WxH")
    return int(parts[0]), int(parts[1])


def _ensure_converted(bundle: Path, cache: Path) -> Path:
    """One-time mlx-video convert (model + T5 + VAE safetensors + config.json)."""
    ready = cache / "model.safetensors"
    if ready.is_file() and (cache / "t5_encoder.safetensors").is_file() and (cache / "vae.safetensors").is_file():
        return cache
    cache.mkdir(parents=True, exist_ok=True)
    if (bundle / "config.json").is_file() and not (cache / "config.json").is_file():
        shutil.copy2(bundle / "config.json", cache / "config.json")
    tok_src = bundle / "google" / "umt5-xxl"
    tok_dst = cache / "google" / "umt5-xxl"
    if tok_src.is_dir() and not tok_dst.is_dir():
        shutil.copytree(tok_src, tok_dst)
    try:
        from mlx_video.models.wan_2.convert import convert_wan_checkpoint
    except ImportError as e:
        raise SystemExit(
            "mlx-video is required for Wan PSNR reference runs. "
            "Install with: pip install git+https://github.com/Blaizzy/mlx-video.git"
        ) from e
    convert_wan_checkpoint(
        str(bundle),
        str(cache),
        dtype="bfloat16",
        model_version="auto",
        quantize=False,
    )
    return cache


def main() -> int:
    parser = argparse.ArgumentParser(description="mlx-video Wan2.2 TI2V-5B reference for benchmark")
    parser.add_argument("--bundle-dir", required=True, help="Official Wan bundle root")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--size", default="480x704", help="WxH pixels (use VAE×patch-aligned sizes)")
    parser.add_argument("--num-frames", type=int, default=17)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scheduler", default="unipc", choices=("unipc", "euler", "dpm++"))
    args = parser.parse_args()

    bundle = Path(args.bundle_dir).resolve()
    if not bundle.is_dir():
        print(f"[wan-mlx-ref] bundle not found: {bundle}", file=sys.stderr)
        return 2

    cache = Path(__file__).resolve().parent / ".cache" / "wan22_mlx_ref" / bundle.name
    model_dir = _ensure_converted(bundle, cache)

    # mlx-video generate hardcodes HF hub ``google/umt5-xxl``; use bundle-local tokenizer offline.
    tok_local = model_dir / "google" / "umt5-xxl"
    if not tok_local.is_dir():
        tok_local = bundle / "google" / "umt5-xxl"
    if tok_local.is_dir():
        import transformers

        _orig_from_pretrained = transformers.AutoTokenizer.from_pretrained

        def _tokenizer_from_local(name: str, *a, **k):
            if str(name).strip() in ("google/umt5-xxl", "umt5-xxl"):
                return _orig_from_pretrained(str(tok_local), *a, **k)
            return _orig_from_pretrained(name, *a, **k)

        transformers.AutoTokenizer.from_pretrained = _tokenizer_from_local  # type: ignore[method-assign]

    try:
        from mlx_video.models.wan_2.generate import generate_video
    except ImportError as e:
        raise SystemExit("mlx-video package not installed") from e

    width, height = _parse_size(args.size)
    neg = args.negative_prompt.strip() or None

    generate_video(
        model_dir=str(model_dir),
        prompt=args.prompt,
        negative_prompt=neg,
        width=width,
        height=height,
        num_frames=int(args.num_frames),
        steps=int(args.steps),
        guide_scale=float(args.guidance),
        shift=float(args.shift),
        seed=int(args.seed),
        output_path=str(args.output),
        scheduler=str(args.scheduler),
        tiling="none",
        no_compile=True,
    )
    print(f"[wan-mlx-ref] saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
