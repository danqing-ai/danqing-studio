"""Real-ESRGAN MLX upscale — tiled inference + job runner."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np
from PIL import Image

from backend.engine.families.esrgan.rrdb_mlx import RRDBNet
from backend.engine.families.esrgan.weights import remap_esrgan_weights

_VARIANTS: dict[str, dict[str, int | str]] = {
    "RealESRGAN_x4plus": {"netscale": 4, "num_block": 23, "scale": 4},
    "RealESRGAN_x2plus": {"netscale": 2, "num_block": 23, "scale": 2},
    "RealESRGAN_x4plus_anime_6B": {"netscale": 4, "num_block": 6, "scale": 4},
}


@dataclass(frozen=True)
class ESRGANUpscaleRuntime:
    model: RRDBNet
    netscale: int
    variant: str
    tile: int = 0
    tile_pad: int = 10
    pre_pad: int = 10


def expected_esrgan_weight_files() -> tuple[str, ...]:
    return ("model.safetensors",)


def _find_esrgan_weight_file(bundle_path: Path) -> Path:
    direct = bundle_path / "model.safetensors"
    if direct.is_file():
        return direct
    matches = sorted(bundle_path.rglob("model.safetensors"))
    if matches:
        return matches[0]
    safes = sorted(bundle_path.rglob("*.safetensors"))
    if len(safes) == 1:
        return safes[0]
    if safes:
        names = [p.name for p in safes[:5]]
        raise RuntimeError(
            f"Real-ESRGAN bundle under {bundle_path} has multiple safetensors {names!r}; "
            "expected a single model.safetensors (mlx-community Real-ESRGAN-x4plus layout)"
        )
    raise RuntimeError(
        f"Real-ESRGAN bundle missing model.safetensors under {bundle_path}; "
        "install from Models → Real-ESRGAN x4+ (ModelScope: mlx-community/Real-ESRGAN-x4plus)"
    )


def _resolve_variant(bundle_path: Path, model_key: str) -> str:
    cfg_path = bundle_path / "config.json"
    if cfg_path.is_file():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            variant = str(data.get("esrgan_variant") or data.get("variant") or "").strip()
            if variant in _VARIANTS:
                return variant
        except Exception:
            pass
    key = model_key.lower()
    if "x2" in key or "2x" in key:
        return "RealESRGAN_x2plus"
    if "anime" in key:
        return "RealESRGAN_x4plus_anime_6B"
    return "RealESRGAN_x4plus"


def validate_esrgan_bundle(bundle_path: Path, model_key: str = "") -> None:
    del model_key
    try:
        _find_esrgan_weight_file(bundle_path)
    except RuntimeError as e:
        missing = [n for n in expected_esrgan_weight_files() if not (bundle_path / n).is_file()]
        if missing:
            raise RuntimeError(str(e)) from e
        raise


def _load_weights_into_model(model: RRDBNet, bundle_path: Path) -> None:
    sf = _find_esrgan_weight_file(bundle_path)
    raw = dict(mx.load(str(sf)).items())
    remapped = remap_esrgan_weights(raw)
    model.load_weights(list(remapped.items()), strict=True)
    mx.eval(model.parameters())


def load_esrgan_upscale_pipeline(
    *,
    bundle_path: Path,
    model_key: str,
    model_cache: Any | None = None,
    cache_key: str | None = None,
    cache_size_gb: float | None = None,
    on_log: Callable[[str, str], None] | None = None,
    tile: int = 0,
) -> ESRGANUpscaleRuntime:
    validate_esrgan_bundle(bundle_path, model_key)
    if model_cache is not None and cache_key:
        cached = model_cache.get(cache_key)
        if cached is not None:
            return cached

    variant = _resolve_variant(bundle_path, model_key)
    spec = _VARIANTS[variant]
    model = RRDBNet(
        scale=int(spec["scale"]),
        num_block=int(spec["num_block"]),
    )
    _load_weights_into_model(model, bundle_path)
    runtime = ESRGANUpscaleRuntime(
        model=model,
        netscale=int(spec["netscale"]),
        variant=variant,
        tile=max(0, int(tile)),
    )
    if on_log:
        on_log("info", f"esrgan loaded variant={variant} tile={runtime.tile} from {bundle_path}")
    if model_cache is not None and cache_key and cache_size_gb is not None:
        model_cache.put(cache_key, runtime, cache_size_gb)
    return runtime


class _RealESRGANer:
    def __init__(self, runtime: ESRGANUpscaleRuntime) -> None:
        self._rt = runtime
        self.scale = runtime.netscale
        self.tile_size = runtime.tile
        self.tile_pad = runtime.tile_pad
        self.pre_pad = runtime.pre_pad
        self.mod_scale = {2: 2, 1: 4}.get(self.scale)

    def _infer(self, nhwc: np.ndarray) -> np.ndarray:
        y = self._rt.model(mx.array(nhwc.astype(np.float32)))
        mx.eval(y)
        return np.array(y).astype(np.float32)

    def _pre_pad(self, img: np.ndarray) -> tuple[np.ndarray, int, int]:
        if self.pre_pad:
            img = np.pad(img, ((0, 0), (0, self.pre_pad), (0, self.pre_pad), (0, 0)), mode="reflect")
        mph = mpw = 0
        if self.mod_scale is not None:
            h, w = img.shape[1], img.shape[2]
            if h % self.mod_scale:
                mph = self.mod_scale - h % self.mod_scale
            if w % self.mod_scale:
                mpw = self.mod_scale - w % self.mod_scale
            img = np.pad(img, ((0, 0), (0, mph), (0, mpw), (0, 0)), mode="reflect")
        return img, mph, mpw

    def _post_crop(self, out: np.ndarray, mph: int, mpw: int) -> np.ndarray:
        h, w = out.shape[1], out.shape[2]
        if self.mod_scale is not None:
            out = out[:, 0 : h - mph * self.scale, 0 : w - mpw * self.scale, :]
        if self.pre_pad:
            h, w = out.shape[1], out.shape[2]
            out = out[:, 0 : h - self.pre_pad * self.scale, 0 : w - self.pre_pad * self.scale, :]
        return out

    def _tile_process(self, img: np.ndarray) -> np.ndarray:
        _, height, width, channel = img.shape
        s = self.scale
        out = np.zeros((1, height * s, width * s, channel), dtype=np.float32)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)
        for y in range(tiles_y):
            for x in range(tiles_x):
                ofs_x, ofs_y = x * self.tile_size, y * self.tile_size
                isx, iex = ofs_x, min(ofs_x + self.tile_size, width)
                isy, iey = ofs_y, min(ofs_y + self.tile_size, height)
                isxp, iexp = max(isx - self.tile_pad, 0), min(iex + self.tile_pad, width)
                isyp, ieyp = max(isy - self.tile_pad, 0), min(iey + self.tile_pad, height)
                tw, th = iex - isx, iey - isy
                tile = img[:, isyp:ieyp, isxp:iexp, :]
                ot = self._infer(tile)
                osxt = (isx - isxp) * s
                osyt = (isy - isyp) * s
                out[:, isy * s : iey * s, isx * s : iex * s, :] = ot[
                    :, osyt : osyt + th * s, osxt : osxt + tw * s, :
                ]
        return out

    def enhance(self, img: np.ndarray, *, outscale: float | None = None) -> np.ndarray:
        h_in, w_in = img.shape[:2]
        max_range = 65535 if img.dtype == np.uint16 else 255
        arr = img.astype(np.float32) / max_range
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        elif arr.shape[2] == 4:
            arr = arr[:, :, 0:3]
        nhwc = arr[None, ...]
        padded, mph, mpw = self._pre_pad(nhwc)
        out = self._tile_process(padded) if self.tile_size > 0 else self._infer(padded)
        out = self._post_crop(out, mph, mpw)
        out_rgb = np.clip(out[0], 0.0, 1.0)

        target_scale = float(outscale if outscale is not None else self.scale)
        if target_scale != float(self.scale):
            target_w = int(round(w_in * target_scale))
            target_h = int(round(h_in * target_scale))
            im = Image.fromarray((out_rgb * 255.0).astype(np.uint8), mode="RGB")
            im = im.resize((target_w, target_h), Image.LANCZOS)
            out_rgb = np.asarray(im).astype(np.float32) / 255.0

        return (out_rgb * 255.0).round().astype(np.uint8)


def run_esrgan_upscale(
    *,
    bundle_path: Path,
    model_key: str,
    source_image: Path,
    scale: int,
    softness: float,
    seed: int | None,
    output_png: Path,
    on_log: Callable[[str, str], None] | None = None,
    pipeline: ESRGANUpscaleRuntime | None = None,
    tile_size: int = 0,
) -> dict[str, Any]:
    del softness, seed
    validate_esrgan_bundle(bundle_path, model_key)
    if scale not in (2, 4):
        raise RuntimeError(f"Real-ESRGAN upscale scale must be 2 or 4, got {scale!r}")
    if not source_image.is_file():
        raise RuntimeError(f"Real-ESRGAN source image not found: {source_image}")

    runtime = pipeline
    if runtime is None:
        runtime = load_esrgan_upscale_pipeline(
            bundle_path=bundle_path,
            model_key=model_key,
            tile=tile_size,
            on_log=on_log,
        )
    elif tile_size > 0:
        runtime = ESRGANUpscaleRuntime(
            model=runtime.model,
            netscale=runtime.netscale,
            variant=runtime.variant,
            tile=tile_size,
            tile_pad=runtime.tile_pad,
            pre_pad=runtime.pre_pad,
        )

    upsampler = _RealESRGANer(runtime)
    img = np.asarray(Image.open(source_image).convert("RGB"))
    outscale = float(scale)
    if on_log:
        on_log(
            "info",
            f"esrgan_upscale variant={runtime.variant} netscale={runtime.netscale} "
            f"outscale={outscale} tile={runtime.tile} src={source_image.name}",
        )
    out = upsampler.enhance(img, outscale=outscale)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(str(output_png))

    return {
        "upscale_backend": "backend.engine.families.esrgan.stem_mlx",
        "variant": runtime.variant,
        "scale": int(scale),
        "tile": runtime.tile,
    }
