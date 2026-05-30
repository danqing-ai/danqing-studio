"""HeartMuLa HeartCodec parity — fixed LM codes vs heartlib-mlx reference WAV."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from tests.benchmark.cases import (
    HEARTMULA_AUDIO_BUNDLE,
    heartmula_bundle_installed,
    mlx_runtime_available,
    resolve_benchmark_data_root,
)
from tests.benchmark.metrics import AudioCompareResult, SanityResult, compare_audio_waveforms

DEFAULT_MIN_SI_SDR_DB = 18.0
DEFAULT_MIN_CORRELATION = 0.90
DEFAULT_WARN_SI_SDR_DB = 12.0


@dataclass(frozen=True)
class CodecParityManifest:
    schema_version: int
    case_id: str
    codes_file: str
    reference_wav: str
    sample_rate: int = 48_000
    codec_steps: int = 20
    codec_guidance: float = 1.25
    chunk_duration_sec: float = 29.76
    codec_seed: int = 4_242_4243
    source: str = "heartlib-mlx"
    source_notes: str = ""

    @classmethod
    def load(cls, path: Path) -> "CodecParityManifest":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            case_id=str(raw.get("case_id", "")),
            codes_file=str(raw["codes_file"]),
            reference_wav=str(raw["reference_wav"]),
            sample_rate=int(raw.get("sample_rate", 48_000)),
            codec_steps=int(raw.get("codec_steps", 20)),
            codec_guidance=float(raw.get("codec_guidance", 1.25)),
            chunk_duration_sec=float(raw.get("chunk_duration_sec", 29.76)),
            codec_seed=int(raw.get("codec_seed", 4_242_4243)),
            source=str(raw.get("source", "heartlib-mlx")),
            source_notes=str(raw.get("source_notes", "")),
        )

    def codes_path(self, manifest_path: Path) -> Path:
        return manifest_path.parent / self.codes_file

    def reference_wav_path(self, manifest_path: Path) -> Path:
        return manifest_path.parent / self.reference_wav


def resolve_codec_parity_manifest(path: str | Path, *, project_root: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def codec_parity_fixtures_ready(manifest_path: Path) -> tuple[bool, str]:
    if not manifest_path.is_file():
        return False, f"manifest missing: {manifest_path}"
    try:
        manifest = CodecParityManifest.load(manifest_path)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return False, f"manifest invalid: {exc}"
    codes_path = manifest.codes_path(manifest_path)
    ref_path = manifest.reference_wav_path(manifest_path)
    if not codes_path.is_file():
        return False, f"codes missing: {codes_path}"
    if not ref_path.is_file():
        return False, f"reference wav missing: {ref_path}"
    return True, ""


def load_codes_npy(path: Path) -> np.ndarray:
    """Load codes as ``(T, K)`` int32 with values in ``[0, 8191]``."""
    arr = np.load(path)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"codes must be 2D (T, K) or (1, T, K); got shape {arr.shape}")
    out = np.asarray(arr, dtype=np.int32)
    if out.size == 0:
        raise ValueError("codes array is empty")
    if np.any(out < 0):
        raise ValueError("codes contain negative indices")
    if np.any(out >= 8192):
        raise ValueError("codes contain LM special tokens (>=8192); trim before codec parity")
    return out


def decode_codes_with_danqing_codec(
    bundle_root: Path,
    codes: np.ndarray,
    *,
    codec_steps: int,
    codec_guidance: float,
    chunk_duration_sec: float,
    codec_seed: int,
) -> np.ndarray:
    """Run DanQing ``HeartCodec.detokenize`` only (no LM)."""
    import mlx.core as mx

    from backend.engine.common.mlx_runtime_fallback import set_random_seed
    from backend.engine.families.heartmula.bundle import (
        mlx_weights_path,
        mlx_weights_ready,
        resolve_heartmula_bundle,
    )
    from backend.engine.families.heartmula.codec_mlx import HeartCodec, HeartCodecConfig
    from backend.engine.families.heartmula.weights_mlx import load_mlx_weights_into_module
    from backend.engine.runtime.mlx import MLXContext

    root = Path(bundle_root)
    if not mlx_weights_ready(root):
        raise RuntimeError(
            f"HeartMuLa MLX weights missing under {root}. "
            "Re-install from download center (heartmula_mlx_weights hook)."
        )
    paths = resolve_heartmula_bundle(root)
    ctx = MLXContext()
    codec_cfg = HeartCodecConfig.from_pretrained(paths.codec_torch)
    codec = HeartCodec(codec_cfg)
    load_mlx_weights_into_module(
        codec,
        mlx_weights_path(paths.codec_torch),
        dtype=mx.float32,
        eval_fn=ctx.eval,
        array_fn=ctx.array,
    )
    ctx.eval(codec.parameters())

    set_random_seed(None, int(codec_seed))
    codes_batch = ctx.array(codes[None, :, :], dtype=mx.int32)
    frame_rate = float(codec_cfg.frame_rate)
    codec_duration = codes.shape[0] / frame_rate
    audio = codec.detokenize(
        codes=codes_batch,
        duration=codec_duration,
        num_steps=int(codec_steps),
        guidance_scale=float(codec_guidance),
    )
    ctx.eval(audio)
    wf = np.array(audio.astype(mx.float32)).reshape(-1)
    wf = wf - float(wf.mean())
    peak = float(np.abs(wf).max())
    if peak < 1e-8:
        raise RuntimeError("HeartCodec decode produced near-silent audio")
    if peak > 1.0:
        wf = (wf / peak * 0.99).astype(np.float32)
    else:
        wf = wf.astype(np.float32)
    return wf


def run_codec_parity_check(
    manifest_path: Path,
    *,
    output_dir: Path,
    case_id: str = "",
    min_si_sdr_db: float = DEFAULT_MIN_SI_SDR_DB,
    min_correlation: float = DEFAULT_MIN_CORRELATION,
    warn_si_sdr_db: float = DEFAULT_WARN_SI_SDR_DB,
) -> SanityResult:
    """Codec-only parity gate (fixed codes vs heartlib reference WAV)."""
    if not mlx_runtime_available():
        return SanityResult(
            ok=True,
            reason="codec_parity_skip_no_mlx",
            mean_luma=0.0,
            std_luma=0.0,
            entropy_bits=0.0,
            laplacian_var=0.0,
            skipped=True,
        )

    ready, why = codec_parity_fixtures_ready(manifest_path)
    if not ready:
        return SanityResult(
            ok=True,
            reason=f"codec_parity_skip:{why}",
            mean_luma=0.0,
            std_luma=0.0,
            entropy_bits=0.0,
            laplacian_var=0.0,
            skipped=True,
        )

    if not heartmula_bundle_installed():
        return SanityResult(
            ok=True,
            reason="codec_parity_skip_missing_heartmula_bundle",
            mean_luma=0.0,
            std_luma=0.0,
            entropy_bits=0.0,
            laplacian_var=0.0,
            skipped=True,
        )

    manifest = CodecParityManifest.load(manifest_path)
    codes = load_codes_npy(manifest.codes_path(manifest_path))
    bundle_root = resolve_benchmark_data_root() / HEARTMULA_AUDIO_BUNDLE

    t0 = time.monotonic()
    try:
        candidate = decode_codes_with_danqing_codec(
            bundle_root,
            codes,
            codec_steps=manifest.codec_steps,
            codec_guidance=manifest.codec_guidance,
            chunk_duration_sec=manifest.chunk_duration_sec,
            codec_seed=manifest.codec_seed,
        )
    except Exception as exc:
        return SanityResult(
            ok=False,
            reason=f"codec_parity_decode_failed:{exc}",
            mean_luma=0.0,
            std_luma=0.0,
            entropy_bits=0.0,
            laplacian_var=0.0,
        )

    decode_s = time.monotonic() - t0
    out_wav = output_dir / f"{case_id or manifest.case_id}_codec_parity.wav"
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf

        sf.write(str(out_wav), candidate, manifest.sample_rate)
    except ImportError:
        pass

    compare: Optional[AudioCompareResult] = compare_audio_waveforms(
        candidate,
        manifest.reference_wav_path(manifest_path),
        sample_rate=manifest.sample_rate,
    )
    si = compare.si_sdr_db
    corr = compare.correlation
    if si is None or corr is None:
        return SanityResult(
            ok=False,
            reason=compare.product_reason or "codec_parity_compare_failed",
            mean_luma=0.0,
            std_luma=0.0,
            entropy_bits=0.0,
            laplacian_var=0.0,
        )

    # Map SI-SDR / correlation into sanity fields for summary printing.
    mean_luma = float(si)
    std_luma = float(corr)
    if si >= min_si_sdr_db and corr >= min_correlation:
        return SanityResult(
            ok=True,
            reason=(
                f"codec_parity_ok(si_sdr={si:.2f}dB,corr={corr:.4f},"
                f"decode={decode_s:.1f}s)"
            ),
            mean_luma=mean_luma,
            std_luma=std_luma,
            entropy_bits=float(compare.rmse or 0.0),
            laplacian_var=float(compare.max_abs_diff or 0.0),
        )
    if si >= warn_si_sdr_db:
        return SanityResult(
            ok=True,
            reason=(
                f"codec_parity_warn(si_sdr={si:.2f}dB<{min_si_sdr_db},"
                f"corr={corr:.4f},decode={decode_s:.1f}s)"
            ),
            mean_luma=mean_luma,
            std_luma=std_luma,
            entropy_bits=float(compare.rmse or 0.0),
            laplacian_var=float(compare.max_abs_diff or 0.0),
        )
    return SanityResult(
        ok=False,
        reason=(
            f"codec_parity_fail(si_sdr={si:.2f}dB,corr={corr:.4f},"
            f"max_diff={compare.max_abs_diff},decode={decode_s:.1f}s)"
        ),
        mean_luma=mean_luma,
        std_luma=std_luma,
        entropy_bits=float(compare.rmse or 0.0),
        laplacian_var=float(compare.max_abs_diff or 0.0),
    )
