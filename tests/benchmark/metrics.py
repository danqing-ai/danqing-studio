"""Benchmark metrics: reference compare + quality sanity with anti-noise gates."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image
from scipy.ndimage import laplace, sobel, uniform_filter


@dataclass
class CompareResult:
    our_hash: str = ""
    ref_hash: str = ""
    match: bool = False
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    pixel_max_diff: Optional[float] = None
    pixel_mean_diff: Optional[float] = None
    ours_time_sec: Optional[float] = None
    ref_time_sec: Optional[float] = None
    product_ok: Optional[bool] = None
    product_reason: str = ""


@dataclass
class AudioCompareResult:
    """Waveform parity vs a reference clip (codec parity, E2E audio ref)."""

    si_sdr_db: Optional[float] = None
    correlation: Optional[float] = None
    rmse: Optional[float] = None
    max_abs_diff: Optional[float] = None
    length_samples: int = 0
    ref_length_samples: int = 0
    product_ok: Optional[bool] = None
    product_reason: str = ""


@dataclass
class SanityResult:
    ok: bool
    reason: str
    mean_luma: float
    std_luma: float
    entropy_bits: float
    laplacian_var: float
    skipped: bool = False
    score: float = 0.0
    subscores: dict[str, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "mean_luma": self.mean_luma,
            "std_luma": self.std_luma,
            "entropy_bits": self.entropy_bits,
            "laplacian_var": self.laplacian_var,
            "skipped": self.skipped,
            "score": self.score,
            "subscores": self.subscores or {},
        }


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _safe_score(x: float) -> float:
    return float(np.clip(x, 0.0, 100.0))


def _entropy_bits(gray: np.ndarray, bins: int = 64) -> float:
    hist, _ = np.histogram(gray.ravel(), bins=bins, range=(0.0, 1.0))
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    p = hist.astype(np.float64) / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p + 1e-30)))


def _high_freq_ratio(gray: np.ndarray) -> float:
    spec = np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))
    power = np.abs(spec) ** 2
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = float(rr.max()) + 1e-9
    high = power[rr >= 0.35 * rmax].sum()
    total = power.sum() + 1e-12
    return float(high / total)


def _neighbor_diff(gray: np.ndarray) -> float:
    dx = np.abs(gray[:, 1:] - gray[:, :-1]).mean()
    dy = np.abs(gray[1:, :] - gray[:-1, :]).mean()
    return float((dx + dy) * 0.5)


def _edge_coherence(gray: np.ndarray) -> tuple[float, float]:
    gx = sobel(gray, axis=1, mode="reflect")
    gy = sobel(gray, axis=0, mode="reflect")
    mag = np.sqrt(gx**2 + gy**2)
    edge_density = float(np.mean(mag > np.percentile(mag, 80)))

    strong = mag > np.percentile(mag, 90)
    if not np.any(strong):
        return edge_density, 0.0
    theta = np.arctan2(gy[strong], gx[strong])
    # 2*theta resolves 180° direction ambiguity for gradient orientation.
    coh = np.hypot(np.mean(np.cos(2.0 * theta)), np.mean(np.sin(2.0 * theta)))
    return edge_density, float(coh)


def hash_image(path: str | Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def load_as_array(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return float(20 * np.log10(1.0 / np.sqrt(mse)))


def compute_ssim(img1: np.ndarray, img2: np.ndarray, data_range: float = 1.0, win_size: int = 7) -> float:
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    if img1.ndim == 3 and img1.shape[-1] == 3:
        w = np.array([0.299, 0.587, 0.114])
        img1 = np.dot(img1, w)
        img2 = np.dot(img2, w)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mu1 = uniform_filter(img1, win_size)
    mu2 = uniform_filter(img2, win_size)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = uniform_filter(img1 ** 2, win_size) - mu1_sq
    sigma2_sq = uniform_filter(img2 ** 2, win_size) - mu2_sq
    sigma12 = uniform_filter(img1 * img2, win_size) - mu12
    num = (2 * mu12 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return float(np.mean(num / den))


def compare_images(our_path: str | Path, ref_path: str | Path) -> CompareResult:
    our_arr = load_as_array(our_path)
    ref_arr = load_as_array(ref_path)
    if our_arr.shape != ref_arr.shape:
        return CompareResult(
            our_hash=hash_image(our_path),
            ref_hash=hash_image(ref_path),
            match=False,
        )
    return CompareResult(
        our_hash=hash_image(our_path),
        ref_hash=hash_image(ref_path),
        match=hash_image(our_path) == hash_image(ref_path),
        psnr=compute_psnr(our_arr, ref_arr),
        ssim=compute_ssim(our_arr, ref_arr),
        pixel_max_diff=float(np.abs(our_arr - ref_arr).max()),
        pixel_mean_diff=float(np.abs(our_arr - ref_arr).mean()),
    )


def _load_video_sampled_gray(path: str | Path, *, max_frames: int = 12) -> list[np.ndarray]:
    import imageio.v3 as iio

    path = Path(path)
    meta = iio.immeta(path)
    n = int(meta.get("n_images", 0) or 0)
    if n < 1:
        n = 1
    picks = np.linspace(0, max(0, n - 1), num=min(max_frames, max(3, n)), dtype=int)
    indices = sorted({int(i) for i in picks.tolist()})
    frames: list[np.ndarray] = []
    for idx in indices:
        arr = np.asarray(iio.imread(path, index=idx), dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            arr = arr[..., :3] / 255.0
            gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        else:
            gray = arr.astype(np.float32)
            if gray.max() > 1.5:
                gray = gray / 255.0
        frames.append(gray.astype(np.float32))
    return frames


def compare_videos(our_path: str | Path, ref_path: str | Path) -> CompareResult:
    try:
        ours = _load_video_sampled_gray(our_path)
        refs = _load_video_sampled_gray(ref_path)
    except Exception:
        return CompareResult(
            our_hash=hash_image(our_path),
            ref_hash=hash_image(ref_path),
            match=False,
        )
    if not ours or not refs:
        return CompareResult(
            our_hash=hash_image(our_path),
            ref_hash=hash_image(ref_path),
            match=False,
        )
    n = min(len(ours), len(refs))
    ours = ours[:n]
    refs = refs[:n]
    psnrs: list[float] = []
    ssims: list[float] = []
    max_diff: list[float] = []
    mean_diff: list[float] = []
    for a, b in zip(ours, refs):
        if a.shape != b.shape:
            return CompareResult(
                our_hash=hash_image(our_path),
                ref_hash=hash_image(ref_path),
                match=False,
            )
        psnrs.append(compute_psnr(a, b))
        ssims.append(compute_ssim(a, b))
        d = np.abs(a - b)
        max_diff.append(float(d.max()))
        mean_diff.append(float(d.mean()))
    return CompareResult(
        our_hash=hash_image(our_path),
        ref_hash=hash_image(ref_path),
        match=hash_image(our_path) == hash_image(ref_path),
        psnr=float(np.mean(psnrs)),
        ssim=float(np.mean(ssims)),
        pixel_max_diff=float(np.mean(max_diff)),
        pixel_mean_diff=float(np.mean(mean_diff)),
    )


def check_output_image(path: str | Path) -> SanityResult:
    return check_output_image_with_thresholds(path, thresholds=None)


def check_output_image_with_thresholds(
    path: str | Path,
    *,
    thresholds: dict[str, float] | None,
) -> SanityResult:
    path = Path(path)
    try:
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    except Exception as e:
        return SanityResult(False, f"load_error:{e!r}", 0.0, 0.0, 0.0, 0.0)

    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    mean_l = float(np.mean(gray))
    std_l = float(np.std(gray))
    ent = _entropy_bits(gray, bins=64)
    lap_var = float(np.var(laplace(gray.astype(np.float64))))
    hf_ratio = _high_freq_ratio(gray)
    n_diff = _neighbor_diff(gray)
    edge_density, edge_coh = _edge_coherence(gray)

    cfg = {
        "min_std": 0.014,
        "min_ent": 1.25,
        "blank_mean_high": 0.94,
        "blank_mean_low": 0.06,
        "blank_std_max": 0.055,
        "noise_hf_high": 0.58,
        "noise_neighbor_high": 0.16,
        "noise_edge_coh_low": 0.14,
        "min_score": 70.0,
    }
    if thresholds:
        cfg.update({k: float(v) for k, v in thresholds.items()})

    min_std = float(cfg["min_std"])
    min_ent = float(cfg["min_ent"])
    if std_l < min_std:
        return SanityResult(
            False,
            f"near_uniform(std={std_l:.5f})",
            mean_l,
            std_l,
            ent,
            lap_var,
            score=5.0,
            subscores={"integrity": 30.0, "anti_garbage": 0.0, "semantic_proxy": 0.0},
        )
    if mean_l >= float(cfg["blank_mean_high"]) and std_l <= float(cfg["blank_std_max"]):
        return SanityResult(
            False,
            f"blank_white(mean={mean_l:.3f})",
            mean_l,
            std_l,
            ent,
            lap_var,
            score=5.0,
            subscores={"integrity": 30.0, "anti_garbage": 0.0, "semantic_proxy": 0.0},
        )
    if mean_l <= float(cfg["blank_mean_low"]) and std_l <= float(cfg["blank_std_max"]):
        return SanityResult(
            False,
            f"blank_black(mean={mean_l:.3f})",
            mean_l,
            std_l,
            ent,
            lap_var,
            score=5.0,
            subscores={"integrity": 30.0, "anti_garbage": 0.0, "semantic_proxy": 0.0},
        )
    if ent < min_ent:
        return SanityResult(
            False,
            f"low_entropy(bits={ent:.2f})",
            mean_l,
            std_l,
            ent,
            lap_var,
            score=10.0,
            subscores={"integrity": 40.0, "anti_garbage": 0.0, "semantic_proxy": 0.0},
        )

    if (
        hf_ratio > float(cfg["noise_hf_high"])
        and n_diff > float(cfg["noise_neighbor_high"])
        and edge_coh < float(cfg["noise_edge_coh_low"])
    ):
        return SanityResult(
            False,
            (
                "noise_like_texture("
                f"hf={hf_ratio:.3f},adj={n_diff:.3f},edge_coh={edge_coh:.3f})"
            ),
            mean_l,
            std_l,
            ent,
            lap_var,
            score=20.0,
            subscores={"integrity": 70.0, "anti_garbage": 10.0, "semantic_proxy": 10.0},
        )

    integrity = 100.0 * _clip01(1.0 - abs(mean_l - 0.5) / 0.5)
    anti_garbage = 100.0 * _clip01(
        0.55 * (1.0 - max(0.0, hf_ratio - 0.38) / 0.45)
        + 0.25 * _clip01(edge_coh / 0.45)
        + 0.20 * (1.0 - _clip01(n_diff / 0.25))
    )
    semantic_proxy = 100.0 * _clip01(
        0.45 * _clip01((ent - 1.2) / 3.5)
        + 0.35 * _clip01(edge_density / 0.22)
        + 0.20 * _clip01((lap_var - 0.0015) / 0.02)
    )
    score = _safe_score(0.5 * anti_garbage + 0.3 * semantic_proxy + 0.2 * integrity)

    if score < float(cfg["min_score"]):
        return SanityResult(
            False,
            f"low_quality_score(score={score:.1f})",
            mean_l,
            std_l,
            ent,
            lap_var,
            score=score,
            subscores={
                "integrity": _safe_score(integrity),
                "anti_garbage": _safe_score(anti_garbage),
                "semantic_proxy": _safe_score(semantic_proxy),
            },
        )
    return SanityResult(
        True,
        "",
        mean_l,
        std_l,
        ent,
        lap_var,
        score=score,
        subscores={
            "integrity": _safe_score(integrity),
            "anti_garbage": _safe_score(anti_garbage),
            "semantic_proxy": _safe_score(semantic_proxy),
        },
    )


def _load_mono_wav(path: str | Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.ndim == 2 else np.asarray(data, dtype=np.float32)
    return np.asarray(mono, dtype=np.float64).reshape(-1), int(sr)


def si_sdr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant SDR (dB) after mean removal."""
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    est = np.asarray(estimate, dtype=np.float64).reshape(-1)
    n = min(ref.size, est.size)
    if n < 64:
        return float("-inf")
    ref = ref[:n] - float(ref[:n].mean())
    est = est[:n] - float(est[:n].mean())
    denom = float(np.dot(ref, ref)) + 1e-12
    scale = float(np.dot(est, ref)) / denom
    projection = scale * ref
    noise = est - projection
    signal_power = float(np.sum(projection**2)) + 1e-12
    noise_power = float(np.sum(noise**2)) + 1e-12
    return float(10.0 * np.log10(signal_power / noise_power))


def compare_audio_waveforms(
    candidate: np.ndarray,
    reference: np.ndarray | Path | str,
    *,
    sample_rate: int = 48_000,
    resample_reference: bool = True,
) -> AudioCompareResult:
    """Compare mono waveforms; reference may be array or WAV path."""
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if isinstance(reference, (str, Path)):
        try:
            ref, ref_sr = _load_mono_wav(reference)
        except Exception as exc:
            return AudioCompareResult(
                product_ok=False,
                product_reason=f"ref_load_error:{exc!r}",
            )
        if resample_reference and ref_sr != sample_rate:
            return AudioCompareResult(
                product_ok=False,
                product_reason=f"ref_sample_rate_mismatch:{ref_sr}!={sample_rate}",
            )
    else:
        ref = np.asarray(reference, dtype=np.float64).reshape(-1)

    n = min(cand.size, ref.size)
    if n < 64:
        return AudioCompareResult(
            product_ok=False,
            product_reason="too_short_for_compare",
            length_samples=int(cand.size),
            ref_length_samples=int(ref.size),
        )
    cand = cand[:n]
    ref = ref[:n]
    diff = cand - ref
    rmse = float(np.sqrt(np.mean(diff**2)))
    max_abs = float(np.max(np.abs(diff)))
    corr_mat = np.corrcoef(cand, ref)
    corr = float(corr_mat[0, 1]) if corr_mat.shape == (2, 2) else 0.0
    si = si_sdr_db(ref, cand)
    return AudioCompareResult(
        si_sdr_db=si,
        correlation=corr,
        rmse=rmse,
        max_abs_diff=max_abs,
        length_samples=int(n),
        ref_length_samples=int(ref.size),
        product_ok=True,
        product_reason="",
    )


def check_output_audio(
    path: str | Path,
    *,
    min_rms: float = 0.01,
    min_peak: float = 0.02,
) -> SanityResult:
    return check_output_audio_with_thresholds(
        path,
        min_rms=min_rms,
        min_peak=min_peak,
        thresholds=None,
    )


def check_output_audio_with_thresholds(
    path: str | Path,
    *,
    min_rms: float = 0.01,
    min_peak: float = 0.02,
    thresholds: dict[str, float] | None,
) -> SanityResult:
    """Reject near-silent / white-noise-like audio without a reference clip."""
    path = Path(path)
    try:
        import soundfile as sf
    except ImportError as e:
        return SanityResult(False, f"soundfile_missing:{e!r}", 0.0, 0.0, 0.0, 0.0)

    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as e:
        return SanityResult(False, f"load_error:{e!r}", 0.0, 0.0, 0.0, 0.0)

    mono = data.mean(axis=1) if data.ndim == 2 else data
    if mono.size < 2048:
        return SanityResult(False, "too_short_audio", 0.0, 0.0, 0.0, 0.0)

    rms = float(np.sqrt(np.mean(mono**2)))
    peak = float(np.max(np.abs(mono)))
    cfg = {
        "min_rms": min_rms,
        "min_peak": min_peak,
        "noise_flatness_high": 0.72,
        "noise_band_ratio_high": 0.86,
        "low_dyn_std": 0.0025,
        "low_dyn_flatness": 0.55,
        "min_score": 70.0,
    }
    if thresholds:
        cfg.update({k: float(v) for k, v in thresholds.items()})

    if rms < float(cfg["min_rms"]):
        return SanityResult(False, f"near_silent(rms={rms:.5f})", rms, peak, 0.0, 0.0)
    if peak < float(cfg["min_peak"]):
        return SanityResult(False, f"low_peak(peak={peak:.5f})", rms, peak, 0.0, 0.0)

    frame = 2048
    hop = 512
    window = np.hanning(frame).astype(np.float32)
    chunks = []
    for i in range(0, max(1, mono.size - frame + 1), hop):
        seg = mono[i:i + frame]
        if seg.size < frame:
            pad = np.zeros(frame, dtype=np.float32)
            pad[:seg.size] = seg
            seg = pad
        chunks.append(seg * window)
    stft = np.abs(np.fft.rfft(np.stack(chunks, axis=0), axis=1)) ** 2
    stft += 1e-12

    flatness = float(np.mean(np.exp(np.mean(np.log(stft), axis=1)) / np.mean(stft, axis=1)))
    freqs = np.fft.rfftfreq(frame, d=1.0 / max(1, int(sr)))
    band = (freqs >= 80.0) & (freqs <= 12000.0)
    band_energy = float(stft[:, band].sum())
    total_energy = float(stft.sum()) + 1e-12
    band_ratio = band_energy / total_energy
    frame_rms = np.sqrt(np.mean(np.square(np.stack(chunks, axis=0)), axis=1))
    dyn_std = float(np.std(frame_rms))

    if flatness > float(cfg["noise_flatness_high"]) and band_ratio > float(cfg["noise_band_ratio_high"]):
        return SanityResult(
            False,
            f"noise_like_audio(flatness={flatness:.3f},band={band_ratio:.3f})",
            rms,
            peak,
            flatness,
            dyn_std,
            score=20.0,
            subscores={"integrity": 80.0, "anti_garbage": 10.0, "semantic_proxy": 10.0},
        )
    if dyn_std < float(cfg["low_dyn_std"]) and flatness > float(cfg["low_dyn_flatness"]):
        return SanityResult(
            False,
            f"low_dynamics(dyn={dyn_std:.4f},flatness={flatness:.3f})",
            rms,
            peak,
            flatness,
            dyn_std,
            score=25.0,
            subscores={"integrity": 80.0, "anti_garbage": 20.0, "semantic_proxy": 10.0},
        )

    integrity = 100.0 * _clip01(min(1.0, peak / 0.8) * min(1.0, rms / 0.08))
    anti_garbage = 100.0 * _clip01(
        0.55 * (1.0 - _clip01((flatness - 0.2) / 0.65))
        + 0.25 * _clip01((band_ratio - 0.35) / 0.55)
        + 0.20 * _clip01(dyn_std / 0.025)
    )
    semantic_proxy = 100.0 * _clip01(
        0.5 * _clip01((rms - 0.01) / 0.08)
        + 0.3 * _clip01(dyn_std / 0.03)
        + 0.2 * _clip01((1.0 - flatness) / 0.8)
    )
    score = _safe_score(0.5 * anti_garbage + 0.3 * semantic_proxy + 0.2 * integrity)
    if score < float(cfg["min_score"]):
        return SanityResult(
            False,
            f"low_quality_score(score={score:.1f})",
            rms,
            peak,
            flatness,
            dyn_std,
            score=score,
            subscores={
                "integrity": _safe_score(integrity),
                "anti_garbage": _safe_score(anti_garbage),
                "semantic_proxy": _safe_score(semantic_proxy),
            },
        )
    return SanityResult(
        True,
        "",
        rms,
        peak,
        flatness,
        dyn_std,
        score=score,
        subscores={
            "integrity": _safe_score(integrity),
            "anti_garbage": _safe_score(anti_garbage),
            "semantic_proxy": _safe_score(semantic_proxy),
        },
    )


def check_output_video(path: str | Path, *, min_frame_std: float = 0.02) -> SanityResult:
    return check_output_video_with_thresholds(path, min_frame_std=min_frame_std, thresholds=None)


def check_output_video_with_thresholds(
    path: str | Path,
    *,
    min_frame_std: float = 0.02,
    thresholds: dict[str, float] | None,
) -> SanityResult:
    """Reject blank / static / noise-flicker MP4 outputs (sample sparse frames)."""
    path = Path(path)
    if not path.is_file():
        return SanityResult(False, "missing_file", 0.0, 0.0, 0.0, 0.0)
    if path.stat().st_size < 8192:
        return SanityResult(False, f"tiny_file(bytes={path.stat().st_size})", 0.0, 0.0, 0.0, 0.0)

    frames: list[np.ndarray] = []
    try:
        import imageio.v3 as iio

        meta = iio.immeta(path)
        n = int(meta.get("n_images", 0) or 0)
        if n < 1:
            n = 1
        picks = np.linspace(0, max(0, n - 1), num=min(8, max(3, n)), dtype=int)
        indices = sorted({int(i) for i in picks.tolist()})
        for idx in indices:
            frame = np.asarray(iio.imread(path, index=idx), dtype=np.float32)
            if frame.ndim == 3 and frame.shape[-1] >= 3:
                frame = frame[..., :3] / 255.0
            frames.append(frame)
    except Exception as e:
        return SanityResult(False, f"video_read_error:{e!r}", 0.0, 0.0, 0.0, 0.0)

    if not frames:
        return SanityResult(False, "no_frames", 0.0, 0.0, 0.0, 0.0)

    stds: list[float] = []
    means: list[float] = []
    frame_entropy: list[float] = []
    frame_hf: list[float] = []
    frame_edge_coh: list[float] = []
    gray_frames: list[np.ndarray] = []
    for arr in frames:
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        gray_frames.append(gray.astype(np.float64))
        means.append(float(np.mean(gray)))
        stds.append(float(np.std(gray)))
        frame_entropy.append(_entropy_bits(gray, bins=64))
        frame_hf.append(_high_freq_ratio(gray))
        _edge_density, coh = _edge_coherence(gray)
        frame_edge_coh.append(coh)

    mean_l = float(np.mean(means))
    std_l = float(np.mean(stds))
    lap_var = float(np.mean([np.var(laplace(g)) for g in gray_frames]))
    ent = float(np.mean(frame_entropy))
    avg_hf = float(np.mean(frame_hf))
    avg_edge_coh = float(np.mean(frame_edge_coh))

    pair_ssim: list[float] = []
    pair_delta: list[float] = []
    for i in range(len(gray_frames) - 1):
        g1 = gray_frames[i]
        g2 = gray_frames[i + 1]
        if g1.shape != g2.shape:
            continue
        pair_ssim.append(compute_ssim(g1, g2))
        pair_delta.append(float(np.mean(np.abs(g1 - g2))))
    temporal_ssim = float(np.mean(pair_ssim)) if pair_ssim else 0.0
    temporal_delta = float(np.mean(pair_delta)) if pair_delta else 0.0

    cfg = {
        "min_frame_std": min_frame_std,
        "blank_mean_high": 0.94,
        "blank_mean_low": 0.06,
        "blank_std_max": 0.04,
        "noise_hf_high": 0.58,
        "noise_temporal_ssim_low": 0.08,
        "noise_temporal_delta_high": 0.20,
        "min_score": 70.0,
    }
    if thresholds:
        cfg.update({k: float(v) for k, v in thresholds.items()})

    if std_l < float(cfg["min_frame_std"]):
        return SanityResult(
            False, f"near_flat_frames(avg_std={std_l:.5f})", mean_l, std_l, ent, lap_var,
        )
    if mean_l >= float(cfg["blank_mean_high"]) and std_l <= float(cfg["blank_std_max"]):
        return SanityResult(False, f"blank_white(mean={mean_l:.3f})", mean_l, std_l, ent, lap_var)
    if mean_l <= float(cfg["blank_mean_low"]) and std_l <= float(cfg["blank_std_max"]):
        return SanityResult(False, f"blank_black(mean={mean_l:.3f})", mean_l, std_l, ent, lap_var)
    if (
        avg_hf > float(cfg["noise_hf_high"])
        and temporal_ssim < float(cfg["noise_temporal_ssim_low"])
        and temporal_delta > float(cfg["noise_temporal_delta_high"])
    ):
        return SanityResult(
            False,
            (
                "noise_flicker("
                f"hf={avg_hf:.3f},t_ssim={temporal_ssim:.3f},t_delta={temporal_delta:.3f})"
            ),
            mean_l,
            std_l,
            ent,
            lap_var,
            score=20.0,
            subscores={"integrity": 70.0, "anti_garbage": 10.0, "semantic_proxy": 10.0},
        )

    integrity = 100.0 * _clip01(1.0 - abs(mean_l - 0.5) / 0.5)
    anti_garbage = 100.0 * _clip01(
        0.45 * (1.0 - _clip01((avg_hf - 0.35) / 0.5))
        + 0.30 * _clip01(avg_edge_coh / 0.45)
        + 0.25 * _clip01((temporal_ssim - 0.08) / 0.35)
    )
    semantic_proxy = 100.0 * _clip01(
        0.4 * _clip01((ent - 1.3) / 3.3)
        + 0.35 * _clip01((temporal_ssim - 0.05) / 0.4)
        + 0.25 * (1.0 - _clip01((temporal_delta - 0.02) / 0.5))
    )
    score = _safe_score(0.5 * anti_garbage + 0.3 * semantic_proxy + 0.2 * integrity)
    if score < float(cfg["min_score"]):
        return SanityResult(
            False,
            f"low_quality_score(score={score:.1f})",
            mean_l,
            std_l,
            ent,
            lap_var,
            score=score,
            subscores={
                "integrity": _safe_score(integrity),
                "anti_garbage": _safe_score(anti_garbage),
                "semantic_proxy": _safe_score(semantic_proxy),
            },
        )
    return SanityResult(
        True,
        "",
        mean_l,
        std_l,
        ent,
        lap_var,
        score=score,
        subscores={
            "integrity": _safe_score(integrity),
            "anti_garbage": _safe_score(anti_garbage),
            "semantic_proxy": _safe_score(semantic_proxy),
        },
    )
