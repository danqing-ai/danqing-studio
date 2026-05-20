"""Image metrics: mflux PSNR/SSIM compare + output sanity (reject flat black/white)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image
from scipy.ndimage import laplace, uniform_filter


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


@dataclass
class SanityResult:
    ok: bool
    reason: str
    mean_luma: float
    std_luma: float
    entropy_bits: float
    laplacian_var: float
    skipped: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "mean_luma": self.mean_luma,
            "std_luma": self.std_luma,
            "entropy_bits": self.entropy_bits,
            "laplacian_var": self.laplacian_var,
            "skipped": self.skipped,
        }


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


def check_output_image(path: str | Path) -> SanityResult:
    path = Path(path)
    try:
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    except Exception as e:
        return SanityResult(False, f"load_error:{e!r}", 0.0, 0.0, 0.0, 0.0)

    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    mean_l = float(np.mean(gray))
    std_l = float(np.std(gray))
    hist, _ = np.histogram(gray.ravel(), bins=32, range=(0.0, 1.0))
    total = float(hist.sum())
    ent = 0.0
    if total > 0:
        p = hist.astype(np.float64) / total
        p = p[p > 0]
        ent = float(-np.sum(p * np.log2(p + 1e-30)))
    lap_var = float(np.var(laplace(gray.astype(np.float64))))

    min_std, min_ent = 0.014, 1.25
    if std_l < min_std:
        return SanityResult(False, f"near_uniform(std={std_l:.5f})", mean_l, std_l, ent, lap_var)
    if mean_l >= 0.94 and std_l <= 0.055:
        return SanityResult(False, f"blank_white(mean={mean_l:.3f})", mean_l, std_l, ent, lap_var)
    if mean_l <= 0.06 and std_l <= 0.055:
        return SanityResult(False, f"blank_black(mean={mean_l:.3f})", mean_l, std_l, ent, lap_var)
    if ent < min_ent:
        return SanityResult(False, f"low_entropy(bits={ent:.2f})", mean_l, std_l, ent, lap_var)
    return SanityResult(True, "", mean_l, std_l, ent, lap_var)


def check_output_audio(
    path: str | Path,
    *,
    min_rms: float = 0.01,
    min_peak: float = 0.02,
) -> SanityResult:
    """Reject near-silent or flat ACE-Step wav/mp3 outputs (no reference upstream)."""
    path = Path(path)
    try:
        import soundfile as sf
    except ImportError as e:
        return SanityResult(False, f"soundfile_missing:{e!r}", 0.0, 0.0, 0.0, 0.0)

    try:
        data, _sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as e:
        return SanityResult(False, f"load_error:{e!r}", 0.0, 0.0, 0.0, 0.0)

    mono = data.mean(axis=1) if data.ndim == 2 else data
    rms = float(np.sqrt(np.mean(mono**2)))
    peak = float(np.max(np.abs(mono)))
    if rms < min_rms:
        return SanityResult(False, f"near_silent(rms={rms:.5f})", rms, peak, 0.0, 0.0)
    if peak < min_peak:
        return SanityResult(False, f"low_peak(peak={peak:.5f})", rms, peak, 0.0, 0.0)
    return SanityResult(True, "", rms, peak, 0.0, 0.0)
