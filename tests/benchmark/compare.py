"""
基准对比工具 — PSNR / SSIM / 像素差异 / 哈希。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


@dataclass
class CompareResult:
    """对比结果。"""
    our_hash: str = ""
    ref_hash: str = ""
    match: bool = False
    psnr: Optional[float] = None       # Peak Signal-to-Noise Ratio
    ssim: Optional[float] = None       # Structural Similarity
    pixel_max_diff: Optional[float] = None   # 最大像素差
    pixel_mean_diff: Optional[float] = None  # 平均像素差
    ours_time_sec: Optional[float] = None
    ref_time_sec: Optional[float] = None


def hash_image(path: str | Path) -> str:
    """计算图像的 SHA256 哈希。"""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def load_as_array(path: str | Path) -> np.ndarray:
    """加载图像为 [H, W, C] float32 numpy 数组（值域 0-1）。"""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算 PSNR (Peak Signal-to-Noise Ratio)。

    值域 0-1，PSNR = 20*log10(1.0 / RMSE)。
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0  # 完全一致
    return float(20 * np.log10(1.0 / np.sqrt(mse)))


def compute_ssim(img1: np.ndarray, img2: np.ndarray,
                 data_range: float = 1.0,
                 win_size: int = 7) -> float:
    """计算 SSIM (Structural Similarity Index)。

    滑动窗口版本，C1/C2 按 skimage 惯例。
    """
    from scipy.ndimage import uniform_filter

    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # 灰度转换（亮度通道）
    if img1.ndim == 3 and img1.shape[-1] == 3:
        w = np.array([0.299, 0.587, 0.114])
        img1 = np.dot(img1, w)
        img2 = np.dot(img2, w)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = uniform_filter(img1, win_size)
    mu2 = uniform_filter(img2, win_size)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2
    sigma1_sq = uniform_filter(img1 ** 2, win_size) - mu1_sq
    sigma2_sq = uniform_filter(img2 ** 2, win_size) - mu2_sq
    sigma12 = uniform_filter(img1 * img2, win_size) - mu12

    num = (2 * mu12 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / den

    return float(np.mean(ssim_map))


def compute_pixel_diff(img1: np.ndarray, img2: np.ndarray) -> tuple[float, float]:
    """计算像素差异：最大差和平均差。"""
    diff = np.abs(img1 - img2)
    return float(diff.max()), float(diff.mean())


def compare_images(our_path: str | Path, ref_path: str | Path) -> CompareResult:
    """对比两张图像。"""
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
        pixel_max_diff=compute_pixel_diff(our_arr, ref_arr)[0],
        pixel_mean_diff=compute_pixel_diff(our_arr, ref_arr)[1],
    )


def compare_images_match_ref_size(our_path: str | Path, ref_path: str | Path) -> CompareResult:
    """对比两张图；若 H×W 不一致，将 ``our`` 双线性缩放到 ``ref`` 尺寸后再算 PSNR/SSIM（视频抽帧常见）。"""
    our_arr = load_as_array(our_path)
    ref_arr = load_as_array(ref_path)
    if our_arr.ndim != 3 or ref_arr.ndim != 3 or our_arr.shape[-1] != 3 or ref_arr.shape[-1] != 3:
        return CompareResult(
            our_hash=hash_image(our_path),
            ref_hash=hash_image(ref_path),
            match=False,
        )
    h0, w0 = our_arr.shape[0], our_arr.shape[1]
    h1, w1 = ref_arr.shape[0], ref_arr.shape[1]
    if (h0, w0) != (h1, w1):
        pil = Image.fromarray((our_arr * 255.0).clip(0, 255).astype(np.uint8))
        pil = pil.resize((w1, h1), Image.Resampling.BICUBIC)
        our_arr = np.asarray(pil, dtype=np.float32) / 255.0

    return CompareResult(
        our_hash=hash_image(our_path),
        ref_hash=hash_image(ref_path),
        match=hash_image(our_path) == hash_image(ref_path),
        psnr=compute_psnr(our_arr, ref_arr),
        ssim=compute_ssim(our_arr, ref_arr),
        pixel_max_diff=compute_pixel_diff(our_arr, ref_arr)[0],
        pixel_mean_diff=compute_pixel_diff(our_arr, ref_arr)[1],
    )
