"""LTX video tail ingest + segment append (ffmpeg, fail loud)."""
from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np

from backend.engine.families.ltx.long_video_plan import num_frames_for_duration_sec
from backend.engine.families.ltx.pipeline_math import AUDIO_SAMPLE_RATE, VIDEO_SPATIAL_SCALE
from backend.utils.video_sr_ffmpeg import require_ffmpeg, require_ffprobe


def _snap_dim(value: int, *, multiple: int = VIDEO_SPATIAL_SCALE) -> int:
    v = max(multiple, int(value))
    return ((v + multiple - 1) // multiple) * multiple


def extract_video_tail(
    video_path: Path,
    *,
    duration_sec: float,
    fps: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract last ``duration_sec`` of video as RGB + mono 16 kHz waveform.

    Returns:
        rgb: ``(F, H, W, 3)`` float32 in ``[-1, 1]``
        audio: ``(T,)`` float32 mono at ``AUDIO_SAMPLE_RATE``
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise RuntimeError(f"LTX extend: video not found: {video_path}")

    ffmpeg = require_ffmpeg()
    require_ffprobe()
    w = _snap_dim(width)
    h = _snap_dim(height)
    rate = max(1.0, float(fps))
    target_frames = num_frames_for_duration_sec(duration_sec, rate)

    vf = (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,fps={rate:.6f},format=rgb24"
    )
    cmd_v = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-sseof",
        f"-{max(0.01, float(duration_sec)):.6f}",
        "-i",
        str(video_path),
        "-an",
        "-vf",
        vf,
        "-frames:v",
        str(target_frames),
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    proc_v = subprocess.run(cmd_v, capture_output=True, check=False)
    if proc_v.returncode != 0:
        err = (proc_v.stderr or proc_v.stdout or b"").decode(errors="ignore")
        raise RuntimeError(f"LTX extend: ffmpeg video tail extract failed: {err[:1200]}")

    raw = proc_v.stdout
    frame_bytes = w * h * 3
    if frame_bytes <= 0 or len(raw) < frame_bytes:
        raise RuntimeError(
            f"LTX extend: insufficient video tail frames from {video_path} "
            f"(got {len(raw)} bytes, need >= {frame_bytes})"
        )
    n_frames = len(raw) // frame_bytes
    rgb = np.frombuffer(raw[: n_frames * frame_bytes], dtype=np.uint8).reshape(n_frames, h, w, 3)
    rgb = rgb.astype(np.float32) / 127.5 - 1.0

    cmd_a = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-sseof",
        f"-{max(0.01, float(duration_sec)):.6f}",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        "-f",
        "f32le",
        "pipe:1",
    ]
    proc_a = subprocess.run(cmd_a, capture_output=True, check=False)
    if proc_a.returncode != 0:
        err = (proc_a.stderr or proc_a.stdout or b"").decode(errors="ignore")
        raise RuntimeError(f"LTX extend: ffmpeg audio tail extract failed: {err[:1200]}")
    audio = np.frombuffer(proc_a.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        audio = np.zeros((1,), dtype=np.float32)
    return rgb, audio


def _read_wav_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        rate = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
    if sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"LTX extend: unsupported WAV sample width {sw}")
    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)
    if rate != AUDIO_SAMPLE_RATE and data.size > 0:
        # Simple resample via ffmpeg round-trip
        ffmpeg = require_ffmpeg()
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
            in_path = tmp_in.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            out_path = tmp_out.name
        try:
            with wave.open(in_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(rate)
                wf.writeframes((np.clip(data, -1, 1) * 32767).astype(np.int16).tobytes())
            cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                in_path,
                "-ac",
                "1",
                "-ar",
                str(AUDIO_SAMPLE_RATE),
                out_path,
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return _read_wav_mono(Path(out_path))
        finally:
            Path(in_path).unlink(missing_ok=True)
            Path(out_path).unlink(missing_ok=True)
    return data.astype(np.float32)


def _extract_video_rgb(video: Path, *, fps: float) -> np.ndarray:
    ffmpeg = require_ffmpeg()
    probe = subprocess.run(
        [ffmpeg, "-i", str(video), "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    import re

    m = re.search(r",\s*(\d{2,5})x(\d{2,5})", probe.stderr or "")
    if not m:
        raise RuntimeError(f"LTX extend: cannot probe size of {video}")
    w, h = int(m.group(1)), int(m.group(2))
    rate = max(1.0, float(fps))
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"fps={rate:.6f},format=rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"LTX extend: rgb extract failed for {video}")
    fb = w * h * 3
    n = len(proc.stdout) // fb
    return np.frombuffer(proc.stdout[: n * fb], dtype=np.uint8).reshape(n, h, w, 3).astype(np.float32) / 127.5 - 1.0


def append_video_with_crossfade(
    *,
    accumulator: Path,
    new_segment: Path,
    output: Path,
    overlap_frames: int,
    fps: float,
) -> None:
    """Append ``new_segment`` to ``accumulator`` with linear crossfade on overlap."""
    acc = Path(accumulator)
    seg = Path(new_segment)
    out = Path(output)
    if not acc.is_file():
        raise RuntimeError(f"LTX extend: accumulator missing: {acc}")
    if not seg.is_file():
        raise RuntimeError(f"LTX extend: new segment missing: {seg}")

    overlap = max(0, int(overlap_frames))
    rate = max(1.0, float(fps))
    acc_rgb = _extract_video_rgb(acc, fps=rate)
    seg_rgb = _extract_video_rgb(seg, fps=rate)

    ffmpeg = require_ffmpeg()
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ta:
        acc_wav = ta.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ts:
        seg_wav = ts.name
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(acc),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(AUDIO_SAMPLE_RATE),
                acc_wav,
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(seg),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(AUDIO_SAMPLE_RATE),
                seg_wav,
            ],
            check=True,
            capture_output=True,
        )
        acc_a = _read_wav_mono(Path(acc_wav))
        seg_a = _read_wav_mono(Path(seg_wav))
    finally:
        Path(acc_wav).unlink(missing_ok=True)
        Path(seg_wav).unlink(missing_ok=True)

    if overlap <= 0 or acc_rgb.shape[0] == 0 or seg_rgb.shape[0] == 0:
        merged_rgb = np.concatenate([acc_rgb, seg_rgb], axis=0)
        merged_a = np.concatenate([acc_a, seg_a], axis=0)
    else:
        ov = min(overlap, acc_rgb.shape[0], seg_rgb.shape[0])
        weights = np.linspace(0.0, 1.0, ov, dtype=np.float32)[:, None, None, None]
        blend = acc_rgb[-ov:] * (1.0 - weights) + seg_rgb[:ov] * weights
        merged_rgb = np.concatenate([acc_rgb[:-ov], blend, seg_rgb[ov:]], axis=0)
        ov_a = min(ov * int(AUDIO_SAMPLE_RATE / rate), acc_a.shape[0], seg_a.shape[0])
        if ov_a > 0:
            aw = np.linspace(0.0, 1.0, ov_a, dtype=np.float32)
            ab = acc_a[-ov_a:] * (1.0 - aw) + seg_a[:ov_a] * aw
            merged_a = np.concatenate([acc_a[:-ov_a], ab, seg_a[ov_a:]], axis=0)
        else:
            merged_a = np.concatenate([acc_a, seg_a], axis=0)

    h, w = merged_rgb.shape[1], merged_rgb.shape[2]
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tw:
        wav_path = tw.name
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tv:
        vid_only = tv.name
    try:
        pcm = np.clip(merged_a, -1.0, 1.0)
        with wave.open(wav_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(AUDIO_SAMPLE_RATE)
            wf.writeframes((pcm * 32767).astype(np.int16).tobytes())

        raw = np.clip((merged_rgb + 1.0) * 127.5, 0, 255).astype(np.uint8).tobytes()
        fps_s = f"{rate:.6f}".rstrip("0").rstrip(".")
        cmd_v = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            fps_s,
            "-i",
            "pipe:0",
            "-i",
            wav_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(out),
        ]
        proc = subprocess.run(cmd_v, input=raw, capture_output=True, check=False)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or b"").decode(errors="ignore")
            raise RuntimeError(f"LTX extend: mux append failed: {err[:1200]}")
    finally:
        Path(wav_path).unlink(missing_ok=True)
        Path(vid_only).unlink(missing_ok=True)


def trim_video_leading(
    video_path: Path,
    *,
    skip_sec: float,
    output_path: Path,
    fps: float,
) -> None:
    """Write ``video_path`` minus first ``skip_sec`` seconds to ``output_path``."""
    ffmpeg = require_ffmpeg()
    skip = max(0.0, float(skip_sec))
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{skip:.6f}",
        "-i",
        str(video_path),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        # Re-encode fallback when stream copy fails
        rate = max(1.0, float(fps))
        cmd2 = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{skip:.6f}",
            "-i",
            str(video_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-r",
            f"{rate:.6f}",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        proc2 = subprocess.run(cmd2, capture_output=True, check=False)
        if proc2.returncode != 0:
            err = (proc2.stderr or proc2.stdout or b"").decode(errors="ignore")
            raise RuntimeError(f"LTX extend: trim failed: {err[:1200]}")
