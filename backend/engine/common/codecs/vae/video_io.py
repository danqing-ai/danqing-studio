"""PIL frame list → MP4 (ffmpeg with imageio fallback)."""
from __future__ import annotations

import subprocess
from typing import Any


def save_pil_frames_to_mp4(frames: list[Any], output_path: str, *, fps: int = 16) -> None:
    """Write RGB PIL frames to H.264 MP4. Prefer ffmpeg; fall back to imageio."""
    if not frames:
        raise RuntimeError("No frames to save")

    try:
        _save_via_ffmpeg(frames, output_path, fps)
        return
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    _save_via_imageio(frames, output_path, fps)


def _save_via_ffmpeg(frames: list[Any], output_path: str, fps: int) -> None:
    import numpy as np

    w, h = frames[0].size
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdin is not None
    for frame in frames:
        arr = np.array(frame.convert("RGB"))
        proc.stdin.write(arr.tobytes())
    proc.stdin.close()
    proc.wait(timeout=120)
    if proc.returncode != 0:
        raise subprocess.SubprocessError(proc.returncode)


def _save_via_imageio(frames: list[Any], output_path: str, fps: int) -> None:
    import imageio
    import numpy as np

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
    for frame in frames:
        writer.append_data(np.array(frame.convert("RGB")))
    writer.close()
