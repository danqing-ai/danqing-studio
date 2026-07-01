"""Bernini-R 1.3B MLX video generation — mlx-video Wan2.1 substrate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from backend.engine.config.model_configs import BerniniConfig


@dataclass
class BerniniMlxGenerator:
    ctx: Any
    bundle_root: Path
    config: BerniniConfig
    entry: Any | None = None
    version_key: str | None = None

    def load(self) -> None:
        return

    def generate_and_save(
        self,
        *,
        prompt: str,
        output_path: str,
        width: int,
        height: int,
        num_frames: int,
        fps: float,
        seed: int,
        steps: int,
        guidance: float,
        step_distill: bool,
        image_path: str | None,
        on_log: Callable[[str, str], None] | None,
    ) -> str:
        _ = fps, step_distill, image_path
        if image_path:
            raise RuntimeError(
                "Bernini-R I2V/r2v is not wired in DanQing yet; use text-to-video (create) only."
            )

        try:
            from mlx_video.models.wan_2.generate import generate_video
        except ImportError as exc:
            raise RuntimeError(
                "Bernini-R requires the mlx-video package. "
                "Install dependencies: pip install mlx-video>=0.1.0"
            ) from exc

        if on_log:
            on_log(
                "info",
                " ".join(
                    [
                        "bernini_generate backend=mlx_video",
                        f"bundle={self.bundle_root}",
                        f"shift={self.config.sample_shift}",
                        f"frames={num_frames}",
                        f"steps={steps}",
                        f"guide={guidance}",
                    ]
                ),
            )

        guide_scale = guidance if guidance > 0 else self.config.default_guidance
        out = generate_video(
            model_dir=str(self.bundle_root),
            prompt=prompt,
            negative_prompt=self.config.default_negative_prompt or None,
            width=int(width),
            height=int(height),
            num_frames=int(num_frames),
            steps=int(steps) if steps > 0 else None,
            guide_scale=float(guide_scale),
            shift=float(self.config.sample_shift),
            seed=int(seed),
            output_path=str(output_path),
            scheduler="unipc" if self.config.use_unipc else "euler",
        )
        return str(out)
