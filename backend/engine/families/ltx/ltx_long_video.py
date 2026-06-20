"""LTX 2.3 multi-extend long video orchestrator (Pass0 T2V + extend loop)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from backend.core.contracts import VideoGenerationRequest, VideoLongVideoSpec
from backend.engine.families.ltx.extend_mlx import extend_and_append, validate_extend_window_frames
from backend.engine.families.ltx.generation_mlx import LTX23MlxGenerator
from backend.engine.families.ltx.long_video_plan import (
    build_long_video_plan,
    duration_sec_from_num_frames,
    num_frames_for_duration_sec,
)
from backend.engine.pipelines.pipeline_progress import emit_complete, emit_phase


def run_ltx_long_video(
    generator: LTX23MlxGenerator,
    *,
    request: VideoGenerationRequest,
    spec: VideoLongVideoSpec,
    output_path: str,
    width: int,
    height: int,
    fps: float,
    seed: int,
    steps: int,
    guidance: float,
    step_distill: bool,
    max_frames: int = 257,
    on_log: Callable[[str, str], None] | None = None,
    on_progress: Callable[..., None] | None = None,
) -> str:
    """Pass0 T2V then latent extend passes until ``target_duration_sec``."""
    plan = build_long_video_plan(
        target_duration_sec=spec.target_duration_sec,
        initial_duration_sec=spec.initial_duration_sec,
        segment_extend_sec=spec.segment_extend_sec,
        reference_duration_sec=spec.reference_duration_sec,
    )
    validate_extend_window_frames(
        reference_sec=spec.reference_duration_sec,
        extend_sec=spec.segment_extend_sec,
        fps=fps,
        max_frames=max_frames,
    )

    pass0_frames = num_frames_for_duration_sec(spec.initial_duration_sec, fps)
    pass0_prompt = (spec.opening_prompt or request.prompt or "").strip()
    if not pass0_prompt:
        raise RuntimeError("LTX long video requires a non-empty opening prompt")

    stage2_steps = int(getattr(generator.config, "ltx_stage2_steps", 3) or 3)
    progress_total = max(1, int(steps) + stage2_steps) * (1 + plan.extend_pass_count)

    if on_log:
        on_log(
            "info",
            f"long_video start target={plan.target_duration_sec:.1f}s "
            f"passes={1 + plan.extend_pass_count} fps={fps}",
        )

    emit_phase(on_progress, phase="generate", progress=0.02, n_steps=progress_total)
    out = Path(output_path)
    work = out.parent
    work.mkdir(parents=True, exist_ok=True)

    generator.generate_and_save(
        prompt=pass0_prompt,
        output_path=str(out),
        width=width,
        height=height,
        num_frames=pass0_frames,
        fps=float(fps),
        seed=seed,
        steps=steps,
        guidance=guidance,
        step_distill=step_distill,
        image_path=None,
        on_log=on_log,
        on_progress=on_progress,
    )

    current_sec = duration_sec_from_num_frames(pass0_frames, fps)
    segment_prompts = list(spec.segment_prompts or [])
    fallback_prompt = (request.prompt or "").strip()

    for pass_idx in range(plan.extend_pass_count):
        if current_sec >= plan.target_duration_sec - 0.5:
            break
        seg_prompt = (
            segment_prompts[pass_idx].strip()
            if pass_idx < len(segment_prompts) and segment_prompts[pass_idx].strip()
            else fallback_prompt
        )
        if not seg_prompt:
            raise RuntimeError(
                f"long_video pass {pass_idx + 1}: empty segment prompt "
                "(provide segment_prompts or main prompt)"
            )
        pass_num = pass_idx + 2
        total_passes = 1 + plan.extend_pass_count
        if on_log:
            on_log(
                "info",
                f"long_video pass {pass_num}/{total_passes} extending +{spec.segment_extend_sec:.1f}s "
                f"(total {current_sec:.1f}s/{plan.target_duration_sec:.1f}s)",
            )
        extend_and_append(
            generator,
            accumulator_mp4=out,
            work_dir=work,
            prompt=seg_prompt,
            width=width,
            height=height,
            reference_sec=spec.reference_duration_sec,
            extend_sec=spec.segment_extend_sec,
            fps=fps,
            seed=seed + 1000 + pass_idx,
            steps=steps,
            guidance=guidance,
            step_distill=step_distill,
            overlap_blend_frames=int(spec.overlap_blend_frames),
            max_frames=max_frames,
            on_log=on_log,
            on_progress=on_progress,
        )
        current_sec += float(spec.segment_extend_sec)
        if int(spec.overlap_blend_frames) > 0:
            current_sec -= int(spec.overlap_blend_frames) / max(1.0, fps)

    emit_complete(on_progress, progress_total)
    if on_log:
        on_log("info", f"long_video complete → {out} (~{current_sec:.1f}s)")
    return str(out)
