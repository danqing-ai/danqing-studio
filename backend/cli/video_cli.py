"""
视频 CLI 命令 — 调用 DanQingVideoEngine。

与 REST API 端点一一对应：
  danqing-video-generate → POST /api/videos/generations → IVideoEngine.generate()
  danqing-video-edit     → POST /api/videos/edits       → IVideoEngine.edit()
"""
from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path

from backend.cli.base import build_engine_context, build_exec_context
from backend.core.contracts import VideoGenerationRequest, VideoEditRequest


def generate(
    model: str,
    prompt: str,
    *,
    negative_prompt: str = "",
    size: str = "1024x1024",
    steps: int | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    output: str = "",
    project_root: Path | None = None,
) -> str:
    """文生视频。对应 POST /api/videos/generations。"""
    ctx = build_engine_context(project_root)
    exec_ctx = build_exec_context(
        work_dir=ctx.path_resolver.get_outputs_dir() / "cli_tmp",
        asset_store=ctx.asset_store,
        on_progress=lambda ev: None,
        on_log=lambda ev: print(f"  [{ev.level}] {ev.message}"),
    )

    request = VideoGenerationRequest(
        model=model,
        prompt=prompt,
        negative_prompt=negative_prompt,
        size=size,
        steps=steps,
        guidance=guidance,
        seed=seed,
    )

    t0 = time.time()
    result = asyncio.run(ctx.video_engine.generate(request, exec_ctx))
    elapsed = time.time() - t0

    if result.metadata.get("status") == "cancelled":
        raise RuntimeError("Generation cancelled")

    if output and result.output_paths:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result.output_paths[0], out)
        print(f"[cli] DONE ({elapsed:.1f}s) -> {out}")
        return str(out)

    if result.output_paths:
        print(f"[cli] DONE ({elapsed:.1f}s) -> {result.output_paths[0]}")
        return result.output_paths[0]

    raise RuntimeError("No output generated")


def edit(
    model: str,
    source_asset_id: str,
    prompt: str,
    *,
    negative_prompt: str = "",
    steps: int | None = None,
    seed: int | None = None,
    output: str = "",
    project_root: Path | None = None,
) -> str:
    """视频编辑。对应 POST /api/videos/edits。"""
    ctx = build_engine_context(project_root)
    exec_ctx = build_exec_context(
        work_dir=ctx.path_resolver.get_outputs_dir() / "cli_tmp",
        asset_store=ctx.asset_store,
        on_progress=lambda ev: None,
        on_log=lambda ev: print(f"  [{ev.level}] {ev.message}"),
    )

    request = VideoEditRequest(
        model=model,
        source_asset_id=source_asset_id,
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        seed=seed,
    )

    t0 = time.time()
    result = asyncio.run(ctx.video_engine.edit(request, exec_ctx))
    elapsed = time.time() - t0

    if result.metadata.get("status") == "cancelled":
        raise RuntimeError("Edit cancelled")

    if output and result.output_paths:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result.output_paths[0], out)
        print(f"[cli] DONE ({elapsed:.1f}s) -> {out}")
        return str(out)

    if result.output_paths:
        print(f"[cli] DONE ({elapsed:.1f}s) -> {result.output_paths[0]}")
        return result.output_paths[0]

    raise RuntimeError("No output generated")
