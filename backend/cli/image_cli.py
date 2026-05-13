"""
图像 CLI 命令 — 调用 DanQingImageEngine。

与 REST API 端点一一对应：
  danqing-generate → POST /api/images/generations → IImageEngine.generate()
  danqing-edit     → POST /api/images/edits       → IImageEngine.edit()
  danqing-upscale  → POST /api/images/upscales    → IImageEngine.upscale()
"""
from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path

from backend.cli.base import build_engine_context, build_exec_context
from backend.core.contracts import (
    ImageGenerationRequest, ImageEditRequest, ImageUpscaleRequest,
)


def generate(
    model: str,
    prompt: str,
    *,
    negative_prompt: str = "",
    size: str = "1024x1024",
    steps: int | None = None,
    guidance: float | None = None,
    seed: int | None = None,
    scheduler: str | None = None,
    output: str = "",
    project_root: Path | None = None,
) -> str:
    """文生图。对应 POST /api/images/generations。"""
    ctx = build_engine_context(project_root)
    exec_ctx = build_exec_context(
        work_dir=ctx.path_resolver.get_outputs_dir() / "cli_tmp",
        asset_store=ctx.asset_store,
        on_progress=lambda ev: None,
        on_log=lambda ev: print(f"  [{ev.level}] {ev.message}"),
    )

    sched = (scheduler or "").strip() or None
    request = ImageGenerationRequest(
        model=model,
        prompt=prompt,
        negative_prompt=negative_prompt,
        size=size,
        steps=steps,
        guidance=guidance,
        seed=seed,
        scheduler=sched,
    )

    if not ctx.image_engine.supports(model, "generate"):
        raise RuntimeError(
            f"Model {model!r} does not support text-to-image (create); "
            "check config/models_registry.json actions."
        )

    t0 = time.time()
    result = asyncio.run(ctx.image_engine.generate(request, exec_ctx))
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
    operation: str,
    prompt: str,
    *,
    source_asset_id: str = "",
    source_image: str = "",
    source_fidelity: float = 0.6,
    negative_prompt: str = "",
    guidance: float | None = None,
    steps: int | None = None,
    seed: int | None = None,
    scheduler: str | None = None,
    output: str = "",
    project_root: Path | None = None,
) -> str:
    """图像编辑。对应 POST /api/images/edits。
    
    source_asset_id 或 source_image 二选一。
    source_image 为本地文件路径，内部自动上传为 asset。
    """
    ctx = build_engine_context(project_root)
    exec_ctx = build_exec_context(
        work_dir=ctx.path_resolver.get_outputs_dir() / "cli_tmp",
        asset_store=ctx.asset_store,
        on_progress=lambda ev: None,
        on_log=lambda ev: print(f"  [{ev.level}] {ev.message}"),
    )

    # 本地文件 → 上传为 asset
    if source_image and not source_asset_id:
        from pathlib import Path
        aid = ctx.asset_store.create_from_file(
            Path(source_image), kind="image", mime_type="image/png",
            source_task_id="",
        )
        source_asset_id = aid
        print(f"[cli] uploaded {source_image} → asset {aid}")

    if not source_asset_id:
        raise ValueError("source_asset_id or source_image is required")

    sched = (scheduler or "").strip() or None
    request = ImageEditRequest(
        model=model,
        operation=operation,
        source_asset_id=source_asset_id,
        prompt=prompt,
        source_fidelity=source_fidelity,
        negative_prompt=negative_prompt,
        guidance=guidance,
        steps=steps,
        seed=seed,
        scheduler=sched,
    )

    if not ctx.image_engine.supports(model, "edit"):
        raise RuntimeError(
            f"Model {model!r} does not support image edit; "
            "check config/models_registry.json actions."
        )

    t0 = time.time()
    result = asyncio.run(ctx.image_engine.edit(request, exec_ctx))
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


def upscale(
    model: str,
    source_asset_id: str = "",
    *,
    source_image: str = "",
    scale_factor: float = 2.0,
    seed: int | None = None,
    denoise: float | None = None,
    output: str = "",
    project_root: Path | None = None,
) -> str:
    """图像放大。对应 POST /api/images/upscales。

    ``source_asset_id`` 或 ``source_image``（本地路径，会先登记为 asset）二选一。
    """
    ctx = build_engine_context(project_root)
    exec_ctx = build_exec_context(
        work_dir=ctx.path_resolver.get_outputs_dir() / "cli_tmp",
        asset_store=ctx.asset_store,
        on_progress=lambda ev: None,
        on_log=lambda ev: print(f"  [{ev.level}] {ev.message}"),
    )

    if source_image and not source_asset_id:
        aid = ctx.asset_store.create_from_file(
            Path(source_image), kind="image", mime_type="image/png",
            source_task_id="",
        )
        source_asset_id = aid
        print(f"[cli] uploaded {source_image} → asset {aid}")

    if not source_asset_id:
        raise ValueError("source_asset_id or source_image is required")

    sf = int(scale_factor)
    if sf not in (2, 4):
        raise ValueError("scale_factor must be 2 or 4 (matches ImageUpscaleRequest.scale)")
    meta: dict = {}
    if seed is not None:
        meta["seed"] = int(seed)
    req_kw: dict = {
        "model": model,
        "source_asset_id": source_asset_id,
        "scale": sf,
        "metadata": meta,
    }
    if denoise is not None:
        req_kw["denoise"] = float(denoise)
    request = ImageUpscaleRequest(**req_kw)

    if not ctx.image_engine.supports(model, "upscale"):
        raise RuntimeError(
            f"Model {model!r} does not support upscale; "
            "check config/models_registry.json actions."
        )

    t0 = time.time()
    result = asyncio.run(ctx.image_engine.upscale(request, exec_ctx))
    elapsed = time.time() - t0

    if result.metadata.get("status") == "cancelled":
        raise RuntimeError("Upscale cancelled")

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
