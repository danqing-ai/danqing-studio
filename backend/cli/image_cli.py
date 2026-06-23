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

from backend.cli.base import build_exec_context, engine_session
from backend.core.contracts import (
    ImageGenerationRequest,
    ImageEditRequest,
    ImageUpscaleRequest,
    StructuralGuide,
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
    controlnet: str = "",
    control_asset_id: str = "",
    control_image: str = "",
    controlnet_strength: float = 0.8,
    output: str = "",
    project_root: Path | None = None,
) -> str:
    """文生图。对应 POST /api/images/generations。"""
    with engine_session(project_root) as ctx:
        exec_ctx = build_exec_context(
            work_dir=ctx.path_resolver.get_outputs_dir() / "cli_tmp",
            asset_store=ctx.asset_store,
            on_progress=lambda ev: None,
            on_log=lambda ev: print(f"  [{ev.level}] {ev.message}"),
        )

        sched = (scheduler or "").strip() or None
        structural_guide: StructuralGuide | None = None
        cn_key = (controlnet or "").strip()
        resolved_control_id = (control_asset_id or "").strip()
        if control_image and not resolved_control_id:
            mask_path = Path(control_image)
            resolved_control_id = ctx.asset_store.create_from_file(
                mask_path,
                kind="image",
                mime_type="image/png",
                source_task_id="",
                metadata={"cli": "structural_guide"},
                source_action="upload",
            )
            print(f"[cli] uploaded control image {control_image} → asset {resolved_control_id}")
        if cn_key or resolved_control_id or control_image:
            if not cn_key:
                raise ValueError(
                    "structural_guide requires --controlnet when using --control-asset-id or --control-image"
                )
            if not resolved_control_id:
                raise ValueError(
                    "structural_guide requires --control-asset-id or --control-image when --controlnet is set"
                )
            from backend.engine.families.flux1.structural import infer_guide_type, is_fill_controlnet

            if is_fill_controlnet(cn_key):
                raise ValueError(
                    "flux-fill-controlnet is for retouch/extend only; use danqing-edit with --operation retouch|extend"
                )
            structural_guide = StructuralGuide(
                asset_id=resolved_control_id,
                model_id=cn_key,
                type=infer_guide_type(cn_key),
                weight=float(controlnet_strength),
            )

        request = ImageGenerationRequest(
            model=model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            size=size,
            steps=steps,
            guidance=guidance,
            seed=seed,
            scheduler=sched,
            structural_guide=structural_guide,
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
    mask_asset_id: str = "",
    mask_image: str = "",
    extend_directions: list[str] | None = None,
    extend_pixels: int = 256,
    source_fidelity: float = 0.6,
    negative_prompt: str = "",
    guidance: float | None = None,
    steps: int | None = None,
    seed: int | None = None,
    scheduler: str | None = None,
    reference_asset_ids: list[str] | None = None,
    reference_images: list[str] | None = None,
    output: str = "",
    project_root: Path | None = None,
) -> str:
    """图像编辑。对应 POST /api/images/edits。

    source_asset_id 或 source_image 二选一。
    retouch 需 mask_asset_id 或 mask_image（白=重绘区域）。
    extend 需 extend_directions（top/bottom/left/right 组合）。
    """
    with engine_session(project_root) as ctx:
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

        resolved_mask_id = (mask_asset_id or "").strip()
        if mask_image and not resolved_mask_id:
            mask_path = Path(mask_image)
            resolved_mask_id = ctx.asset_store.create_from_file(
                mask_path, kind="image", mime_type="image/png", source_task_id="",
            )
            print(f"[cli] uploaded mask {mask_image} → asset {resolved_mask_id}")

        extend_spec = None
        if operation == "retouch":
            if not resolved_mask_id:
                raise ValueError(
                    "retouch requires --mask-image or --mask-asset-id (white = repaint region)"
                )
        elif operation == "extend":
            dirs = [d for d in (extend_directions or []) if d in ("top", "bottom", "left", "right")]
            if not dirs:
                raise ValueError(
                    "extend requires --extend-directions (comma-separated: top,bottom,left,right)"
                )
            from backend.core.contracts import ExtendSpec

            extend_spec = ExtendSpec(
                directions=dirs,
                pixels=max(64, min(2048, int(extend_pixels))),
            )

        sched = (scheduler or "").strip() or None
        ref_ids = [str(x).strip() for x in (reference_asset_ids or []) if str(x).strip()]
        for ref_image in reference_images or []:
            ref_path = Path(ref_image)
            ref_aid = ctx.asset_store.create_from_file(
                ref_path, kind="image", mime_type="image/png", source_task_id="",
            )
            ref_ids.append(ref_aid)
            print(f"[cli] uploaded reference {ref_image} → asset {ref_aid}")

        request = ImageEditRequest(
            model=model,
            operation=operation,
            source_asset_id=source_asset_id,
            reference_asset_ids=ref_ids,
            mask_asset_id=resolved_mask_id or None,
            extend=extend_spec,
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
    with engine_session(project_root) as ctx:
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
