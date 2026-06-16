"""Z-Image DiT merge CLI — mirrors POST /api/tools/z-image/merge."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from backend.cli.base import build_exec_context, engine_session
from backend.core.contracts import ZImageMergeRequest


def merge_z_image(
    *,
    model_a: str,
    model_b: str,
    model_c: str | None = None,
    method: str = "weighted_sum",
    alpha: float = 0.5,
    output_name: str,
    project_root: Path | None = None,
) -> dict:
    with engine_session(project_root) as ctx:
        from backend.engine.danqing_tools_engine import DanQingToolsEngine

        tools = DanQingToolsEngine(
            ctx.path_resolver,
            ctx.model_registry,
            ctx.runtimes,
        )
        req = ZImageMergeRequest(
            model_a=model_a,
            model_b=model_b,
            model_c=model_c,
            method=method,  # type: ignore[arg-type]
            alpha=alpha,
            output_name=output_name,
        )
        work = ctx.path_resolver.get_outputs_dir() / "cli_merge" / f"run_{int(time.time())}"
        work.mkdir(parents=True, exist_ok=True)
        exec_ctx = build_exec_context(
            work_dir=work,
            asset_store=ctx.asset_store,
            on_log=lambda ev: print(f"  [{ev.level}] {ev.message}"),
        )

        async def _run():
            return await tools.merge_z_image(req, exec_ctx)

        result = asyncio.run(_run())
        meta = dict(result.metadata or {})
        print(f"Merge done: {meta.get('z_image_merge', meta)}")
        return meta
