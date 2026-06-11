"""LoRA training worker — isolated subprocess (MLX must not block the API process)."""

from __future__ import annotations

import asyncio
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from backend.core.contracts import (
    ExecutionContext,
    LogEvent,
    LoraTrainingRequest,
    ProgressEvent,
)
from backend.core.interfaces import TaskStatus
from backend.persistence.asset_store import SQLiteAssetStore
from backend.persistence.v3_task_store import V3TaskStore

TRAINING_PROGRESS_FILENAME = "training_progress.json"


class DbBackedCancelToken:
    """Poll task row so API cancel works across processes."""

    def __init__(self, store: V3TaskStore, task_id: str) -> None:
        self._store = store
        self._task_id = task_id

    def cancel(self) -> None:
        self._store.mark_cancelled(self._task_id)

    def is_cancelled(self) -> bool:
        row = self._store.get_task(self._task_id)
        if not row:
            return True
        return row.get("status") == TaskStatus.CANCELLED.value

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise asyncio.CancelledError()


def _throttled_progress(
    store: V3TaskStore,
    task_id: str,
    work_dir: Path,
    *,
    min_interval_s: float = 1.0,
) -> Callable[[ProgressEvent], None]:
    last_ts = 0.0
    last_phase = ""
    progress_path = work_dir / TRAINING_PROGRESS_FILENAME

    def on_progress(ev: ProgressEvent) -> None:
        nonlocal last_ts, last_phase
        now = time.monotonic()
        phase = str(ev.phase or "")
        if (
            phase != last_phase
            or ev.step == 1
            or (ev.total and ev.step == ev.total)
            or now - last_ts >= min_interval_s
        ):
            store.update_progress(task_id, ev.progress)
            meta = {
                "step": ev.step,
                "total": ev.total,
                "eta_seconds": ev.eta_seconds,
                "message": ev.message,
                "phase": ev.phase,
            }
            progress_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            last_ts = now
            last_phase = phase

    return on_progress


def _build_exec_context(
    *,
    store: V3TaskStore,
    asset_store: SQLiteAssetStore,
    task_id: str,
    work_dir: Path,
) -> ExecutionContext:
    return ExecutionContext(
        task_id=task_id,
        cancel_token=DbBackedCancelToken(store, task_id),
        on_progress=_throttled_progress(store, task_id, work_dir),
        on_log=lambda ev: store.append_log(task_id, ev.message, ev.level),
        work_dir=work_dir,
        asset_store=asset_store,
        trace=None,
    )


def _resolve_runner(family: str):
    if family == "flux1":
        from backend.engine.training.flux_dreambooth_mlx import run_flux_dreambooth_training

        return run_flux_dreambooth_training
    if family == "z_image":
        from backend.engine.training.z_image_dreambooth_mlx import run_z_image_dreambooth_training

        return run_z_image_dreambooth_training
    if family == "qwen_image":
        from backend.engine.training.qwen_image_dreambooth_mlx import run_qwen_image_dreambooth_training

        return run_qwen_image_dreambooth_training
    raise RuntimeError(f"Unsupported LoRA training family {family!r}")


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    from backend.cli.base import build_engine_context

    task_id = str(job["task_id"])
    work_dir = Path(job["work_dir"])
    family = str(job["family"])
    request = LoraTrainingRequest.model_validate(job["request"])

    bootstrap_raw = job.get("bootstrap_root") or job.get("project_root")
    bootstrap_root = Path(bootstrap_raw) if bootstrap_raw else None
    # Install root (default_config/), not workspace — same as main.py PathResolver bootstrap.
    if bootstrap_root is not None and not (bootstrap_root / "default_config").is_dir():
        bootstrap_root = None
    ctx_bundle = build_engine_context(bootstrap_root)
    try:
        root = ctx_bundle.path_resolver.get_project_root()
        db_path = root / "db" / "studio.db"
        store = V3TaskStore(db_path)
        asset_store = SQLiteAssetStore(db_path, root / "outputs" / "assets")
        exec_ctx = _build_exec_context(
            store=store,
            asset_store=asset_store,
            task_id=task_id,
            work_dir=work_dir,
        )
        entry = ctx_bundle.model_registry.require(request.base_model.split(":", 1)[0])
        runtime = ctx_bundle.runtimes.get("mlx")
        if runtime is None:
            raise RuntimeError("LoRA training requires MLX runtime (Apple Silicon)")

        runner = _resolve_runner(family)
        return runner(
            request,
            exec_ctx,
            registry=ctx_bundle.model_registry,
            project_root=root,
            runtime=runtime,
            path_resolver=ctx_bundle.path_resolver,
        )
    finally:
        from backend.cli.base import release_engine_context

        release_engine_context(ctx_bundle)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: python -m backend.engine.training.lora_worker <job.json>", file=sys.stderr)
        return 2
    job_path = Path(args[0])
    if not job_path.is_file():
        print(f"job file not found: {job_path}", file=sys.stderr)
        return 2

    job = json.loads(job_path.read_text(encoding="utf-8"))
    result_path = Path(job["result_path"])
    try:
        result = run_job(job)
        result_path.write_text(json.dumps({"ok": True, "result": result}, ensure_ascii=False), encoding="utf-8")
        return 0
    except asyncio.CancelledError:
        result_path.write_text(
            json.dumps({"ok": False, "cancelled": True, "error": "cancelled"}),
            encoding="utf-8",
        )
        return 3
    except Exception as e:
        result_path.write_text(
            json.dumps(
                {
                    "ok": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
