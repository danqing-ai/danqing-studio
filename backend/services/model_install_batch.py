"""Shared batch model install helpers."""

from __future__ import annotations

import asyncio
from typing import Optional

from backend.core.interfaces import DownloadProgress, ISettingsService
from backend.core.model_registry import ModelRegistry
from backend.services.setup_recommendations import topological_install_order


async def run_batch_model_install(
    *,
    items: list[dict[str, str]],
    registry: ModelRegistry,
    download_service,
    settings_service: ISettingsService,
    locale: str,
) -> list[dict]:
    """Start registry model downloads; each item may include ``version_key``."""
    ordered = topological_install_order(items, registry)
    detailed = settings_service.get_models_detailed_status()
    results: list[dict] = []

    for item in ordered:
        model_id = item["model_id"]
        version = (item.get("version_key") or "").strip() or None
        if registry.get(model_id) is None:
            results.append(
                {
                    "model_id": model_id,
                    "model_name": model_id,
                    "status": "failed",
                    "error": "model not found in registry",
                }
            )
            continue
        try:
            config = download_service.get_model_download_config(model_id)

            if detailed.get(model_id, {}).get("ready"):
                results.append(
                    {
                        "model_id": model_id,
                        "model_name": model_id,
                        "status": "skipped",
                        "reason": "already_installed",
                    }
                )
                continue

            progress_queue: asyncio.Queue = asyncio.Queue()

            async def on_progress(progress: DownloadProgress, _queue=progress_queue):
                await _queue.put(progress)

            asyncio.create_task(
                download_service.download_model(
                    model_id,
                    version=version,
                    progress_callback=on_progress,
                )
            )
            first_progress = await asyncio.wait_for(progress_queue.get(), timeout=5.0)
            results.append(
                {
                    "model_id": model_id,
                    "model_name": model_id,
                    "version_key": version,
                    "status": "started",
                    "task_id": first_progress.task_id,
                }
            )
        except Exception as e:
            results.append(
                {
                    "model_id": model_id,
                    "model_name": model_id,
                    "status": "failed",
                    "error": str(e),
                }
            )

    return results


def normalize_batch_install_items(
    *,
    model_ids: Optional[list[str]] = None,
    items: Optional[list[dict]] = None,
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    if items:
        for row in items:
            if not isinstance(row, dict):
                continue
            model_id = str(row.get("model_id") or row.get("model_name") or "").strip()
            if not model_id:
                continue
            version_key = str(row.get("version_key") or row.get("version") or "").strip()
            normalized.append({"model_id": model_id, "version_key": version_key})
    elif model_ids:
        for model_id in model_ids:
            mid = str(model_id).strip()
            if mid:
                normalized.append({"model_id": mid, "version_key": ""})
    return normalized
