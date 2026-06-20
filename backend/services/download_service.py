"""
Download service implementation - supports HuggingFace and HTTP dual-source downloads.
"""

import os
# Configure HF mirror site
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import logging
import time
import uuid
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import asyncio
import shutil

from backend.core.bundle_repos import (
    bundle_local_paths,
    bundle_repos_from_version,
    primary_and_follow_ups,
    version_primary_local_path,
)
from backend.core.install_hooks import install_hooks_from_version, run_install_hooks
from backend.core.bundle_manifest import skips_full_family_bundle_contract, write_bundle_manifest
from backend.core.interfaces import (
    IDownloadService, IPathResolver, IConfigStore,
    DownloadTask, DownloadProgress, TaskStatus, ConversionTask
)
from backend.core.i18n import t as tt, get_locale
from backend.core.registry_format import resolve_registry_label
from backend.core.downloaders import HTTPDownloader
from backend.services.hf_repo_resolve import resolve_huggingface_repo_id

logger = logging.getLogger(__name__)


class DownloadService(IDownloadService):
    """Download service implementation.

    Automatically selects the download method based on the source field in the model registry:
    - huggingface: uses huggingface_hub.snapshot_download, polls directory size for progress
    - modelscope: uses modelscope.snapshot_download, polls directory size for progress
    - civitai / http: uses HTTPDownloader (aiohttp)
    """

    def __init__(self, path_resolver: IPathResolver, config_store: Optional[IConfigStore] = None):
        self._path_resolver = path_resolver
        self._config = config_store
        self._downloads: Dict[str, DownloadTask] = {}
        self._progress: Dict[str, DownloadProgress] = {}
        self._conversions: Dict[str, ConversionTask] = {}
        self._conversion_events: Dict[str, asyncio.Event] = {}
        self._persist_path = path_resolver.get_workspace_config_dir() / ".download_tasks.json"

        # Read tokens from config
        hf_token = None
        civitai_token = None
        if self._config:
            settings = self._config.load()
            hf_token = settings.huggingface_token or None
            civitai_token = settings.civitai_token or None

        # Initialize downloaders
        self._http_downloader = HTTPDownloader(
            civitai_token=civitai_token,
            huggingface_token=hf_token,
        )
        self._cancelled_downloads: set = set()
        self._token = hf_token

        # Active model download dedup: (model_name, version) -> task_id
        self._active_model_downloads: Dict[tuple, str] = {}
        # Concurrency lock protecting dedup checks and task creation
        self._download_lock = asyncio.Lock()

        # Load persisted download tasks
        self._load_persisted_downloads()

    def _persist_downloads(self) -> None:
        """Persist download tasks to JSON file (retains all states, including completed and failed)."""
        try:
            data = {}
            for task_id, task in self._downloads.items():
                progress = self._progress.get(task_id)
                data[task_id] = {
                    "id": task.id,
                    "url": task.url,
                    "target_path": task.target_path,
                    "status": task.status.value,
                    "progress": task.progress,
                    "total_size": progress.total_size if progress else 0,
                    "downloaded_size": progress.downloaded_size if progress else 0,
                    "speed": progress.speed if progress else "",
                    "filename": progress.filename if progress else task.url,
                    "error_message": progress.error_message if progress else "",
                    "model_name": getattr(task, '_model_name', None),
                    "version": getattr(task, '_version', None),
                    "is_lora": getattr(task, '_is_lora', False),
                    "lora_filename": getattr(task, '_filename', None),
                }
            with open(self._persist_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[DownloadService] Failed to persist download tasks: {e}")

    def _load_persisted_downloads(self) -> None:
        """Load persisted download tasks from JSON file.

        After process restart, tasks in running state are marked as paused,
        waiting for the user to manually resume.
        """
        if not self._persist_path.exists():
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for task_id, item in data.items():
                task = DownloadTask(
                    id=item["id"],
                    url=item["url"],
                    target_path=item["target_path"],
                )
                raw_status = item.get("status", "running")
                # After process restart, tasks that were running become paused
                if raw_status == "running":
                    task.status = TaskStatus.PAUSED
                else:
                    task.status = TaskStatus(raw_status)
                task.progress = item.get("progress", 0)
                # Restore metadata
                if item.get("model_name"):
                    task._model_name = item["model_name"]
                if item.get("version"):
                    task._version = item["version"]
                if item.get("is_lora"):
                    task._is_lora = item["is_lora"]
                if item.get("lora_filename"):
                    task._filename = item["lora_filename"]
                self._downloads[task_id] = task
                self._progress[task_id] = DownloadProgress(
                    task_id=task_id,
                    status=task.status.value,
                    progress=item.get("progress", 0),
                    total_size=item.get("total_size", 0),
                    downloaded_size=item.get("downloaded_size", 0),
                    speed=item.get("speed", ""),
                    filename=item.get("filename", task.url),
                    error_message=item.get("error_message", "")
                )
        except Exception as e:
            print(f"[DownloadService] Failed to load persisted download tasks: {e}")



    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry."""
        registry_path = self._path_resolver.get_models_registry_path()
        if not registry_path.exists():
            return {}
        try:
            from backend.catalog.loader import expand_catalog_document

            with open(registry_path, "r", encoding="utf-8") as f:
                return expand_catalog_document(json.load(f)).get("models", {})
        except Exception as e:
            print(f"[DownloadService] Failed to load model registry: {e}")
            return {}

    def get_model_download_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get download configuration info for a model."""
        registry = self._load_registry()
        return registry.get(model_name)

    @staticmethod
    def _check_hf_connectivity(timeout: float = 10.0) -> bool:
        """Check if HuggingFace mirror is accessible."""
        import urllib.request
        try:
            endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
            req = urllib.request.Request(
                endpoint,
                method="HEAD",
                headers={"User-Agent": "huggingface_hub"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    @staticmethod
    def _check_modelscope_connectivity(timeout: float = 10.0) -> bool:
        """Check if ModelScope is accessible."""
        import urllib.request
        try:
            endpoint = "https://www.modelscope.cn"
            req = urllib.request.Request(
                endpoint,
                method="HEAD",
                headers={"User-Agent": "modelscope"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    @staticmethod
    def _parse_size_to_bytes(size_str: str) -> int:
        """Parse size string from registry (e.g. '23.8GB') into bytes."""
        if not size_str:
            return 0
        size_str = size_str.strip().upper()
        units = {
            'GB': 1024 ** 3,
            'MB': 1024 ** 2,
            'KB': 1024,
            'B': 1,
        }
        for unit, multiplier in units.items():
            if size_str.endswith(unit):
                try:
                    value = float(size_str[:-len(unit)])
                    return int(value * multiplier)
                except ValueError:
                    return 0
        return 0

    def _bundle_estimated_bytes(self, ver_config: dict[str, Any] | None) -> int:
        entries = bundle_repos_from_version(ver_config)
        if not entries:
            return 0
        total = 0
        for spec in entries:
            sz = spec.get("size")
            if sz:
                total += self._parse_size_to_bytes(str(sz))
        return total

    def _resolve_registry_path(self, local_path: str) -> Path:
        return self._path_resolver.resolve_registry_local_path(local_path)

    def _maybe_assemble_hunyuan_ms_bundle(self, target: Path, ver_config: dict[str, Any] | None) -> None:
        if not ver_config:
            return
        variant = ver_config.get("hunyuan_ms_variant")
        if not variant:
            return
        from backend.services.hunyuan_ms_bundle import assemble_hunyuan_modelscope_bundle

        assemble_hunyuan_modelscope_bundle(target, str(variant))

    def _maybe_assemble_wan_distill_bundle(self, target: Path, ver_config: dict[str, Any] | None) -> None:
        if not ver_config:
            return
        variant = ver_config.get("wan_distill_variant")
        if not variant:
            return
        from backend.services.wan_distill_bundle import assemble_wan_distill_bundle

        assemble_wan_distill_bundle(target, str(variant))

    async def _finalize_version_install(
        self,
        *,
        model_name: str,
        version: str | None,
        ver_config: dict[str, Any] | None,
        target: Path,
        task_id: str,
        display_name: str,
        on_progress: Callable[[DownloadProgress], Any],
    ) -> None:
        """Post-download steps: Hunyuan MS assembly + install_hooks + bundle manifest."""
        if ver_config and ver_config.get("hunyuan_ms_variant"):
            self._maybe_assemble_hunyuan_ms_bundle(target, ver_config)
        if ver_config and ver_config.get("wan_distill_variant"):
            self._maybe_assemble_wan_distill_bundle(target, ver_config)

        hooks = install_hooks_from_version(ver_config)
        if hooks:
            logger.info(
                "Running %d install hook(s) for %s:%s at %s",
                len(hooks),
                model_name,
                version or "default",
                target,
            )
            await on_progress(
                DownloadProgress(
                    task_id=task_id,
                    status="running",
                    progress=0.99,
                    filename=display_name,
                    speed="",
                )
            )
            loop = asyncio.get_event_loop()

            def _run_hooks() -> None:
                run_install_hooks(
                    model_name=model_name,
                    version_key=version,
                    ver_config=ver_config,
                    bundle_root=target,
                )

            await loop.run_in_executor(None, _run_hooks)

        config = self.get_model_download_config(model_name) or {}
        family = str(config.get("family") or "").strip()
        category = str(config.get("category") or "").strip()
        if family and target.is_dir() and not skips_full_family_bundle_contract(category):
            try:
                manifest_path = write_bundle_manifest(
                    target,
                    model_id=model_name,
                    family=family,
                )
                logger.info("Wrote bundle manifest: %s", manifest_path)
            except Exception as exc:
                logger.error(
                    "Failed to write bundle.manifest.json for %s at %s: %s",
                    model_name,
                    target,
                    exc,
                )
                raise RuntimeError(
                    f"Post-download manifest generation failed for {model_name!r}: {exc}"
                ) from exc

    def _sync_download_bundle_repo(
        self,
        spec: dict[str, Any],
        *,
        default_source: str,
    ) -> str:
        """Download one ``bundle_repos`` follow-up entry."""
        repo = str(spec["repo_id"]).strip()
        dest = self._resolve_registry_path(str(spec["local_path"]).strip())
        dest.mkdir(parents=True, exist_ok=True)
        source = str(spec.get("source") or default_source).strip().lower()
        patterns = spec.get("allow_patterns")

        if source == "huggingface":
            from huggingface_hub import snapshot_download as hf_snapshot

            if not self._check_hf_connectivity():
                raise ConnectionError(
                    "Cannot connect to model download server (HF_ENDPOINT=%s), please check network connection"
                    % os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
                )
            return hf_snapshot(
                repo_id=resolve_huggingface_repo_id(repo),
                local_dir=str(dest),
                allow_patterns=patterns if patterns else None,
                token=self._token,
            )

        if source == "modelscope":
            from modelscope import snapshot_download as ms_snapshot

            if not self._check_modelscope_connectivity():
                raise ConnectionError(
                    "Cannot connect to ModelScope, please check network connection"
                )
            return ms_snapshot(
                model_id=repo,
                local_dir=str(dest),
                allow_patterns=patterns if patterns else None,
            )

        raise RuntimeError(f"Unsupported bundle repo source {source!r} for {repo!r}")

    async def download_model(self, model_name: str, version: Optional[str] = None,
                            progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
                            existing_task_id: Optional[str] = None) -> str:
        """Download model by registry model name (supports base models and LoRA)."""
        config = self.get_model_download_config(model_name)
        if not config:
            loc = get_locale()
            raise ValueError(tt("error.model_not_in_registry", loc, name=model_name))
        if config.get("stub_no_download"):
            loc = get_locale()
            raise ValueError(tt("error.audio_stub_no_download", loc, name=model_name))

        source = config.get("source", "huggingface")
        repo_id = config.get("repo_id")
        download_url = config.get("download_url")
        local_path = config.get("local_path", f"models/{model_name}")

        # Build friendly display name: model name + version name
        loc = get_locale()
        model_display_name = resolve_registry_label(
            config.get("name"), model_name, locale=loc
        )

        version_display_name = ""
        ver_config = None
        if version and config.get("versions"):
            ver_config = config["versions"].get(version)
            if ver_config:
                version_display_name = resolve_registry_label(
                    ver_config.get("name"), version, locale=loc
                )
                if not repo_id:
                    repo_id = ver_config.get("repo_id")
                if not download_url:
                    download_url = ver_config.get("download_url")
                if ver_config.get("local_path"):
                    local_path = ver_config["local_path"]
                vs = ver_config.get("source")
                if isinstance(vs, str) and vs.strip():
                    source = vs.strip().lower()
                primary_spec, _follow = primary_and_follow_ups(ver_config)
                if primary_spec:
                    repo_id = primary_spec["repo_id"]
                    local_path = primary_spec["local_path"]
                    if primary_spec.get("source"):
                        source = str(primary_spec["source"]).strip().lower()

        if source == "huggingface" and repo_id:
            repo_id = resolve_huggingface_repo_id(repo_id)

        # Parse estimated size for this version from registry (fallback if HF API fails)
        estimated_size = 0
        bundle_bytes = self._bundle_estimated_bytes(ver_config)
        if bundle_bytes > 0:
            estimated_size = bundle_bytes
        elif ver_config and ver_config.get("size"):
            estimated_size = self._parse_size_to_bytes(ver_config["size"])
        elif config.get("size"):
            estimated_size = self._parse_size_to_bytes(config["size"])

        # Unified display name
        display_name = model_display_name
        if version_display_name:
            display_name = f"{model_display_name} {version_display_name}"

        target = self._path_resolver.resolve_registry_local_path(local_path)
        target.mkdir(parents=True, exist_ok=True)

        # Dedup check: only one running/queued download per model per version
        dedup_key = (model_name, version)
        existing_active_id = self._active_model_downloads.get(dedup_key)
        if existing_active_id and existing_active_id in self._downloads:
            existing_task = self._downloads[existing_active_id]
            if existing_task.status == TaskStatus.RUNNING:
                # Running task already exists, return existing task_id
                return existing_active_id
            elif existing_task.status == TaskStatus.PAUSED:
                # Paused task exists; if resuming, continue; otherwise return existing task_id
                if not existing_task_id or existing_task_id != existing_active_id:
                    return existing_active_id

        if existing_task_id and existing_task_id in self._downloads:
            task_id = existing_task_id
            task = self._downloads[task_id]
            task.status = TaskStatus.RUNNING
            task.target_path = str(target)
        else:
            task_id = str(uuid.uuid4())
            task = DownloadTask(
                id=task_id,
                url=repo_id or download_url or "",
                target_path=str(target)
            )
        task._model_name = model_name
        task._version = version
        self._downloads[task_id] = task
        self._active_model_downloads[dedup_key] = task_id
        task.status = TaskStatus.RUNNING

        # Internal progress callback (store progress + external callback + persist)
        async def on_progress(progress: DownloadProgress):
            self._progress[task_id] = progress
            self._persist_downloads()
            if progress_callback:
                await self._async_callback(progress_callback, progress)

        # When resuming download, preserve existing progress; don't reset to 0
        existing_progress = self._progress.get(task_id)
        if progress_callback:
            if existing_progress and existing_progress.downloaded_size > 0:
                await self._async_callback(progress_callback, existing_progress)
            else:
                display_name = model_display_name
                if version_display_name:
                    display_name = f"{model_display_name} {version_display_name}"
                await self._async_callback(progress_callback, DownloadProgress(
                    task_id=task_id,
                    status="running",
                    progress=0,
                    total_size=0,
                    downloaded_size=0,
                    speed="",
                    error_message="",
                    filename=display_name
                ))

        try:
            if source == "huggingface" and repo_id:
                # Inline HuggingFace download: snapshot_download + poll directory size for progress
                from huggingface_hub import snapshot_download

                def calc_downloaded_for(p: Path) -> int:
                    if not p.exists():
                        return 0
                    total = 0
                    for dirpath, _dirnames, filenames in os.walk(p):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            try:
                                total += os.path.getsize(fp)
                            except OSError:
                                pass
                    return total

                def format_speed(bytes_per_sec):
                    if bytes_per_sec > 1024 * 1024 * 1024:
                        return f"{bytes_per_sec / (1024 * 1024 * 1024):.1f} GB/s"
                    elif bytes_per_sec > 1024 * 1024:
                        return f"{bytes_per_sec / (1024 * 1024):.1f} MB/s"
                    elif bytes_per_sec > 1024:
                        return f"{bytes_per_sec / 1024:.1f} KB/s"
                    else:
                        return f"{bytes_per_sec:.1f} B/s"

                total_size = estimated_size if estimated_size > 0 else (existing_progress.total_size if existing_progress else 0)

                loop = asyncio.get_event_loop()

                def do_download():
                    # Check network connectivity first; avoid snapshot_download silently falling back to existing dir
                    if not self._check_hf_connectivity():
                        raise ConnectionError(
                            "Cannot connect to model download server (HF_ENDPOINT=%s), please check network connection" % os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
                        )
                    # Let HF client use default tqdm, retain its log output
                    return snapshot_download(
                        repo_id=repo_id,
                        local_dir=str(target),
                        token=self._token,
                    )

                download_future = loop.run_in_executor(None, do_download)

                # Poll directory size to calculate progress (HF)
                last_downloaded = calc_downloaded_for(target)
                last_report_time = time.time()

                try:
                    while not download_future.done():
                        await asyncio.sleep(2)  # Check every 2 seconds
                        if task_id in self._cancelled_downloads:
                            break

                        current_downloaded = calc_downloaded_for(target)
                        now = time.time()

                        # Calculate speed
                        dt = now - last_report_time
                        speed_str = ""
                        if dt >= 1:
                            speed_bytes = (current_downloaded - last_downloaded) / dt
                            speed_str = format_speed(speed_bytes)
                            last_downloaded = current_downloaded
                            last_report_time = now

                        progress = current_downloaded / total_size if total_size > 0 else 0

                        await on_progress(DownloadProgress(
                            task_id=task_id,
                            status="running",
                            progress=min(progress, 1.0),
                            total_size=total_size,
                            downloaded_size=current_downloaded,
                            speed=speed_str,
                            filename=display_name
                        ))

                    result_path = await download_future

                    _, follow_ups = primary_and_follow_ups(ver_config)
                    if follow_ups:
                        label_parts = [display_name]
                        for spec in follow_ups:
                            repo_label = str(spec.get("name") or str(spec["repo_id"]).split("/")[-1])
                            label_parts.append(repo_label)
                            companion_label = " + ".join(label_parts)

                            def do_download_c(_spec=spec):
                                return self._sync_download_bundle_repo(
                                    _spec, default_source="huggingface",
                                )

                            c_future = loop.run_in_executor(None, do_download_c)
                            last_b = calc_downloaded_for(target)
                            last_report_time = time.time()
                            while not c_future.done():
                                await asyncio.sleep(2)
                                if task_id in self._cancelled_downloads:
                                    break
                                cur = calc_downloaded_for(target)
                                now = time.time()
                                dt = now - last_report_time
                                speed_str = ""
                                if dt >= 1:
                                    speed_bytes = (cur - last_b) / dt
                                    speed_str = format_speed(speed_bytes)
                                    last_b = cur
                                    last_report_time = now
                                prog = cur / total_size if total_size > 0 else 0.0
                                await on_progress(DownloadProgress(
                                    task_id=task_id,
                                    status="running",
                                    progress=min(prog, 1.0),
                                    total_size=total_size,
                                    downloaded_size=cur,
                                    speed=speed_str,
                                    filename=companion_label,
                                ))
                            await c_future
                        final_downloaded = calc_downloaded_for(target)
                        await self._finalize_version_install(
                            model_name=model_name,
                            version=version,
                            ver_config=ver_config,
                            target=target,
                            task_id=task_id,
                            display_name=companion_label,
                            on_progress=on_progress,
                        )
                        await on_progress(DownloadProgress(
                            task_id=task_id,
                            status="completed",
                            progress=1.0,
                            total_size=total_size,
                            downloaded_size=final_downloaded,
                            speed="",
                            filename=companion_label,
                        ))
                        result = result_path
                    else:
                        final_downloaded = calc_downloaded_for(target)
                        await self._finalize_version_install(
                            model_name=model_name,
                            version=version,
                            ver_config=ver_config,
                            target=target,
                            task_id=task_id,
                            display_name=display_name,
                            on_progress=on_progress,
                        )
                        await on_progress(DownloadProgress(
                            task_id=task_id,
                            status="completed",
                            progress=1.0,
                            total_size=total_size,
                            downloaded_size=final_downloaded,
                            speed="",
                            filename=display_name,
                        ))
                        result = result_path

                except asyncio.CancelledError:
                    await on_progress(DownloadProgress(
                        task_id=task_id,
                        status="cancelled",
                        progress=0,
                        filename=display_name
                    ))
                    raise
                except Exception as e:
                    await on_progress(DownloadProgress(
                        task_id=task_id,
                        status="failed",
                        progress=0,
                        error_message=str(e),
                        filename=display_name
                    ))
                    raise
            elif source == "modelscope" and repo_id:
                # Inline ModelScope download: snapshot_download + poll directory size for progress
                from modelscope import snapshot_download

                def calc_ms_tree(p: Path) -> int:
                    if not p.exists():
                        return 0
                    total = 0
                    for dirpath, _dirnames, filenames in os.walk(p):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            try:
                                total += os.path.getsize(fp)
                            except OSError:
                                pass
                    temp_dir = p / "._____temp"
                    if temp_dir.exists():
                        for dirpath, _dirnames, filenames in os.walk(temp_dir):
                            for f in filenames:
                                fp = os.path.join(dirpath, f)
                                try:
                                    total += os.path.getsize(fp)
                                except OSError:
                                    pass
                    return total

                def format_speed_ms(bytes_per_sec):
                    if bytes_per_sec > 1024 * 1024 * 1024:
                        return f"{bytes_per_sec / (1024 * 1024 * 1024):.1f} GB/s"
                    elif bytes_per_sec > 1024 * 1024:
                        return f"{bytes_per_sec / (1024 * 1024):.1f} MB/s"
                    elif bytes_per_sec > 1024:
                        return f"{bytes_per_sec / 1024:.1f} KB/s"
                    else:
                        return f"{bytes_per_sec:.1f} B/s"

                total_size = estimated_size if estimated_size > 0 else (existing_progress.total_size if existing_progress else 0)

                loop = asyncio.get_event_loop()

                def do_download_ms():
                    # Check network connectivity first
                    if not self._check_modelscope_connectivity():
                        raise ConnectionError(
                            "Cannot connect to ModelScope, please check network connection"
                        )
                    allow_patterns = None
                    if ver_config:
                        primary_spec, _ = primary_and_follow_ups(ver_config)
                        if primary_spec and primary_spec.get("allow_patterns"):
                            allow_patterns = primary_spec["allow_patterns"]
                        else:
                            allow_patterns = ver_config.get("allow_patterns")
                    return snapshot_download(
                        model_id=repo_id,
                        local_dir=str(target),
                        allow_patterns=allow_patterns if allow_patterns else None,
                    )

                download_future = loop.run_in_executor(None, do_download_ms)

                # Poll directory size to calculate progress
                last_downloaded = calc_ms_tree(target)
                last_report_time = time.time()

                try:
                    while not download_future.done():
                        await asyncio.sleep(2)  # Check every 2 seconds
                        if task_id in self._cancelled_downloads:
                            break

                        current_downloaded = calc_ms_tree(target)
                        now = time.time()

                        # Calculate speed
                        dt = now - last_report_time
                        speed_str = ""
                        if dt >= 1:
                            speed_bytes = (current_downloaded - last_downloaded) / dt
                            speed_str = format_speed_ms(speed_bytes)
                            last_downloaded = current_downloaded
                            last_report_time = now

                        progress = current_downloaded / total_size if total_size > 0 else 0

                        await on_progress(DownloadProgress(
                            task_id=task_id,
                            status="running",
                            progress=min(progress, 1.0),
                            total_size=total_size,
                            downloaded_size=current_downloaded,
                            speed=speed_str,
                            filename=display_name
                        ))

                    result_path = await download_future

                    _, follow_ups = primary_and_follow_ups(ver_config)
                    if follow_ups:
                        label_parts = [display_name]
                        for spec in follow_ups:
                            repo_label = str(spec.get("name") or str(spec["repo_id"]).split("/")[-1])
                            label_parts.append(repo_label)
                            companion_label = " + ".join(label_parts)

                            def do_download_ms_c(_spec=spec):
                                return self._sync_download_bundle_repo(
                                    _spec, default_source="modelscope",
                                )

                            c_future = loop.run_in_executor(None, do_download_ms_c)
                            last_b = calc_ms_tree(target)
                            last_report_time = time.time()
                            while not c_future.done():
                                await asyncio.sleep(2)
                                if task_id in self._cancelled_downloads:
                                    break
                                cur = calc_ms_tree(target)
                                now = time.time()
                                dt = now - last_report_time
                                speed_str = ""
                                if dt >= 1:
                                    speed_bytes = (cur - last_b) / dt
                                    speed_str = format_speed_ms(speed_bytes)
                                    last_b = cur
                                    last_report_time = now
                                prog = cur / total_size if total_size > 0 else 0.0
                                await on_progress(DownloadProgress(
                                    task_id=task_id,
                                    status="running",
                                    progress=min(prog, 1.0),
                                    total_size=total_size,
                                    downloaded_size=cur,
                                    speed=speed_str,
                                    filename=companion_label,
                                ))
                            await c_future
                        final_downloaded = calc_ms_tree(target)
                        await self._finalize_version_install(
                            model_name=model_name,
                            version=version,
                            ver_config=ver_config,
                            target=target,
                            task_id=task_id,
                            display_name=companion_label,
                            on_progress=on_progress,
                        )
                        await on_progress(DownloadProgress(
                            task_id=task_id,
                            status="completed",
                            progress=1.0,
                            total_size=total_size,
                            downloaded_size=final_downloaded,
                            speed="",
                            filename=companion_label,
                        ))
                        result = result_path
                    else:
                        final_downloaded = calc_ms_tree(target)
                        await self._finalize_version_install(
                            model_name=model_name,
                            version=version,
                            ver_config=ver_config,
                            target=target,
                            task_id=task_id,
                            display_name=display_name,
                            on_progress=on_progress,
                        )
                        await on_progress(DownloadProgress(
                            task_id=task_id,
                            status="completed",
                            progress=1.0,
                            total_size=total_size,
                            downloaded_size=final_downloaded,
                            speed="",
                            filename=display_name,
                        ))
                        result = result_path

                except asyncio.CancelledError:
                    await on_progress(DownloadProgress(
                        task_id=task_id,
                        status="cancelled",
                        progress=0,
                        filename=display_name
                    ))
                    raise
                except Exception as e:
                    await on_progress(DownloadProgress(
                        task_id=task_id,
                        status="failed",
                        progress=0,
                        error_message=str(e),
                        filename=display_name
                    ))
                    raise
            elif download_url:
                # Download from download_url, target is a file under a directory
                # If files specifies specific files, download one by one
                files = config.get("files")
                if files and isinstance(files, list):
                    for i, file_pattern in enumerate(files):
                # Simplified: assume files is a list of specific filenames
                        file_url = f"{download_url}/{file_pattern}" if not file_pattern.startswith("http") else file_pattern
                        file_target = target / file_pattern
                        await self._http_downloader.download(
                            task_id=f"{task_id}_{i}",
                            url=file_url,
                            target=file_target,
                            progress_callback=on_progress
                        )
                    result = str(target)
                else:
                    # Single file download, infer filename from URL
                    filename = download_url.split("/")[-1].split("?")[0] or "model.safetensors"
                    file_target = target / filename
                    result = await self._http_downloader.download(
                        task_id=task_id,
                        url=download_url,
                        target=file_target,
                        progress_callback=on_progress
                    )
            else:
                loc = get_locale()
                raise ValueError(tt("error.model_no_download_source", loc, name=model_name))

            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            self._active_model_downloads.pop(dedup_key, None)
            self._persist_downloads()
            return result

        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            self._active_model_downloads.pop(dedup_key, None)
            self._persist_downloads()
            raise
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            self._active_model_downloads.pop(dedup_key, None)
            self._persist_downloads()
            raise

    async def download_lora(self, url: str, filename: str,
                           progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
                           existing_task_id: Optional[str] = None) -> str:
        """Download LoRA (generic HTTP)."""
        if existing_task_id and existing_task_id in self._downloads:
            task_id = existing_task_id
            task = self._downloads[task_id]
            task.status = TaskStatus.RUNNING
            target = Path(task.target_path)
        else:
            task_id = str(uuid.uuid4())
            target = self._path_resolver.get_loras_dir() / filename
            target.parent.mkdir(parents=True, exist_ok=True)

            task = DownloadTask(
                id=task_id,
                url=url,
                target_path=str(target)
            )
            task._is_lora = True
            task._filename = filename
            self._downloads[task_id] = task
            task.status = TaskStatus.RUNNING

        async def on_progress(progress: DownloadProgress):
            self._progress[task_id] = progress
            self._persist_downloads()
            if progress_callback:
                await self._async_callback(progress_callback, progress)

        # Send initial progress immediately so the API layer returns task_id without blocking the frontend
        if progress_callback:
            await self._async_callback(progress_callback, DownloadProgress(
                task_id=task_id,
                status="running",
                progress=0,
                total_size=0,
                downloaded_size=0,
                speed="",
                error_message="",
                filename=filename
            ))

        try:
            result = await self._http_downloader.download(
                task_id=task_id,
                url=url,
                target=target,
                progress_callback=on_progress
            )
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            self._persist_downloads()
            return result
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            self._persist_downloads()
            raise
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            self._persist_downloads()
            raise

    async def download_lora_from_hub(
        self,
        source: str,
        *,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
        url: Optional[str] = None,
        civitai_version_id: Optional[int] = None,
        base_model: Optional[str] = None,
        display_name: Optional[str] = None,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
        existing_task_id: Optional[str] = None,
    ) -> str:
        """Download a remote LoRA from Hugging Face, ModelScope, CivitAI, or direct URL."""
        from backend.services.lora_search import (
            resolve_huggingface_lora_filename,
            resolve_huggingface_lora_url,
            resolve_modelscope_lora_filename,
            resolve_modelscope_lora_url,
        )
        from backend.third_party.civitai_client import CivitAIClient

        src = (source or "").strip().lower()
        download_url = (url or "").strip()
        target_name = (filename or "").strip()

        if src == "civitai":
            if not download_url and civitai_version_id:
                client = CivitAIClient(api_key=self._http_downloader._civitai_token)
                try:
                    version = await client.get_model_version(civitai_version_id)
                    primary = next((f for f in version.files if f.primary), version.files[0] if version.files else None)
                    if not primary:
                        raise RuntimeError("CivitAI version has no downloadable file")
                    download_url = primary.download_url
                    target_name = target_name or primary.name or f"civitai-{civitai_version_id}.safetensors"
                finally:
                    await client.close()
        elif src == "huggingface":
            if not repo_id:
                raise ValueError("repo_id is required for Hugging Face LoRA download")
            if not target_name:
                target_name = await resolve_huggingface_lora_filename(
                    repo_id, hf_token=self._token
                )
            if not download_url:
                download_url = resolve_huggingface_lora_url(repo_id, target_name)
        elif src == "modelscope":
            if not repo_id:
                raise ValueError("repo_id is required for ModelScope LoRA download")
            if not target_name:
                target_name = resolve_modelscope_lora_filename(repo_id)
            if not download_url:
                download_url = resolve_modelscope_lora_url(repo_id, target_name)
        elif src in ("http", "url"):
            if not download_url:
                raise ValueError("url is required for direct LoRA download")
        else:
            raise ValueError(f"Unsupported LoRA download source: {source}")

        if not download_url:
            raise RuntimeError("Could not resolve LoRA download URL")
        if not target_name:
            target_name = download_url.rsplit("/", 1)[-1].split("?")[0] or "lora.safetensors"
        if not target_name.endswith(".safetensors"):
            target_name = f"{target_name}.safetensors"

        result_path = await self.download_lora(
            download_url,
            target_name,
            progress_callback=progress_callback,
            existing_task_id=existing_task_id,
        )
        if base_model:
            self._register_remote_lora(
                result_path,
                base_model=base_model,
                display_name=display_name or repo_id or target_name,
                repo_id=repo_id or "",
                remote_hub_source=src,
            )
        return result_path

    def _register_remote_lora(
        self,
        absolute_path: str,
        *,
        base_model: str,
        display_name: str,
        repo_id: str,
        remote_hub_source: str,
    ) -> None:
        from backend.engine.training.user_lora_registry import list_user_loras, register_user_lora

        project_root = self._path_resolver.get_project_root()
        config_dir = self._path_resolver.get_workspace_config_dir()
        path = Path(absolute_path)
        try:
            rel = path.relative_to(project_root)
        except ValueError:
            rel = Path("models/Lora") / path.name
        local_path = str(rel).replace("\\", "/")

        for item in list_user_loras(config_dir):
            if str(item.get("local_path") or "") == local_path:
                return
            if repo_id and str(item.get("repo_id") or "") == repo_id and str(item.get("base_model") or "") == base_model:
                return

        register_user_lora(
            config_dir,
            name=display_name.strip() or path.stem,
            base_model=base_model.split(":", 1)[0].strip(),
            local_path=local_path,
            source="remote_download",
            repo_id=repo_id,
            remote_hub_source=remote_hub_source,
        )

    def get_registry_models(self) -> Dict[str, Any]:
        """Expanded registry ``models`` map for download/search helpers."""
        return self._load_registry()

    def list_downloads(self) -> List[DownloadTask]:
        """List all download tasks."""
        return list(self._downloads.values())

    async def cancel_download(self, task_id: str) -> bool:
        """Cancel a download task."""
        task = self._downloads.get(task_id)
        if not task:
            return False

        self._cancelled_downloads.add(task_id)
        await self._http_downloader.cancel(task_id)

        task.status = TaskStatus.CANCELLED
        model_name = getattr(task, '_model_name', None)
        version = getattr(task, '_version', None)
        if model_name is not None:
            self._active_model_downloads.pop((model_name, version), None)
        self._persist_downloads()
        return True

    def delete_download(self, task_id: str) -> bool:
        """Delete a download task (remove from memory and persistence)."""
        task = self._downloads.get(task_id)
        if not task:
            return False

        # If the task is still running, cancel first
        if task.status == TaskStatus.RUNNING:
            self._cancelled_downloads.add(task_id)
            # Async cancel HTTP download
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._http_downloader.cancel(task_id))
            except RuntimeError:
                pass

        # Clean up dedup mapping
        model_name = getattr(task, '_model_name', None)
        version = getattr(task, '_version', None)
        if model_name is not None:
            self._active_model_downloads.pop((model_name, version), None)

        del self._downloads[task_id]
        self._progress.pop(task_id, None)
        self._persist_downloads()
        return True

    async def resume_download(self, task_id: str,
                              progress_callback: Optional[Callable[[DownloadProgress], None]] = None) -> str:
        """Resume a download task (after process restart).

        Determines whether this is a model download or LoRA download based on
        persisted metadata, and restarts the download reusing the original task_id.
        """
        task = self._downloads.get(task_id)
        if not task or task.status != TaskStatus.PAUSED:
            loc = get_locale()
            raise ValueError(tt("error.download_not_paused", loc, id=task_id))

        model_name = getattr(task, '_model_name', None)
        version = getattr(task, '_version', None)
        is_lora = getattr(task, '_is_lora', False)
        filename = getattr(task, '_filename', None)

        if model_name:
            return await self.download_model(
                model_name, version=version,
                progress_callback=progress_callback,
                existing_task_id=task_id
            )
        elif is_lora and filename:
            return await self.download_lora(
                task.url, filename,
                progress_callback=progress_callback,
                existing_task_id=task_id
            )
        else:
            loc = get_locale()
            raise ValueError(tt("error.download_missing_metadata", loc, id=task_id))

    def get_progress(self, task_id: str) -> Optional[DownloadProgress]:
        """Get progress of a single download task."""
        return self._progress.get(task_id)

    async def convert_model(self, model_name: str, from_version: str, to_version: str,
                           progress_callback: Optional[Callable[[ConversionTask], None]] = None) -> str:
        """Build a derived weight layout via int4|int8 MLX quantization (``to_version`` names)."""
        config = self.get_model_download_config(model_name)
        if not config:
            loc = get_locale()
            raise ValueError(tt("error.model_not_in_registry", loc, name=model_name))

        versions = config.get("versions", {})
        from_ver_config = versions.get(from_version)
        to_ver_config = versions.get(to_version)

        if not from_ver_config:
            loc = get_locale()
            raise ValueError(tt("error.source_version_not_found", loc, version=from_version))
        if not to_ver_config:
            loc = get_locale()
            raise ValueError(tt("error.target_version_not_found", loc, version=to_version))

        loc = get_locale()
        if to_ver_config.get("source_type") != "derived":
            raise ValueError(tt("error.target_version_not_derived", loc, version=to_version))
        declared_parent = to_ver_config.get("from_version")
        if not declared_parent:
            raise ValueError(tt("error.derived_version_missing_parent", loc, version=to_version))
        if declared_parent != from_version:
            raise ValueError(
                tt(
                    "error.derived_version_parent_mismatch",
                    loc,
                    version=to_version,
                    expected=declared_parent,
                    got=from_version,
                )
            )

        from_path = self._resolve_version_path(from_ver_config)
        if not from_path.exists():
            loc = get_locale()
            raise ValueError(tt("error.source_version_not_ready", loc, version=from_version))

        to_path = self._resolve_version_path(to_ver_config)
        if to_path.exists():
            return str(to_path)

        task_id = str(uuid.uuid4())
        task = ConversionTask(
            id=task_id,
            model_name=model_name,
            from_version=from_version,
            to_version=to_version,
            status=TaskStatus.RUNNING,
            progress=0.0,
            stage="pending",
            output_path=str(to_path)
        )
        self._conversions[task_id] = task
        self._conversion_events[task_id] = asyncio.Event()

        async def update_progress(stage: str, progress: float):
            task.stage = stage
            task.progress = progress
            if progress_callback:
                await self._async_callback(progress_callback, task)

        if progress_callback:
            await self._async_callback(progress_callback, task)

        try:
            quantize_bits = (to_ver_config.get("quantization") or {}).get("bits")
            if not quantize_bits:
                raise ValueError(tt("error.cannot_determine_quantization", get_locale(), version=to_version))

            await update_progress("loading", 0.1)

            family = str(config.get("family") or "")
            from backend.core.derived_quant_layout import (
                copy_non_quantized_bundle,
                resolve_derived_quant_layout,
            )

            plan = resolve_derived_quant_layout(
                family=family,
                from_root=from_path,
                to_root=to_path,
                to_ver_config=to_ver_config,
            )

            def _quantize_target(
                load_paths: tuple[Path, ...],
                *,
                bits: int,
                output_dir: Path,
                shard_prefix: str,
                single_output_file: Path | None,
            ) -> int:
                import mlx.core as mx

                from backend.core.derived_quant_mlx import (
                    quantize_linear_weights_dict,
                    save_quantized_weight_bundle,
                )

                weights: dict[str, Any] = {}
                for sf in load_paths:
                    weights.update(dict(mx.load(str(sf))))
                if not weights:
                    raise RuntimeError(f"No safetensors weights found in {load_paths!r}")
                quantized = quantize_linear_weights_dict(weights, bits)
                return save_quantized_weight_bundle(
                    quantized,
                    output_dir=output_dir,
                    shard_prefix=shard_prefix,
                    bits=bits,
                    single_output_file=single_output_file,
                )

            def _do_conversion():
                """Execute quantization in a thread (blocking MLX computation)."""
                from backend.core.derived_quant_layout import copy_component_companion_files

                done = _quantize_target(
                    plan.load_paths,
                    bits=int(quantize_bits),
                    output_dir=plan.output_dir,
                    shard_prefix=plan.output_shard_prefix,
                    single_output_file=plan.single_output_file,
                )
                total = len([k for k in plan.load_paths])

                for target in plan.component_targets:
                    _quantize_target(
                        target.load_paths,
                        bits=int(target.bits),
                        output_dir=target.output_dir,
                        shard_prefix=target.output_shard_prefix,
                        single_output_file=target.single_output_file,
                    )
                    copy_component_companion_files(
                        from_path / target.subdir,
                        to_path / target.subdir,
                    )

                copy_non_quantized_bundle(from_path, to_path, plan)

                return done, total

            done, total = await asyncio.to_thread(_do_conversion)

            if self._conversion_events[task_id].is_set():
                if to_path.exists():
                    shutil.rmtree(to_path)
                raise asyncio.CancelledError("Conversion cancelled")

            await update_progress("completed", 1.0)

            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            task.stage = "completed"
            return str(to_path)

        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            task.stage = "cancelled"
            if to_path.exists():
                shutil.rmtree(to_path)
            raise
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.stage = "error"
            task.error_message = str(e)
            if to_path.exists():
                shutil.rmtree(to_path)
            raise
        finally:
            self._conversion_events.pop(task_id, None)

    def _resolve_version_path(self, version_config: Dict[str, Any]) -> Path:
        """Resolve install root for a version (bundle_repos[0] or local_path)."""
        return self._path_resolver.resolve_registry_local_path(
            version_primary_local_path(version_config)
        )

    def list_conversions(self) -> List[ConversionTask]:
        """List all conversion tasks."""
        return list(self._conversions.values())

    async def cancel_conversion(self, task_id: str) -> bool:
        """Cancel a conversion task."""
        task = self._conversions.get(task_id)
        if not task or task.status != TaskStatus.RUNNING:
            return False

        event = self._conversion_events.get(task_id)
        if event:
            event.set()

        task.status = TaskStatus.CANCELLED
        return True

    def get_conversion_progress(self, task_id: str) -> Optional[ConversionTask]:
        """Get progress of a single conversion task."""
        return self._conversions.get(task_id)

    async def delete_model(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Delete a model or specified version.

        Uses the local_path field in models_registry.json to find and delete the model file directory.
        """
        config = self.get_model_download_config(model_name)
        if not config:
            loc = get_locale()
            return {"success": False, "deleted_paths": [], "error": tt("error.model_not_in_registry", loc, name=model_name)}

        deleted_paths = []

        if version and config.get("versions"):
            ver_config = config["versions"].get(version)
            if not ver_config:
                loc = get_locale()
                return {"success": False, "deleted_paths": [], "error": tt("error.version_not_found", loc, version=version)}

            deleted_set: set[str] = set()
            paths = bundle_local_paths(ver_config)
            if not paths:
                paths = [version_primary_local_path(ver_config)]
            for lp in paths:
                cpath = self._path_resolver.resolve_registry_local_path(lp)
                key = str(cpath)
                if key in deleted_set:
                    continue
                deleted_set.add(key)
                if cpath.exists():
                    shutil.rmtree(cpath)
                    deleted_paths.append(key)
        else:
            versions = config.get("versions", {})
            if versions:
                deleted_set = set()
                for ver_key, ver_config in versions.items():
                    paths = bundle_local_paths(ver_config)
                    if not paths:
                        paths = [version_primary_local_path(ver_config)]
                    for lp in paths:
                        cpath = self._path_resolver.resolve_registry_local_path(lp)
                        key = str(cpath)
                        if key in deleted_set:
                            continue
                        deleted_set.add(key)
                        if cpath.exists():
                            shutil.rmtree(cpath)
                            deleted_paths.append(key)
            else:
                local_path = config.get("local_path", f"models/{model_name}")
                target = self._path_resolver.resolve_registry_local_path(local_path)
                if target.exists():
                    shutil.rmtree(target)
                    deleted_paths.append(str(target))

        return {"success": True, "deleted_paths": deleted_paths, "error": None}

    @staticmethod
    async def _async_callback(callback: Callable, progress):
        """Safely invoke an async callback."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(progress)
            else:
                callback(progress)
        except Exception as e:
            print(f"[DownloadService] Callback exception: {e}")
