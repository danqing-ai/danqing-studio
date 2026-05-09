"""
下载服务实现 - 支持 HuggingFace 和 HTTP 双源下载
"""

import os
# 配置 HF 镜像站
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import time
import uuid
import json
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any

import asyncio
import shutil

from backend.core.interfaces import (
    IDownloadService, IPathResolver, IConfigStore,
    DownloadTask, DownloadProgress, TaskStatus, ConversionTask
)
from backend.core.i18n import t as tt, get_locale
from backend.core.downloaders import HTTPDownloader


class DownloadService(IDownloadService):
    """下载服务实现

    根据模型注册表的 source 字段自动选择下载方式：
    - huggingface: 使用 huggingface_hub.snapshot_download，轮询目录大小算进度
    - modelscope: 使用 modelscope.snapshot_download，轮询目录大小算进度
    - civitai / http: 使用 HTTPDownloader (aiohttp)
    """

    def __init__(self, path_resolver: IPathResolver, config_store: Optional[IConfigStore] = None):
        self._path_resolver = path_resolver
        self._config = config_store
        self._downloads: Dict[str, DownloadTask] = {}
        self._progress: Dict[str, DownloadProgress] = {}
        self._conversions: Dict[str, ConversionTask] = {}
        self._conversion_events: Dict[str, asyncio.Event] = {}
        self._persist_path = path_resolver.get_project_root() / "config" / ".download_tasks.json"

        # 读取配置中的 token
        hf_token = None
        civitai_token = None
        if self._config:
            settings = self._config.load()
            hf_token = settings.huggingface_token or None
            civitai_token = settings.civitai_token or None

        # 初始化下载器
        self._http_downloader = HTTPDownloader(civitai_token=civitai_token)
        self._cancelled_downloads: set = set()
        self._token = hf_token

        # 活跃模型下载去重：(model_name, version) -> task_id
        self._active_model_downloads: Dict[tuple, str] = {}
        # 保护去重检查和任务创建的并发锁
        self._download_lock = asyncio.Lock()

        # 加载持久化的下载任务
        self._load_persisted_downloads()

    def _persist_downloads(self) -> None:
        """持久化下载任务到 JSON 文件（保留所有状态，包括已完成和失败）"""
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
            print(f"[DownloadService] 持久化下载任务失败: {e}")

    def _load_persisted_downloads(self) -> None:
        """从 JSON 文件加载持久化的下载任务

        进程重启后，将 running 状态的任务标记为 paused，
        等待用户手动恢复。
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
                    target_path=item["target_path"]
                )
                raw_status = item.get("status", "running")
                # 进程重启后，原来的 running 任务变为 paused
                if raw_status == "running":
                    task.status = TaskStatus.PAUSED
                else:
                    task.status = TaskStatus(raw_status)
                task.progress = item.get("progress", 0)
                # 恢复元数据
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
            print(f"[DownloadService] 加载持久化下载任务失败: {e}")



    def _load_registry(self) -> Dict[str, Any]:
        """加载模型注册表"""
        registry_path = self._path_resolver.get_project_root() / "config" / "models_registry.json"
        if not registry_path.exists():
            return {}
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                return json.load(f).get("models", {})
        except Exception as e:
            print(f"[DownloadService] 加载模型注册表失败: {e}")
            return {}

    def get_model_download_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取模型的下载配置信息"""
        registry = self._load_registry()
        return registry.get(model_name)

    @staticmethod
    def _check_hf_connectivity(timeout: float = 10.0) -> bool:
        """检查 HuggingFace 镜像站是否可访问"""
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
        """检查 ModelScope（魔塔社区）是否可访问"""
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
        """将注册表中的 size 字符串（如 '23.8GB'）解析为字节数"""
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

    async def download_model(self, model_name: str, version: Optional[str] = None,
                            progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
                            existing_task_id: Optional[str] = None) -> str:
        """按注册表模型名称下载模型（支持基础模型和 LoRA）"""
        config = self.get_model_download_config(model_name)
        if not config:
            loc = get_locale()
            raise ValueError(tt("error.model_not_in_registry", loc, name=model_name))

        source = config.get("source", "huggingface")
        repo_id = config.get("repo_id")
        download_url = config.get("download_url")
        local_path = config.get("local_path", f"models/{model_name}")

        # 构造友好显示名称：模型名称 + 版本名称
        model_display_name = model_name
        name_cfg = config.get("name", {})
        if isinstance(name_cfg, dict):
            model_display_name = name_cfg.get("zh") or name_cfg.get("en") or model_name

        version_display_name = ""
        ver_config = None
        if version and config.get("versions"):
            ver_config = config["versions"].get(version)
            if ver_config:
                version_display_name = ver_config.get("name", version)
                if not repo_id:
                    repo_id = ver_config.get("repo_id")
                if not download_url:
                    download_url = ver_config.get("download_url")
                if ver_config.get("local_path"):
                    local_path = ver_config["local_path"]

        # 解析注册表中该版本的预估大小（HF API 失败时用作 fallback）
        estimated_size = 0
        if ver_config and ver_config.get("size"):
            estimated_size = self._parse_size_to_bytes(ver_config["size"])
        elif config.get("size"):
            estimated_size = self._parse_size_to_bytes(config["size"])

        # 统一显示名称
        display_name = model_display_name
        if version_display_name:
            display_name = f"{model_display_name} {version_display_name}"

        # 解析本地路径
        if local_path.startswith("models/"):
            # 提取相对路径（可能含子目录，如 Base/Z-Image-Turbo-mflux-4bit-fp16）
            rel_path = local_path[len("models/"):]
            target = self._path_resolver.get_models_dir() / rel_path
        else:
            target = Path(local_path)

        # 确保目标目录存在
        target.mkdir(parents=True, exist_ok=True)

        # 去重检查：同一模型同一版本只能有一个运行中/排队的下载任务
        dedup_key = (model_name, version)
        existing_active_id = self._active_model_downloads.get(dedup_key)
        if existing_active_id and existing_active_id in self._downloads:
            existing_task = self._downloads[existing_active_id]
            if existing_task.status == TaskStatus.RUNNING:
                # 已存在运行中任务，直接返回已有 task_id
                return existing_active_id
            elif existing_task.status == TaskStatus.PAUSED:
                # 已存在暂停任务，如果是恢复下载则继续，否则返回已有 task_id
                if not existing_task_id or existing_task_id != existing_active_id:
                    return existing_active_id

        if existing_task_id and existing_task_id in self._downloads:
            task_id = existing_task_id
            task = self._downloads[task_id]
            task.status = TaskStatus.RUNNING
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

        # 内部进度回调（存储进度 + 外部回调 + 持久化）
        async def on_progress(progress: DownloadProgress):
            self._progress[task_id] = progress
            self._persist_downloads()
            if progress_callback:
                await self._async_callback(progress_callback, progress)

        # 恢复下载时保留已有进度，不重置为 0
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
                # 内联 HuggingFace 下载：snapshot_download + 轮询目录大小算进度
                from huggingface_hub import snapshot_download

                def calc_downloaded():
                    if not target.exists():
                        return 0
                    total = 0
                    for dirpath, dirnames, filenames in os.walk(target):
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
                    # 先检查网络是否通畅，避免 snapshot_download 静默回退到已有目录
                    if not self._check_hf_connectivity():
                        raise ConnectionError(
                            "无法连接到模型下载服务器 (HF_ENDPOINT=%s)，请检查网络连接" % os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
                        )
                    # 让 HF client 使用默认 tqdm，保留其日志输出
                    return snapshot_download(
                        repo_id=repo_id,
                        local_dir=str(target),
                        token=self._token,
                    )

                download_future = loop.run_in_executor(None, do_download)

                # 轮询目录大小计算进度
                last_downloaded = calc_downloaded()
                last_report_time = time.time()

                try:
                    while not download_future.done():
                        await asyncio.sleep(2)  # 每2秒检查一次
                        if task_id in self._cancelled_downloads:
                            break

                        current_downloaded = calc_downloaded()
                        now = time.time()

                        # 计算速度
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

                    # 最终进度
                    final_downloaded = calc_downloaded()
                    await on_progress(DownloadProgress(
                        task_id=task_id,
                        status="completed",
                        progress=1.0,
                        total_size=total_size,
                        downloaded_size=final_downloaded,
                        speed="",
                        filename=display_name
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
                # 内联 ModelScope（魔塔社区）下载：snapshot_download + 轮询目录大小算进度
                from modelscope import snapshot_download

                def calc_downloaded_ms():
                    if not target.exists():
                        return 0
                    total = 0
                    # 计算目标目录中的文件大小
                    for dirpath, dirnames, filenames in os.walk(target):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            try:
                                total += os.path.getsize(fp)
                            except OSError:
                                pass
                    # 计算临时目录中的文件大小（ModelScope 断点续传）
                    temp_dir = target / '._____temp'
                    if temp_dir.exists():
                        for dirpath, dirnames, filenames in os.walk(temp_dir):
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
                    # 先检查网络是否通畅
                    if not self._check_modelscope_connectivity():
                        raise ConnectionError(
                            "无法连接到 ModelScope（魔塔社区），请检查网络连接"
                        )
                    return snapshot_download(
                        model_id=repo_id,
                        local_dir=str(target),
                    )

                download_future = loop.run_in_executor(None, do_download_ms)

                # 轮询目录大小计算进度
                last_downloaded = calc_downloaded_ms()
                last_report_time = time.time()

                try:
                    while not download_future.done():
                        await asyncio.sleep(2)  # 每2秒检查一次
                        if task_id in self._cancelled_downloads:
                            break

                        current_downloaded = calc_downloaded_ms()
                        now = time.time()

                        # 计算速度
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

                    # 最终进度
                    final_downloaded = calc_downloaded_ms()
                    await on_progress(DownloadProgress(
                        task_id=task_id,
                        status="completed",
                        progress=1.0,
                        total_size=total_size,
                        downloaded_size=final_downloaded,
                        speed="",
                        filename=display_name
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
                # 从 download_url 下载，目标为目录下的文件
                # 如果 files 指定了具体文件，逐个下载
                files = config.get("files")
                if files and isinstance(files, list):
                    for i, file_pattern in enumerate(files):
                        # 这里简化处理，假设 files 是具体文件名
                        file_url = f"{download_url}/{file_pattern}" if not file_pattern.startswith("http") else file_pattern
                        file_target = target / file_pattern
                        await self._http_downloader.download(
                            task_id=f"{task_id}_{i}",
                            source=file_url,
                            target=file_target,
                            progress_callback=on_progress
                        )
                    result = str(target)
                else:
                    # 单文件下载，从 URL 推断文件名
                    filename = download_url.split("/")[-1].split("?")[0] or "model.safetensors"
                    file_target = target / filename
                    result = await self._http_downloader.download(
                        task_id=task_id,
                        source=download_url,
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
        """下载 LoRA（HTTP 通用）"""
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

        # 立即发送初始进度，让 API 层立刻返回 task_id，不阻塞前端
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
                source=url,
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

    def list_downloads(self) -> List[DownloadTask]:
        """列出所有下载任务"""
        return list(self._downloads.values())

    async def cancel_download(self, task_id: str) -> bool:
        """取消下载任务"""
        task = self._downloads.get(task_id)
        if not task:
            return False

        self._cancelled_downloads.add(task_id)
        await self._http_downloader.cancel(task_id)

        task.status = TaskStatus.CANCELLED
        # 清理去重映射
        model_name = getattr(task, '_model_name', None)
        version = getattr(task, '_version', None)
        if model_name is not None:
            self._active_model_downloads.pop((model_name, version), None)
        self._persist_downloads()
        return True

    def delete_download(self, task_id: str) -> bool:
        """删除下载任务（从内存和持久化中移除）"""
        task = self._downloads.get(task_id)
        if not task:
            return False

        # 如果任务仍在运行，先取消
        if task.status == TaskStatus.RUNNING:
            self._cancelled_downloads.add(task_id)
            # 异步取消 HTTP 下载
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._http_downloader.cancel(task_id))
            except RuntimeError:
                pass

        # 清理去重映射
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
        """恢复下载任务（进程重启后）

        根据持久化的元数据判断是模型下载还是 LoRA 下载，
        重新启动下载并复用原 task_id。
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
        """获取单个下载任务的进度"""
        return self._progress.get(task_id)

    async def convert_model(self, model_name: str, from_version: str, to_version: str,
                           progress_callback: Optional[Callable[[ConversionTask], None]] = None) -> str:
        """通过 MLX 原生 API 生成量化版本（不依赖 mflux）。

        流程：
        1. 加载源目录（fp16）的 safetensors 权重字典
        2. 对每个 2D weight 创建 nn.Linear 并用 to_quantized() 量化
        3. 将量化后的参数（weight/scales/biases）分片保存到目标目录
        4. 复制 VAE / text_encoder / tokenizer 等未量化组件
        """
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
            quantize_bits = 4 if "int4" in to_version else (8 if "int8" in to_version else None)
            if not quantize_bits:
                raise ValueError(tt("error.cannot_determine_quantization", get_locale(), version=to_version))

            await update_progress("loading", 0.1)

            def _do_conversion():
                """在线程中执行量化（阻塞 MLX 计算）。"""
                import mlx.core as mx
                import mlx.nn as nn
                from mlx.utils import tree_flatten

                # 1. 加载源权重
                weights = {}
                transformer_dir = from_path / "transformer"
                if transformer_dir.exists():
                    for sf in sorted(transformer_dir.glob("*.safetensors")):
                        weights.update(dict(mx.load(str(sf))))
                else:
                    # 兜底：直接在根目录找
                    for sf in sorted(from_path.glob("*.safetensors")):
                        weights.update(dict(mx.load(str(sf))))

                if not weights:
                    raise RuntimeError(f"No safetensors weights found in {from_path}")

                # 2. 量化 — 参考 mflux ModelSaver 思路：
                #    对每个 2D Linear weight 创建 nn.Linear → to_quantized()
                #    跳过 Embedding / 1D bias / norm 等
                quantized = {}
                processed_bias_keys = set()
                total = len([k for k in weights if k.endswith(".weight")])
                done = 0

                for key, tensor in weights.items():
                    if not key.endswith(".weight"):
                        continue
                    if tensor.ndim != 2 or tensor.shape[0] <= 1 or tensor.shape[1] <= 1:
                        continue
                    # 跳过 Embedding（key 含 embed / vocab / token）
                    if any(x in key.lower() for x in ("embed", "vocab", "token")):
                        quantized[key] = tensor
                        continue

                    in_features = int(tensor.shape[1])
                    out_features = int(tensor.shape[0])
                    bias_key = key.replace(".weight", ".bias")
                    has_bias = bias_key in weights

                    linear = nn.Linear(in_features, out_features, bias=has_bias)
                    linear.weight = tensor
                    if has_bias:
                        linear.bias = weights[bias_key]
                        processed_bias_keys.add(bias_key)

                    q_linear = linear.to_quantized(bits=quantize_bits)
                    base = key[:-7]  # strip ".weight"
                    quantized[f"{base}.weight"] = q_linear.weight
                    quantized[f"{base}.scales"] = q_linear.scales
                    quantized[f"{base}.biases"] = q_linear.biases
                    done += 1

                # 保留未量化参数（norm / conv / 未处理的 bias 等）
                for key, tensor in weights.items():
                    if key in processed_bias_keys or key in quantized:
                        continue
                    quantized[key] = tensor

                # 3. 分片保存（同 mflux：最大 2GB/片）
                max_shard_bytes = 2 << 30
                shards = []
                current_shard = {}
                current_size = 0

                for key, value in quantized.items():
                    if current_size + value.nbytes > max_shard_bytes and current_shard:
                        shards.append(current_shard)
                        current_shard = {}
                        current_size = 0
                    current_shard[key] = value
                    current_size += value.nbytes
                if current_shard:
                    shards.append(current_shard)

                # 确保输出目录
                to_path.mkdir(parents=True, exist_ok=True)
                transformer_out = to_path / "transformer"
                transformer_out.mkdir(parents=True, exist_ok=True)

                weight_map = {}
                for i, shard in enumerate(shards):
                    shard_name = f"model_{i:05d}.safetensors"
                    mx.save_safetensors(
                        str(transformer_out / shard_name),
                        shard,
                        metadata={"quantization_level": str(quantize_bits)},
                    )
                    for k in shard.keys():
                        weight_map[k] = shard_name

                # 写 model.safetensors.index.json（兼容 HF 格式）
                index_data = {
                    "metadata": {"quantization_level": str(quantize_bits)},
                    "weight_map": weight_map,
                }
                with open(transformer_out / "model.safetensors.index.json", "w") as f:
                    json.dump(index_data, f, indent=2)

                # 4. 复制未量化组件（VAE / text_encoder / tokenizer 等）
                for subdir in ("vae", "text_encoder", "tokenizer", "text_encoder_2", "tokenizer_2"):
                    src = from_path / subdir
                    if src.exists():
                        dst = to_path / subdir
                        if not dst.exists():
                            shutil.copytree(src, dst)

                return done, total

            done, total = await asyncio.to_thread(_do_conversion)

            if self._conversion_events[task_id].is_set():
                if to_path.exists():
                    shutil.rmtree(to_path)
                raise asyncio.CancelledError("转换已取消")

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
        """解析版本的本地路径"""
        local_path = version_config.get("local_path", "")
        if local_path.startswith("models/"):
            rel_path = local_path[len("models/"):]
            return self._path_resolver.get_models_dir() / rel_path
        return Path(local_path)

    def list_conversions(self) -> List[ConversionTask]:
        """列出所有转换任务"""
        return list(self._conversions.values())

    async def cancel_conversion(self, task_id: str) -> bool:
        """取消转换任务"""
        task = self._conversions.get(task_id)
        if not task or task.status != TaskStatus.RUNNING:
            return False

        event = self._conversion_events.get(task_id)
        if event:
            event.set()

        task.status = TaskStatus.CANCELLED
        return True

    def get_conversion_progress(self, task_id: str) -> Optional[ConversionTask]:
        """获取单个转换任务的进度"""
        return self._conversions.get(task_id)

    async def delete_model(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """删除模型或指定版本

        根据 models_registry.json 中的 local_path 查找并删除模型文件目录。
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

            ver_path = self._resolve_version_path(ver_config)
            if ver_path.exists():
                shutil.rmtree(ver_path)
                deleted_paths.append(str(ver_path))
        else:
            versions = config.get("versions", {})
            if versions:
                for ver_key, ver_config in versions.items():
                    ver_path = self._resolve_version_path(ver_config)
                    if ver_path.exists():
                        shutil.rmtree(ver_path)
                        deleted_paths.append(str(ver_path))
            else:
                local_path = config.get("local_path", f"models/{model_name}")
                if local_path.startswith("models/"):
                    rel_path = local_path[len("models/"):]
                    target = self._path_resolver.get_models_dir() / rel_path
                else:
                    target = Path(local_path)
                if target.exists():
                    shutil.rmtree(target)
                    deleted_paths.append(str(target))

        return {"success": True, "deleted_paths": deleted_paths, "error": None}

    @staticmethod
    async def _async_callback(callback: Callable, progress):
        """安全调用异步回调"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(progress)
            else:
                callback(progress)
        except Exception as e:
            print(f"[DownloadService] 回调异常: {e}")
