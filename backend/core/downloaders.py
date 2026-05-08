"""
下载器模块 - 统一接口封装 HF Hub 和 HTTP 下载
"""

import time
import asyncio
import aiohttp
from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, Any
from pathlib import Path

from backend.core.interfaces import DownloadProgress


class IDownloader(ABC):
    """下载器抽象接口"""

    @abstractmethod
    async def download(self, task_id: str, source: str, target: Path,
                      progress_callback: Optional[Callable[[DownloadProgress], None]] = None) -> str:
        """
        执行下载

        Args:
            task_id: 任务ID
            source: 下载源（repo_id 或 url）
            target: 本地目标路径
            progress_callback: 进度回调

        Returns:
            下载完成的本地路径
        """
        pass

    @abstractmethod
    async def cancel(self, task_id: str) -> bool:
        """取消下载任务"""
        pass


class HTTPDownloader(IDownloader):
    """HTTP 通用下载器

    使用 aiohttp 流式下载，支持：
    - 断点续传（通过 Range 请求）
    - 进度回调
    - 取消下载
    - CivitAI 直链（支持 API Key）
    """

    def __init__(self, civitai_token: Optional[str] = None):
        self._civitai_token = civitai_token
        self._cancelled: set = set()
        self._active_sessions: Dict[str, aiohttp.ClientSession] = {}

    async def download(self, task_id: str, url: str, target: Path,
                      progress_callback: Optional[Callable[[DownloadProgress], None]] = None) -> str:
        """HTTP 流式下载"""
        self._cancelled.discard(task_id)

        # 确保目标目录存在
        target.parent.mkdir(parents=True, exist_ok=True)

        # 断点续传：检查已下载大小
        downloaded_size = 0
        if target.exists():
            downloaded_size = target.stat().st_size

        # 构建请求头
        headers = {}
        if downloaded_size > 0:
            headers["Range"] = f"bytes={downloaded_size}-"

        # CivitAI 鉴权
        if "civitai.com" in url and self._civitai_token:
            headers["Authorization"] = f"Bearer {self._civitai_token}"

        last_report = {"time": time.time(), "bytes": 0}
        mode = "ab" if downloaded_size > 0 else "wb"

        session = aiohttp.ClientSession()
        self._active_sessions[task_id] = session

        try:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=None)) as resp:
                if resp.status not in (200, 206):
                    raise Exception(f"HTTP {resp.status}: {resp.reason}")

                # 获取总大小
                total_size = int(resp.headers.get("content-length", 0))
                if downloaded_size > 0 and resp.status == 206:
                    # Range 请求成功，总大小 = 已下载 + 剩余
                    total_size += downloaded_size
                elif downloaded_size > 0 and resp.status == 200:
                    # 服务器不支持断点续传，从头开始
                    downloaded_size = 0
                    mode = "wb"

                with open(target, mode) as f:
                    async for chunk in resp.content.iter_chunked(8192):
                        if task_id in self._cancelled:
                            raise asyncio.CancelledError("用户取消下载")

                        f.write(chunk)
                        downloaded_size += len(chunk)

                        # 计算进度
                        progress = downloaded_size / total_size if total_size > 0 else 0

                        # 计算速度（每0.5秒报告一次）
                        now = time.time()
                        dt = now - last_report["time"]
                        if dt >= 0.5:
                            speed_bytes = (downloaded_size - last_report["bytes"]) / dt
                            speed_str = self._format_speed(speed_bytes)
                            last_report["time"] = now
                            last_report["bytes"] = downloaded_size

                            if progress_callback:
                                await self._safe_callback(progress_callback, DownloadProgress(
                                    task_id=task_id,
                                    status="running",
                                    progress=progress,
                                    total_size=total_size,
                                    downloaded_size=downloaded_size,
                                    speed=speed_str,
                                    filename=target.name
                                ))

            # 下载完成
            if progress_callback:
                await self._safe_callback(progress_callback, DownloadProgress(
                    task_id=task_id,
                    status="completed",
                    progress=1.0,
                    total_size=total_size,
                    downloaded_size=downloaded_size,
                    speed="",
                    filename=target.name
                ))

            return str(target)

        except asyncio.CancelledError:
            if progress_callback:
                await self._safe_callback(progress_callback, DownloadProgress(
                    task_id=task_id,
                    status="cancelled",
                    progress=0,
                    filename=target.name
                ))
            raise
        except Exception as e:
            if progress_callback:
                await self._safe_callback(progress_callback, DownloadProgress(
                    task_id=task_id,
                    status="failed",
                    progress=0,
                    error_message=str(e),
                    filename=target.name
                ))
            raise
        finally:
            await session.close()
            self._active_sessions.pop(task_id, None)

    async def cancel(self, task_id: str) -> bool:
        self._cancelled.add(task_id)
        session = self._active_sessions.get(task_id)
        if session:
            await session.close()
        return True

    @staticmethod
    def _format_speed(bytes_per_sec: float) -> str:
        if bytes_per_sec > 1024 * 1024 * 1024:
            return f"{bytes_per_sec / (1024 * 1024 * 1024):.1f} GB/s"
        elif bytes_per_sec > 1024 * 1024:
            return f"{bytes_per_sec / (1024 * 1024):.1f} MB/s"
        elif bytes_per_sec > 1024:
            return f"{bytes_per_sec / 1024:.1f} KB/s"
        else:
            return f"{bytes_per_sec:.1f} B/s"

    @staticmethod
    async def _safe_callback(callback: Callable, progress: DownloadProgress):
        """安全调用回调"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(progress)
            else:
                callback(progress)
        except Exception:
            pass
