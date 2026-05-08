"""
MFLUX Studio v2.0 - 接口定义层
面向接口编程，所有层之间只依赖此模块的接口定义
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Union
from enum import Enum
from datetime import datetime
from pathlib import Path


class TaskStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class GenerationParams:
    """图像生成参数
    
    通用参数 + 模型特有参数(extra_params)
    """
    prompt: str
    negative_prompt: str = ""
    model: str = ""  # 模型 key（如 "z-image-turbo"）
    version: str = ""  # 版本 key（如 "mflux-4bit"），空字符串表示默认版本
    width: int = 1024
    height: int = 1024
    steps: int = 4
    guidance: float = 3.5
    seed: Optional[int] = None
    lora: str = ""
    lora_scale: float = 0.8
    img2img: bool = False
    image_path: str = ""
    strength: float = 0.4
    edit_mode: str = "text_to_image"  # "text_to_image" | "image_to_image" | "inpainting"
    mask_path: str = ""
    # 模型特有参数：温度、max_tokens、system_prompt 等
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationTask:
    """生成任务"""
    id: str
    params: GenerationParams
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    output_path: str = ""
    error_message: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    logs: List[str] = field(default_factory=list)


# ===== 视频生成接口 =====

@dataclass
class LoRAConfig:
    """LoRA 配置"""
    path: str
    weight: float
    tags: List[str] = field(default_factory=list)


@dataclass
class VideoGenerationParams:
    """视频生成参数"""
    prompt: str
    negative_prompt: str = ""
    model: str = ""  # 模型 key（如 "ltx-2.3-distilled"）
    version: str = ""  # 版本 key
    width: int = 768
    height: int = 512
    num_frames: int = 97
    fps: int = 24
    steps: int = 4
    guide_scale: float = 3.0
    shift: float = 0.0
    seed: Optional[int] = None
    image_path: str = ""  # 图生视频起始图
    loras: List[LoRAConfig] = field(default_factory=list)
    # 模型特有参数
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoGenerationTask:
    """视频生成任务"""
    id: str
    params: VideoGenerationParams
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    output_path: str = ""
    error_message: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    logs: List[str] = field(default_factory=list)


@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    path: str
    type: str = "flux"  # flux, lora, etc.
    size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DownloadTask:
    """下载任务"""
    id: str
    url: str
    target_path: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    total_size: int = 0
    downloaded_size: int = 0
    error_message: str = ""


@dataclass
class ConversionTask:
    """模型转换任务（生成量化版本）"""
    id: str
    model_name: str
    from_version: str
    to_version: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    stage: str = ""  # loading, quantizing, saving, completed, cancelled, error
    error_message: str = ""
    output_path: str = ""


@dataclass
class AppSettings:
    """应用设置"""
    language: str = "zh"
    theme: str = "dark"
    default_model: str = ""
    auto_save_prompts: bool = True
    output_format: str = "png"
    civitai_token: str = ""
    nsfw_enabled: bool = False
    huggingface_token: str = ""
    mlx_memory_limit: int = 120
    model_cache_ttl_minutes: int = 30
    queue_image_first: bool = False
    custom_models_dir: str = ""
    custom_loras_dir: str = ""
    custom_outputs_dir: str = ""


# ===== 任务持久化接口 =====

class ITaskStore(ABC):
    """任务持久化存储接口"""

    @abstractmethod
    def save_task(self, task: Union[GenerationTask, VideoGenerationTask]) -> None:
        """保存或更新任务"""
        pass

    @abstractmethod
    def get_task(self, task_id: str) -> Optional[Union[GenerationTask, VideoGenerationTask]]:
        """获取单个任务"""
        pass

    @abstractmethod
    def list_tasks(
        self, limit: int = 100, offset: int = 0
    ) -> List[Union[GenerationTask, VideoGenerationTask]]:
        """列出任务"""
        pass

    @abstractmethod
    def delete_task(self, task_id: str) -> bool:
        """删除任务"""
        pass

    @abstractmethod
    def append_log(self, task_id: str, message: str, level: str = "info") -> None:
        """追加任务日志"""
        pass

    @abstractmethod
    def get_logs(self, task_id: str, offset: int = 0, limit: int = 1000) -> List[Dict[str, Any]]:
        """获取任务日志"""
        pass

    @abstractmethod
    def update_progress(self, task_id: str, progress: float) -> None:
        """更新任务进度"""
        pass

    @abstractmethod
    def update_status(self, task_id: str, status: TaskStatus) -> None:
        """更新任务状态"""
        pass


class IV3TaskStore(ABC):
    """v3 任务持久化（``studio.db``：JSON params、字典行视图）。与遗留 ``ITaskStore``（GenerationTask 模型）并存。"""

    @abstractmethod
    def insert_task(
        self,
        task_id: str,
        kind: str,
        model_id: str,
        params: Dict[str, Any],
        *,
        priority: int = 100,
        status: TaskStatus = TaskStatus.QUEUED,
    ) -> None:
        ...

    @abstractmethod
    def update_status(self, task_id: str, status: TaskStatus) -> None:
        ...

    @abstractmethod
    def update_progress(self, task_id: str, progress: float) -> None:
        ...

    @abstractmethod
    def update_task_priority(self, task_id: str, priority: int) -> bool:
        ...

    @abstractmethod
    def mark_running(self, task_id: str) -> None:
        ...

    @abstractmethod
    def mark_completed(self, task_id: str, result: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def mark_failed(self, task_id: str, message: str) -> None:
        ...

    @abstractmethod
    def mark_cancelled(self, task_id: str) -> None:
        ...

    @abstractmethod
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        ...

    @abstractmethod
    def list_tasks(
        self,
        limit: int = 200,
        offset: int = 0,
        *,
        kind: Optional[str] = None,
        status: Optional[str] = None,
        since: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def append_log(self, task_id: str, message: str, level: str = "info") -> None:
        ...

    @abstractmethod
    def get_logs(self, task_id: str, offset: int = 0, limit: int = 500) -> List[Dict[str, Any]]:
        ...


# ===== 引擎层接口 =====
# v3 媒体引擎见 backend.core.media_interfaces（IImageEngine / IVideoEngine）

# ===== 持久化层接口 =====

class IConfigStore(ABC):
    """配置持久化接口"""

    @abstractmethod
    def load(self) -> AppSettings:
        pass

    @abstractmethod
    def save(self, settings: AppSettings) -> None:
        pass


class IPresetStore(ABC):
    """预设持久化接口"""

    @abstractmethod
    def load_all(self) -> Dict[str, Dict[str, Any]]:
        pass

    @abstractmethod
    def save(self, name: str, preset: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def delete(self, name: str) -> None:
        pass


@dataclass
class DownloadProgress:
    """下载进度信息"""
    task_id: str
    status: str  # pending, running, completed, failed, cancelled
    progress: float  # 0.0 - 1.0
    total_size: int = 0
    downloaded_size: int = 0
    speed: str = ""  # 如 "12.5 MB/s"
    error_message: str = ""
    filename: str = ""


class IDownloadService(ABC):
    """下载服务接口 - 支持 HuggingFace、ModelScope 和 HTTP 多源下载"""

    @abstractmethod
    async def download_model(self, model_name: str,
                            progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
                            existing_task_id: Optional[str] = None) -> str:
        """按注册表模型名称下载基础模型

        根据 models_registry.json 中的 source 字段自动选择下载器：
        - huggingface: 使用 huggingface_hub 下载
        - modelscope: 使用 modelscope 下载（魔塔社区）
        - civitai/http: 使用 aiohttp 下载
        """
        pass

    @abstractmethod
    async def download_lora(self, url: str, filename: str,
                           progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
                           existing_task_id: Optional[str] = None) -> str:
        """下载 LoRA（HTTP 通用）"""
        pass

    @abstractmethod
    def list_downloads(self) -> List[DownloadTask]:
        """列出所有下载任务"""
        pass

    @abstractmethod
    async def cancel_download(self, task_id: str) -> bool:
        """取消下载任务"""
        pass

    @abstractmethod
    def delete_download(self, task_id: str) -> bool:
        """删除下载任务"""
        pass

    @abstractmethod
    def get_progress(self, task_id: str) -> Optional[DownloadProgress]:
        """获取单个下载任务的进度"""
        pass

    @abstractmethod
    async def resume_download(self, task_id: str,
                             progress_callback: Optional[Callable[[DownloadProgress], None]] = None) -> str:
        """恢复下载任务（进程重启后）"""
        pass

    @abstractmethod
    def get_model_download_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取模型的下载配置信息"""
        pass

    @abstractmethod
    async def convert_model(self, model_name: str, from_version: str, to_version: str,
                           progress_callback: Optional[Callable[[ConversionTask], None]] = None) -> str:
        """生成模型的量化版本（derived）
        
        Args:
            model_name: 模型 key
            from_version: 源版本 key（如 "fp16"）
            to_version: 目标版本 key（如 "int4"）
            progress_callback: 进度回调函数
            
        Returns:
            输出目录路径
        """
        pass

    @abstractmethod
    def list_conversions(self) -> List[ConversionTask]:
        """列出所有转换任务"""
        pass

    @abstractmethod
    async def cancel_conversion(self, task_id: str) -> bool:
        """取消转换任务"""
        pass

    @abstractmethod
    def get_conversion_progress(self, task_id: str) -> Optional[ConversionTask]:
        """获取单个转换任务的进度"""
        pass

    @abstractmethod
    async def delete_model(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """删除模型或指定版本

        Args:
            model_name: 模型名称（注册表中的 key）
            version: 可选，指定要删除的版本 key；不传则删除整个模型

        Returns:
            {"success": bool, "deleted_paths": List[str], "error": Optional[str]}
        """
        pass


@dataclass
class ModelConfig:
    """models_registry 单条模型视图；actions 为 v2 动词块（create / rewrite / …）。"""

    engine: str
    type: str
    name: Dict[str, str] = field(default_factory=dict)
    description: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    recommended: bool = False
    category: str = "base_models"
    dependencies: List[str] = field(default_factory=list)
    source: str = "huggingface"
    download_url: Optional[str] = None
    files: Optional[List[str]] = None
    versions: Optional[Dict[str, Any]] = None
    negative_prompt_support: bool = False
    base_model: Optional[str] = None
    nsfw: bool = False
    media: str = "image"
    actions: Dict[str, Any] = field(default_factory=dict)


class ISettingsService(ABC):
    """设置服务接口"""

    @abstractmethod
    def get_settings(self) -> AppSettings:
        pass

    @abstractmethod
    def update_settings(self, settings: AppSettings) -> None:
        pass

    @abstractmethod
    def get_available_models(self) -> List[ModelInfo]:
        pass

    @abstractmethod
    def refresh_models(self) -> None:
        pass

    @abstractmethod
    def install_environment(self) -> bool:
        pass

    @abstractmethod
    def check_environment(self) -> bool:
        pass

    @abstractmethod
    def get_model_registry(self) -> Dict[str, ModelConfig]:
        """获取模型注册配置"""
        pass

    @abstractmethod
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """获取单个模型配置"""
        pass


# ===== 工具接口 =====

class IPathResolver(ABC):
    """路径解析器接口"""

    @abstractmethod
    def get_models_dir(self) -> Path:
        pass

    @abstractmethod
    def get_loras_dir(self) -> Path:
        pass

    @abstractmethod
    def get_outputs_dir(self) -> Path:
        pass

    @abstractmethod
    def get_venv_python(self) -> Path:
        pass

    @abstractmethod
    def get_project_root(self) -> Path:
        pass
