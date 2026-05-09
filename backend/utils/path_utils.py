"""
路径解析器和系统工具
"""

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

from backend.core.interfaces import IPathResolver


class PathResolver(IPathResolver):
    """路径解析器实现"""
    
    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
                # PyInstaller 打包后
                exe_dir = Path(sys.executable).parent.resolve()
                # macOS .app bundle: 可执行文件在 Contents/MacOS/，数据在 Contents/Resources/
                if sys.platform == "darwin" and exe_dir.name == "MacOS" and (exe_dir.parent / "Resources").exists():
                    self._root = exe_dir.parent / "Resources"
                else:
                    self._root = exe_dir
            else:
                # 开发环境：从当前文件向上回溯到项目根目录
                self._root = Path(__file__).parent.parent.parent.parent.resolve()
        else:
            self._root = project_root
        
        # 确保目录存在
        self.get_models_dir().mkdir(parents=True, exist_ok=True)
        self.get_loras_dir().mkdir(parents=True, exist_ok=True)
        self.get_outputs_dir().mkdir(parents=True, exist_ok=True)
        (self._root / "config").mkdir(parents=True, exist_ok=True)
        (self._root / "db").mkdir(parents=True, exist_ok=True)
    
    def get_project_root(self) -> Path:
        return self._root
    
    def get_models_dir(self) -> Path:
        return self._root / "models"
    
    def get_loras_dir(self) -> Path:
        return self._root / "models" / "Lora"
    
    def get_outputs_dir(self) -> Path:
        return self._root / "outputs"
    
    def get_venv_python(self) -> Path:
        return self._root / ".venv" / "bin" / "python"
    
    def get_config_path(self) -> Path:
        return self._root / "config" / ".app_config.json"
    
    def get_presets_path(self) -> Path:
        return self._root / "config" / "presets.json"


def get_system_info() -> dict:
    """获取系统信息"""
    info = {
        "platform": platform.system(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "memory_gb": get_memory_gb(),
        "dependencies": {}
    }
    
    # 获取关键依赖版本
    deps = {
        "mlx": "mlx.core",
        "mflux": "mflux",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "Pillow": "PIL",
        "huggingface_hub": "huggingface_hub"
    }
    for dep_name, import_name in deps.items():
        try:
            import importlib
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "unknown")
            if version == "unknown" and import_name == "mlx.core":
                # Try mlx.core specifically
                import mlx.core
                version = getattr(mlx.core, "__version__", "unknown")
            info["dependencies"][dep_name] = version
        except Exception:
            info["dependencies"][dep_name] = "not installed"
        
        # Fallback to importlib.metadata if module doesn't expose __version__
        if info["dependencies"][dep_name] == "unknown":
            try:
                from importlib.metadata import version
                info["dependencies"][dep_name] = version(dep_name)
            except Exception:
                pass
    
    return info


def get_memory_gb() -> float:
    """获取系统内存大小（GB）"""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        if platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, check=True
                )
                return int(result.stdout.strip()) / (1024 ** 3)
            except:
                return 0
        return 0


def is_apple_silicon() -> bool:
    """检查是否为 Apple Silicon"""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def open_directory(path: Path) -> None:
    """打开目录"""
    if platform.system() == "Darwin":
        subprocess.run(["open", str(path)])
    elif platform.system() == "Windows":
        subprocess.run(["explorer", str(path)])
    else:
        subprocess.run(["xdg-open", str(path)])
