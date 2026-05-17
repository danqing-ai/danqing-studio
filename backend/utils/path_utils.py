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
from backend.utils.workspace import (
    prepare_data_directories,
    seed_workspace_from_bootstrap,
)


class PathResolver(IPathResolver):
    """路径解析器实现"""
    
    def __init__(self, project_root: Optional[Path] = None):
        bundle_root: Path | None = None
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            bundle_root = Path(sys._MEIPASS).resolve()
            exe_dir = Path(sys.executable).parent.resolve()
            if project_root is not None:
                bootstrap = Path(project_root).resolve()
            else:
                raw_ws = os.environ.get("DANQING_USER_DATA_DIR", "").strip()
                if raw_ws:
                    bootstrap = Path(raw_ws).expanduser().resolve()
                elif (
                    sys.platform == "darwin"
                    and exe_dir.name == "MacOS"
                    and (exe_dir.parent / "Resources").exists()
                ):
                    bootstrap = exe_dir.parent / "Resources"
                else:
                    bootstrap = exe_dir
        elif project_root is None:
            bootstrap = Path(__file__).parent.parent.parent.parent.resolve()
        else:
            bootstrap = Path(project_root).resolve()

        self._bundle_root = bundle_root
        self._bootstrap = bootstrap
        self._root = prepare_data_directories(self._bootstrap)
        if bundle_root is not None:
            seed_workspace_from_bootstrap(bundle_root, self._root)
    
    def get_bootstrap_root(self) -> Path:
        return self._bootstrap

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

    def resolve_registry_local_path(self, local_path: str) -> Path:
        """Resolve registry paths against the effective workspace (never the repo bootstrap root)."""
        text = (local_path or "").strip()
        if not text:
            raise ValueError("local_path is required")
        candidate = Path(text).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        if text.startswith("models/"):
            return (self.get_models_dir() / text[len("models/") :]).resolve()
        return (self.get_project_root() / text).resolve()

    def remap_legacy_data_path(self, path: str | Path) -> Path:
        """Rewrite paths that still point at bootstrap ``models/`` / ``outputs/`` / ``db/`` after workspace migration."""
        raw = Path(path).expanduser()
        if not raw.is_absolute():
            raw = (self.get_bootstrap_root() / raw).resolve()
        else:
            raw = raw.resolve()
        workspace = self.get_project_root().resolve()
        bootstrap = self.get_bootstrap_root().resolve()
        if workspace == bootstrap:
            return raw
        try:
            rel = raw.relative_to(bootstrap)
        except ValueError:
            return raw
        if rel.parts and rel.parts[0] in ("models", "outputs", "db"):
            return (workspace / rel).resolve()
        return raw


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


