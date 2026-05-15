"""
Shared PyInstaller metadata for DanQing Studio (desktop sidecar + legacy bundle).
"""

from __future__ import annotations

import sys
from pathlib import Path


def get_hidden_imports() -> list[str]:
    """Collect hidden imports for PyInstaller."""
    return [
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.loops.auto",
        "uvicorn.logging",
        "fastapi.middleware.cors",
        "fastapi.staticfiles",
        "backend.api.routes.adapters",
        "backend.api.routes.assets",
        "backend.api.routes.audios",
        "backend.api.routes.download",
        "backend.api.routes.gallery",
        "backend.api.routes.images",
        "backend.api.routes.models",
        "backend.api.routes.presets",
        "backend.api.routes.queue",
        "backend.api.routes.registry",
        "backend.api.routes.settings",
        "backend.api.routes.system",
        "backend.api.routes.tasks",
        "backend.api.routes.videos",
        "backend.core.container",
        "backend.core.i18n",
        "backend.core.interfaces",
        "backend.core.contracts",
        "backend.core.asset_interfaces",
        "backend.core.media_interfaces",
        "backend.core.model_registry",
        "backend.core.registry_format",
        "backend.core.task_kinds",
        "backend.engine.engine_registry",
        "backend.engine.base",
        "backend.engine.mlx_runtime",
        "backend.engine.model_cache",
        "backend.engine.danqing_image_engine",
        "backend.engine.danqing_video_engine",
        "backend.engine.danqing_audio_engine",
        "backend.engine.pipelines",
        "backend.engine.pipelines.image_pipeline",
        "backend.engine.pipelines.image_upscale_pipeline",
        "backend.engine.pipelines.video_pipeline",
        "backend.engine.pipelines.video_upscale_pipeline",
        "backend.engine.common.safetensors_affine_quant",
        "backend.engine._transformer_registry",
        "backend.engine.families",
        "backend.engine.families.fibo",
        "backend.engine.families.flux1",
        "backend.engine.families.flux2",
        "backend.engine.families.qwen",
        "backend.engine.families.z_image",
        "backend.engine.families.z_image.text_encoder_cuda",
        "backend.engine.families.seedvr2",
        "backend.engine.families.seedvr2.video_restore_mlx",
        "backend.engine.families.ltx",
        "backend.engine.families.wan",
        "backend.engine.families.cogvideox",
        "backend.engine.common.text_encoders.clip_cuda",
        "backend.engine.common.text_encoders.t5_cuda",
        "backend.engine.common.text_encoders.qwen25vl_cuda",
        "backend.services.services",
        "backend.services.download_service",
        "backend.persistence.stores",
        "backend.persistence.asset_store",
        "backend.persistence.v3_task_store",
        "backend.scheduler.task_scheduler",
        "backend.utils.path_utils",
        "backend.utils.video_sr_ffmpeg",
        "PIL",
        "PIL._imagingtk",
        "PIL._tkinter_finder",
        "psutil",
        "aiohttp",
        "python_multipart",
        "pydantic",
        "huggingface_hub",
        "safetensors",
        "tqdm",
        "requests",
        "mlx",
        "mlx.core",
        "mlx._reprlib_fix",
    ]


def get_data_files(project_root: Path) -> list[str]:
    data: list[str] = []
    separator = ";" if sys.platform == "win32" else ":"

    frontend_dir = project_root / "frontend"
    if frontend_dir.exists():
        data.append(f"{frontend_dir}{separator}frontend")

    config_dir = project_root / "config"
    if config_dir.exists():
        locales_dir = config_dir / "locales"
        if locales_dir.exists():
            data.append(f"{locales_dir}{separator}config/locales")

        registry_file = config_dir / "models_registry.json"
        if registry_file.exists():
            data.append(f"{registry_file}{separator}config")

        presets_file = config_dir / "presets.json"
        if presets_file.exists():
            data.append(f"{presets_file}{separator}config")

    return data


def get_binary_files(project_root: Path) -> list[str]:
    binaries: list[str] = []
    separator = ";" if sys.platform == "win32" else ":"

    if sys.platform == "darwin":
        venv_lib = project_root / ".venv" / "lib"
        if venv_lib.exists():
            for site in venv_lib.glob("python3.*/site-packages"):
                mlx_lib = site / "mlx" / "lib"
                if mlx_lib.exists():
                    for dylib in mlx_lib.glob("*.dylib"):
                        binaries.append(f"{dylib}{separator}mlx/lib")
                    break

    return binaries


def ensure_runtime_hook_file(project_root: Path) -> Path:
    """Write PyInstaller runtime hook; returns path to hook file."""
    hook_file = project_root / "scripts" / "pyinstaller_runtime_hook.py"
    hook_content = r"""
import os
import sys
from pathlib import Path

# PyInstaller: ensure writable dirs exist (Tauri sets DANQING_USER_DATA_DIR).
if getattr(sys, "frozen", False):
    raw = os.environ.get("DANQING_USER_DATA_DIR")
    if raw:
        app_dir = Path(raw).expanduser().resolve()
    else:
        app_dir = Path(sys.executable).parent.resolve()
    for dir_name in ("models", "outputs", "db", "config"):
        (app_dir / dir_name).mkdir(parents=True, exist_ok=True)
"""
    hook_file.parent.mkdir(parents=True, exist_ok=True)
    hook_file.write_text(hook_content.strip() + "\n", encoding="utf-8")
    return hook_file


def get_runtime_hooks(project_root: Path) -> list[str]:
    return [str(ensure_runtime_hook_file(project_root))]


def get_exclude_modules() -> list[str]:
    """Training / viz extras not required for inference; shrinks analysis noise."""
    return [
        "tensorboard",
        "tensorboard_data_server",
        "torch.utils.tensorboard",
    ]


def apply_pyinstaller_packaging_filters() -> None:
    """Only affects the PyInstaller parent process (not the frozen app)."""
    import logging
    import warnings

    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module=r"PyInstaller\.utils\.hooks",
    )
    # torch.distributed.elastic: "Redirects are currently not supported in Windows or MacOs"
    for name in (
        "torch.distributed.elastic",
        "torch.distributed.elastic.multiprocessing",
        "torch.distributed.elastic.multiprocessing.redirects",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


def pyinstaller_hooks_dir(project_root: Path) -> Path:
    return project_root / "scripts" / "pyinstaller_hooks"
