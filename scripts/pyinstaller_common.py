"""
Shared PyInstaller metadata for DanQing Studio desktop sidecar.

Profiles:
  mlx  — Apple Silicon / MLX inference only: no ``*_cuda`` modules, no PyTorch/CUDA stack.
  full — include CUDA hidden imports (Linux/Windows CUDA builds).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from out_paths import FRONTEND_DIST, PROJECT_ROOT

# Hidden imports that pull PyTorch / CUDA-only modules (full profile only).
_CUDA_HIDDEN_IMPORTS: tuple[str, ...] = (
    "backend.engine.families.z_image.text_encoder_cuda",
    "backend.engine.common.text_encoders.clip_cuda",
    "backend.engine.common.text_encoders.t5_cuda",
    "backend.engine.common.text_encoders.qwen25vl_cuda",
)

# Runtime modules that must not appear in mlx profile bundles.
_MLX_EXCLUDED_MODULES: tuple[str, ...] = (
    # PyTorch / CUDA GPU stack
    "torch",
    "torchvision",
    "torchaudio",
    "torchgen",
    "functorch",
    "triton",
    # DanQing CUDA backends
    "backend.engine.runtime.cuda",
    "backend.engine.families.z_image.text_encoder_cuda",
    "backend.engine.common.text_encoders.t5_cuda",
    "backend.engine.common.text_encoders.clip_cuda",
    "backend.engine.common.text_encoders.qwen25vl_cuda",
    "backend.engine.families.ace_step.transformer_cuda",
    "backend.engine.families.ace_step.vae_cuda",
    # Common venv bloat not used by MLX inference paths
    "cv2",
    "opencv_python",
    "pyarrow",
    "datasets",
    "pandas",
    "matplotlib",
    "scipy",
    "sklearn",
    "accelerate",
    "bitsandbytes",
    "tensorboard",
    "tensorboard_data_server",
    "torch.utils.tensorboard",
    "hf_xet",
    "soundfile",
)

_MLX_CORE_HIDDEN_IMPORTS: tuple[str, ...] = (
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
    "backend.engine.danqing_image_engine",
    "backend.engine.danqing_video_engine",
    "backend.engine.danqing_audio_engine",
    "backend.engine.pipelines",
    "backend.engine.pipelines.image_pipeline",
    "backend.engine.pipelines.image_upscale_pipeline",
    "backend.engine.pipelines.video_pipeline",
    "backend.engine.pipelines.video_upscale_pipeline",
    "backend.engine.pipelines.music_pipeline",
    "backend.engine.families.ace_step",
    "backend.engine.families.heartmula",
    "backend.engine.families.heartmula.install_hook",
    "backend.engine.families.heartmula.generation_mlx",
    "backend.engine.families.heartmula.weights_mlx",
    "backend.core.install_hooks",
    "backend.engine.families.heartmula.mlx",
    "backend.engine.common.safetensors_affine_quant",
    "backend.engine._transformer_registry",
    "backend.engine.families",
    "backend.engine.families.fibo",
    "backend.engine.families.flux1",
    "backend.engine.families.flux2",
    "backend.engine.families.qwen",
    "backend.engine.families.z_image",
    "backend.engine.families.seedvr2",
    "backend.engine.families.seedvr2.video_restore_mlx",
    "backend.engine.families.ltx",
    "backend.engine.families.wan",
    "backend.engine.families.wan.transformer_mlx",
    "backend.engine.families.wan.vae_mlx",
    "backend.engine.families.wan.vae",
    "backend.engine.families.wan.conditioning",
    "backend.engine.families.wan.attention_mlx",
    "backend.engine.families.wan.text_encoder_mlx",
    "backend.engine.families.cogvideox",
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
    "mlx_lm",
)


def packaging_profile() -> str:
    """``mlx`` (default on macOS) or ``full``."""
    raw = os.environ.get("DANQING_PYINSTALLER_PROFILE", "").strip().lower()
    if raw in ("mlx", "full"):
        return raw
    if sys.platform == "darwin":
        return "mlx"
    return "full"


def get_hidden_imports(profile: str | None = None) -> list[str]:
    profile = profile or packaging_profile()
    imports = list(_MLX_CORE_HIDDEN_IMPORTS)
    if profile == "full":
        imports.extend(_CUDA_HIDDEN_IMPORTS)
    return imports


def get_exclude_modules(profile: str | None = None) -> list[str]:
    profile = profile or packaging_profile()
    if profile == "full":
        return [
            "tensorboard",
            "tensorboard_data_server",
            "torch.utils.tensorboard",
        ]
    return list(_MLX_EXCLUDED_MODULES)


def get_data_files(project_root: Path | None = None, *, profile: str | None = None) -> list[str]:
    profile = profile or packaging_profile()
    root = project_root or PROJECT_ROOT
    data: list[str] = []
    separator = ";" if sys.platform == "win32" else ":"

    frontend_dist = FRONTEND_DIST
    if not frontend_dist.is_dir() or not any(frontend_dist.iterdir()):
        raise SystemExit(
            "out/frontend/dist is missing or empty. Build the UI first:\n"
            "  make frontend-build   # or: cd frontend && npm run build"
        )
    data.append(f"{frontend_dist}{separator}frontend/dist")

    default_cfg = root / "default_config"
    if default_cfg.is_dir():
        data.append(f"{default_cfg}{separator}default_config")

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
                    for pattern in ("*.dylib", "*.metallib"):
                        for lib_file in mlx_lib.glob(pattern):
                            binaries.append(f"{lib_file}{separator}mlx/lib")
                    break

    return binaries


def ensure_runtime_hook_file(project_root: Path) -> Path:
    """Write PyInstaller runtime hook; returns path to hook file."""
    hook_file = project_root / "scripts" / "pyinstaller_runtime_hook.py"
    hook_content = r"""
import os
import sys
from pathlib import Path

# PyInstaller: writable dirs + MLX metallib next to bundled dylibs.
if getattr(sys, "frozen", False):
    raw = os.environ.get("DANQING_USER_DATA_DIR")
    if raw:
        app_dir = Path(raw).expanduser().resolve()
    else:
        app_dir = Path(sys.executable).parent.resolve()
    for dir_name in ("models", "outputs", "db", "config"):
        (app_dir / dir_name).mkdir(parents=True, exist_ok=True)

    # MLX metallib is copied next to the executable by scripts/prune_sidecar.layout_mlx_runtime.
"""
    hook_file.parent.mkdir(parents=True, exist_ok=True)
    hook_file.write_text(hook_content.strip() + "\n", encoding="utf-8")
    return hook_file


def get_runtime_hooks(project_root: Path) -> list[str]:
    return [str(ensure_runtime_hook_file(project_root))]


def apply_pyinstaller_packaging_filters() -> None:
    """Only affects the PyInstaller parent process (not the frozen app)."""
    import logging
    import warnings

    os.environ.setdefault("DANQING_PYINSTALLER_PROFILE", packaging_profile())

    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module=r"PyInstaller\.utils\.hooks",
    )
    for name in (
        "torch.distributed.elastic",
        "torch.distributed.elastic.multiprocessing",
        "torch.distributed.elastic.multiprocessing.redirects",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


def pyinstaller_hooks_dir(project_root: Path) -> Path:
    return project_root / "scripts" / "pyinstaller_hooks"
