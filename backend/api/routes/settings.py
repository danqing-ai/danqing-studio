"""
API 路由 - 设置相关
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from backend.core.container import get_container
from backend.core.interfaces import ISettingsService, AppSettings, ModelConfig, IPresetStore
from backend.core.i18n import t, resolve_locale

router = APIRouter(prefix="/api/settings", tags=["settings"])


class ModelRegistryResponse(BaseModel):
    models: dict
    engines: dict


class SettingsResponse(BaseModel):
    language: str
    theme: str
    default_model: str
    auto_save_prompts: bool
    output_format: str
    mlx_memory_limit: int
    model_cache_ttl_minutes: int = 30
    queue_image_first: bool = False
    civitai_token: str = ""
    huggingface_token: str = ""
    nsfw_enabled: bool = False
    custom_models_dir: str
    custom_loras_dir: str
    custom_outputs_dir: str


class SettingsUpdateRequest(BaseModel):
    language: Optional[str] = None
    theme: Optional[str] = None
    default_model: Optional[str] = None
    auto_save_prompts: Optional[bool] = None
    output_format: Optional[str] = None
    mlx_memory_limit: Optional[int] = None
    model_cache_ttl_minutes: Optional[int] = None
    queue_image_first: Optional[bool] = None
    civitai_token: Optional[str] = None
    huggingface_token: Optional[str] = None
    nsfw_enabled: Optional[bool] = None
    custom_models_dir: Optional[str] = None
    custom_loras_dir: Optional[str] = None
    custom_outputs_dir: Optional[str] = None


class ModelResponse(BaseModel):
    name: str
    path: str
    type: str
    size: int
    size_human: str


class SystemInfoResponse(BaseModel):
    """GET /system：含 mlx_memory_limit（GB，与 AppSettings 一致）供前端 Plan E4 提交前软提示。"""

    platform: str
    architecture: str
    processor: str
    python_version: str
    memory_gb: float
    env_ready: bool
    dependencies: Optional[Dict[str, str]] = None
    mlx_memory_limit: int = 120


def get_settings_service():
    return get_container().resolve(ISettingsService)


@router.get("", response_model=SettingsResponse)
def get_settings():
    """获取设置"""
    service = get_settings_service()
    settings = service.get_settings()
    return SettingsResponse(**settings.__dict__)


@router.put("")
def update_settings(request: SettingsUpdateRequest):
    """更新设置"""
    service = get_settings_service()
    settings = service.get_settings()
    payload = request.model_dump(exclude_unset=True)

    for key, value in payload.items():
        if value is not None:
            setattr(settings, key, value)

    service.update_settings(settings)
    return {"success": True}


@router.get("/models", response_model=List[ModelResponse])
def list_models():
    """列出可用模型"""
    service = get_settings_service()
    models = service.get_available_models()
    return [
        ModelResponse(
            name=m.name,
            path=m.path,
            type=m.type,
            size=m.size,
            size_human=_human_readable_size(m.size)
        )
        for m in models
    ]


@router.get("/loras", response_model=List[ModelResponse])
def list_loras():
    """列出可用的 LoRA 模型"""
    service = get_settings_service()
    models = service.get_available_models()
    loras = [m for m in models if m.type == "lora"]
    return [
        ModelResponse(
            name=m.name,
            path=m.path,
            type=m.type,
            size=m.size,
            size_human=_human_readable_size(m.size)
        )
        for m in loras
    ]


@router.post("/refresh")
def refresh_models():
    """刷新模型列表"""
    service = get_settings_service()
    service.refresh_models()
    return {"success": True}


@router.get("/system", response_model=SystemInfoResponse)
def get_system_info():
    """获取系统信息"""
    service = get_settings_service()
    info = service.get_system_info()
    info["env_ready"] = service.check_environment()
    settings = service.get_settings()
    info["mlx_memory_limit"] = int(getattr(settings, "mlx_memory_limit", 120) or 120)
    return SystemInfoResponse(**info)


@router.post("/install")
def install_environment():
    """安装环境"""
    service = get_settings_service()
    success = service.install_environment()
    return {"success": success}


@router.get("/registry")
def get_model_registry():
    """获取模型注册表"""
    service = get_settings_service()
    registry = service.get_model_registry()
    # 同时返回每个模型的就绪状态
    status = service.get_models_status()
    return {
        "models": {
            key: {
                "name": val.name,
                "description": val.description,
                "engine": val.engine,
                "type": val.type,
                "category": val.category,
                "media": getattr(val, "media", "image"),
                "actions": getattr(val, "actions", None) or {},
                "parameters": val.parameters,
                "recommended": val.recommended,
                "dependencies": val.dependencies,
                "base_model": val.base_model,
                "nsfw": val.nsfw,
                "ready": status.get(key, False),
                "source": val.source,
                "versions": val.versions,
            }
            for key, val in registry.items()
        }
    }


@router.get("/models/status")
def get_models_status():
    """获取所有模型的就绪状态（简化版）"""
    service = get_settings_service()
    status = service.get_models_status()
    return status


@router.get("/models/status/detailed")
def get_models_detailed_status():
    """获取模型详细状态（区分未下载/文件缺失/已就绪）"""
    service = get_settings_service()
    return service.get_models_detailed_status()


@router.get("/loras/compatible/{model_name}")
def get_compatible_loras(model_name: str):
    """获取与指定模型兼容的 LoRA（从注册表读取 base_model 字段）"""
    service = get_settings_service()
    rows = service.lora_adapter_picklist(model_name)
    return [{"name": r["name"], "path": r["id"], "base_model": r.get("base_model", "")} for r in rows]


@router.get("/controlnets/compatible/{model_name}")
def get_compatible_controlnets(model_name: str):
    """获取与指定模型兼容的 ControlNet（从注册表读取 base_model 字段）"""
    service = get_settings_service()
    registry = service.get_model_registry()
    detailed_status = service.get_models_detailed_status()

    results = []
    for key, config in registry.items():
        if config.category != "controlnets":
            continue
        if config.type not in ("controlnet",):
            continue

        net_base = config.base_model or ""
        # 匹配逻辑：FLUX 模型兼容所有 FLUX ControlNet
        if model_name.startswith("flux"):
            is_compatible = net_base == "" or net_base.startswith("flux")
        else:
            is_compatible = net_base == model_name

        if not is_compatible:
            continue

        # 检查就绪状态
        status_info = detailed_status.get(key, {})
        ready = status_info.get("ready", False)
        versions_ready = {}
        for vk, vs in status_info.get("versions", {}).items():
            versions_ready[vk] = vs.get("ready", False)

        results.append({
            "name": config.name or key,
            "name_en": getattr(config, "name_en", None) or config.name or key,
            "key": key,
            "base_model": net_base,
            "ready": ready,
            "versions_ready": versions_ready,
            "actions": getattr(config, "actions", None) or {},
            "parameters": config.parameters,
        })

    return results


@router.get("/disk-space")
def get_disk_space():
    """获取磁盘空间使用情况"""
    service = get_settings_service()
    space = service.get_disk_space()
    return space


@router.post("/models/{model_name}/parameters")
def update_model_parameters(model_name: str, request: dict, req: Request = None):
    locale = resolve_locale(req.headers.get("Accept-Language")) if req else "zh"
    import json
    import shutil
    from datetime import datetime
    from backend.core.container import get_container
    from backend.core.interfaces import IPathResolver
    
    path_resolver = get_container().resolve(IPathResolver)
    registry_path = path_resolver.get_project_root() / "config" / "models_registry.json"
    
    if not registry_path.exists():
        return {"success": False, "error": t("error.registry_not_found", locale)}
    
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
        
        if model_name not in registry.get("models", {}):
            return {"success": False, "error": t("error.model_not_found", locale, name=model_name)}
        
        # 备份原文件（加上时间戳）
        backup_path = registry_path.parent / f"models_registry.json.backup.{datetime.now().strftime('%Y%m%d%H%M%S')}"
        shutil.copy2(registry_path, backup_path)
        
        # 更新参数
        model = registry["models"][model_name]
        if "parameters" not in model:
            model["parameters"] = {}
        
        # Plan C3：凡注册表里带 default 的可写标量/枚举，均允许从设置页更新（与 RegistryParamsForm 对齐）
        for key, value in request.items():
            spec = model["parameters"].get(key)
            if not isinstance(spec, dict) or "default" not in spec:
                continue
            if spec.get("type") == "bool" and str(key).endswith("_support"):
                continue
            if isinstance(value, dict) and "default" in value:
                spec["default"] = value["default"]
            else:
                spec["default"] = value

        # 写回文件
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)
        
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/system/monitor")
def get_system_monitor():
    """获取实时系统资源监控（CPU、内存、GPU）"""
    import platform
    import subprocess
    
    data = {
        "cpu_percent": 0,
        "memory": {
            "total_gb": 0,
            "used_gb": 0,
            "percent": 0
        },
        "gpu": None
    }
    
    try:
        import psutil
        # CPU 使用率
        data["cpu_percent"] = psutil.cpu_percent(interval=0.5)
        
        # 内存信息
        mem = psutil.virtual_memory()
        data["memory"] = {
            "total_gb": round(mem.total / (1024**3), 1),
            "used_gb": round(mem.used / (1024**3), 1),
            "percent": mem.percent
        }
    except ImportError:
        # psutil 未安装，尝试用系统命令
        try:
            if platform.system() == "Darwin":
                # CPU 使用率
                result = subprocess.run(
                    ["top", "-l", "1", "-n", "0"],
                    capture_output=True, text=True, timeout=3
                )
                for line in result.stdout.split("\n"):
                    if "CPU usage" in line:
                        # 提取用户+系统 CPU 使用率
                        parts = line.split(":")[-1].strip().split(",")
                        user = float(parts[0].strip().replace("% user", ""))
                        sys = float(parts[1].strip().replace("% sys", ""))
                        data["cpu_percent"] = round(user + sys, 1)
                        break
                
                # 内存信息
                result = subprocess.run(
                    ["vm_stat"],
                    capture_output=True, text=True, timeout=3
                )
                page_size = 4096
                mem_stats = {}
                for line in result.stdout.split("\n"):
                    if "page size of" in line:
                        # Extract page size from "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
                        import re
                        match = re.search(r'page size of (\d+)', line)
                        if match:
                            page_size = int(match.group(1))
                    elif ":" in line and "Pages" in line:
                        key, val = line.split(":")
                        try:
                            mem_stats[key.strip()] = int(val.strip().replace(".", ""))
                        except ValueError:
                            pass
                
                if mem_stats:
                    used_pages = mem_stats.get("Pages active", 0) + mem_stats.get("Pages wired down", 0)
                    free_pages = mem_stats.get("Pages free", 0)
                    inactive_pages = mem_stats.get("Pages inactive", 0)
                    total_pages = free_pages + used_pages + inactive_pages
                    if total_pages > 0:
                        data["memory"]["total_gb"] = round(total_pages * page_size / (1024**3), 1)
                        data["memory"]["used_gb"] = round(used_pages * page_size / (1024**3), 1)
                        data["memory"]["percent"] = round(used_pages / total_pages * 100, 1)
        except Exception:
            pass
    except Exception:
        pass
    
    # GPU 信息
    try:
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            # 尝试获取 Apple Silicon 芯片型号
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, timeout=2
                )
                chip_model = result.stdout.strip()
                # 提取芯片型号，如 "Apple M1 Max" -> "M1 Max"
                if "Apple" in chip_model:
                    chip_model = chip_model.replace("Apple ", "")
            except Exception:
                chip_model = "Apple Silicon"
            
            data["gpu"] = {
                "available": True,
                "model": chip_model,
                "note": "Unified Memory Architecture",
                "memory_gb": data["memory"]["total_gb"],
                "type": "integrated"
            }
    except Exception:
        pass
    
    return data


@router.get("/presets")
def get_presets():
    """获取所有提示词模板"""
    preset_store = get_container().resolve(IPresetStore)
    return preset_store.load_all()


class PresetCreateRequest(BaseModel):
    name: str
    preset: dict


_PRESET_MEDIA_SCOPES = frozenset({"image", "video"})


def _validate_preset_plan_v2(preset: dict) -> None:
    """Plan G：预设必须含非空 applies_to 与显式 media_scope（无运行时默认）。"""
    if not isinstance(preset, dict):
        raise HTTPException(status_code=400, detail="preset must be an object")
    app = preset.get("applies_to")
    if not isinstance(app, list) or len(app) == 0:
        raise HTTPException(status_code=400, detail="preset.applies_to must be a non-empty array")
    ms = preset.get("media_scope")
    if ms not in _PRESET_MEDIA_SCOPES:
        raise HTTPException(
            status_code=400,
            detail="preset.media_scope must be image or video",
        )


@router.post("/presets")
def create_preset(request: PresetCreateRequest):
    """创建或更新提示词模板"""
    _validate_preset_plan_v2(request.preset)
    preset_store = get_container().resolve(IPresetStore)
    preset_store.save(request.name, request.preset)
    return {"success": True}


@router.delete("/presets/{name}")
def delete_preset(name: str):
    """删除提示词模板"""
    preset_store = get_container().resolve(IPresetStore)
    preset_store.delete(name)
    return {"success": True}


def _human_readable_size(size_bytes: int) -> str:
    """转换为人类可读的大小"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"
