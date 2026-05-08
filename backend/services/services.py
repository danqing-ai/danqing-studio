"""
服务层实现 — 设置（生成任务由 TaskScheduler + 媒体路由负责）
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from backend.core.interfaces import (
    ISettingsService,
    IConfigStore,
    IPathResolver,
    AppSettings,
    ModelInfo,
    ModelConfig,
)
from backend.core.media_interfaces import IImageEngine, IVideoEngine
from backend.utils.path_utils import get_system_info
from backend.core.registry_format import (
    media_from_record,
    parse_i18n_string_pair,
    typed_parameters,
)


class SettingsService(ISettingsService):
    """设置服务实现"""
    
    def __init__(
        self,
        config_store: IConfigStore,
        path_resolver: IPathResolver,
        image_engine: IImageEngine,
        video_engine: IVideoEngine,
    ):
        self._config = config_store
        self._path_resolver = path_resolver
        self._image_engine = image_engine
        self._video_engine = video_engine
    
    def get_settings(self) -> AppSettings:
        return self._config.load()
    
    def update_settings(self, settings: AppSettings) -> None:
        self._config.save(settings)
    
    def get_available_models(self) -> List[ModelInfo]:
        models = []
        models_dir = self._path_resolver.get_models_dir()
        registry = self.get_model_registry()
        
        if not models_dir.exists():
            return models
        
        category_type_map = {
            'base': 'diffusion', 'base_models': 'diffusion',
            'controlnet': 'controlnet', 'controlnets': 'controlnet',
            'upscaler': 'upscaler', 'upscalers': 'upscaler',
            'tool': 'tool', 'tools': 'tool',
            'lora': 'lora', 'loras': 'lora',
        }
        
        valid_extensions = [".safetensors", ".bin", ".pt"]
        model_indicators = ["model_index.json"]
        
        def is_model_file(f: Path) -> bool:
            return f.suffix in valid_extensions or f.name in model_indicators
        
        for subdir in models_dir.iterdir():
            if not subdir.is_dir() or subdir.name.startswith('.'):
                continue
            
            category_name = subdir.name.lower()
            category_type = category_type_map.get(category_name, 'unknown')
            
            # 1. 扫描子目录（第三级目录作为模型包）
            for model_dir in subdir.iterdir():
                if not model_dir.is_dir() or model_dir.name.startswith('.'):
                    continue
                has_files = any(is_model_file(f) for f in model_dir.rglob("*"))
                if has_files:
                    size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
                    model_type = category_type
                    if category_name in ['base', 'base_models'] and model_dir.name in registry:
                        model_type = registry[model_dir.name].type or category_type
                    models.append(ModelInfo(name=model_dir.name, path=str(model_dir), type=model_type, size=size))
            
            # 2. 扫描直接文件（第三级文件作为单文件模型）
            for f in subdir.iterdir():
                if f.is_file() and not f.name.startswith('.') and f.suffix in valid_extensions:
                    models.append(ModelInfo(
                        name=f.stem,
                        path=str(f),
                        type=category_type,
                        size=f.stat().st_size
                    ))
        
        return models
    
    def refresh_models(self) -> None:
        # 重新扫描
        pass
    
    def install_environment(self) -> bool:
        # TODO: 实现环境安装
        return False
    
    def check_environment(self) -> bool:
        return self._image_engine.is_available() or self._video_engine.is_available()
    
    def get_system_info(self) -> dict:
        return get_system_info()

    def get_model_registry(self) -> Dict[str, ModelConfig]:
        """读取 models_registry.json 获取所有模型配置"""
        registry_path = self._path_resolver.get_project_root() / "config" / "models_registry.json"
        if not registry_path.exists():
            return {}
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            models = {}
            for key, val in data.get("models", {}).items():
                if not isinstance(val, dict):
                    continue
                nz, ne, dz, de = parse_i18n_string_pair(
                    val.get("name"),
                    val.get("description"),
                    key,
                )
                media = media_from_record(val)
                params = typed_parameters(val.get("parameters") or {})
                acts = val.get("actions") if isinstance(val.get("actions"), dict) else {}
                neg = val.get("negative_prompt_support")
                if not isinstance(neg, bool):
                    p0 = val.get("parameters") or {}
                    neg = bool(p0.get("negative_prompt_support")) if isinstance(p0, dict) else False
                models[key] = ModelConfig(
                    name={"zh": nz, "en": ne},
                    description={"zh": dz, "en": de},
                    engine=val.get("engine", ""),
                    type=val.get("type", ""),
                    parameters=params,
                    recommended=val.get("recommended", False),
                    category=val.get("category", "base_models"),
                    dependencies=val.get("dependencies", []),
                    source=val.get("source", "huggingface"),
                    download_url=val.get("download_url"),
                    files=val.get("files"),
                    versions=val.get("versions"),
                    negative_prompt_support=neg,
                    base_model=val.get("base_model"),
                    nsfw=val.get("nsfw", False),
                    media=media,
                    actions=dict(acts),
                )
            return models
        except Exception as e:
            print(f"读取模型注册表失败: {e}")
            return {}

    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """获取单个模型配置"""
        registry = self.get_model_registry()
        return registry.get(model_name)

    def lora_adapter_picklist(self, for_model: Optional[str] = None) -> List[Dict[str, Any]]:
        """已安装 LoRA 的适配器列表；`for_model` 为图像模型 id 时按 base_model 过滤。"""
        registry = self.get_model_registry()
        installed = self.get_available_models()
        installed_names = {m.name for m in installed}
        out: List[Dict[str, Any]] = []
        for key, config in registry.items():
            if getattr(config, "category", None) != "loras":
                continue
            if key not in installed_names:
                continue
            lora_base = config.base_model or ""
            if for_model:
                model_base = for_model
                if model_base.startswith("flux2"):
                    ok = lora_base == "" or lora_base.startswith("flux2") or lora_base == model_base
                elif model_base.startswith("flux1"):
                    ok = lora_base == "" or lora_base.startswith("flux1") or lora_base == model_base
                elif model_base.startswith("flux"):
                    ok = lora_base == "" or lora_base.startswith("flux") or lora_base == model_base
                elif model_base.startswith("wan"):
                    ok = lora_base == "" or lora_base.startswith("wan") or lora_base == model_base
                else:
                    ok = lora_base == model_base
                if not ok:
                    continue
            out.append(
                {
                    "kind": "lora",
                    "id": key,
                    "name": (config.name or {}).get("zh") or (config.name or {}).get("en") or key,
                    "base_model": lora_base,
                }
            )
        return out

    def get_models_status(self) -> Dict[str, bool]:
        """获取所有注册模型的就绪状态
        
        Returns:
            模型名称 -> 是否就绪的字典
        """
        # 复用详细状态的逻辑，避免路径解析不一致
        detailed = self.get_models_detailed_status()
        return {k: v['ready'] for k, v in detailed.items()}
    
    def get_models_detailed_status(self) -> Dict[str, Dict[str, Any]]:
        """获取模型详细状态（按版本检查，区分未下载/文件缺失/已就绪）"""
        from pathlib import Path
        
        status = {}
        registry = self.get_model_registry()
        project_root = self._path_resolver.get_project_root()
        
        for model_name, config in registry.items():
            versions = getattr(config, 'versions', None) or {}
            model_status = {
                "versions": {},
                "ready": False,
                "status": "not_downloaded"
            }
            any_ready = False
            
            for version_key, version_config in versions.items():
                # 从版本配置获取 local_path
                local_path = version_config.get('local_path') if isinstance(version_config, dict) else None
                if not local_path:
                    local_path = f"models/{model_name}-{version_key}"
                
                model_dir = project_root / local_path
                
                # 如果路径不存在，尝试大小写不敏感搜索
                if not model_dir.exists():
                    models_base = self._path_resolver.get_models_dir()
                    if models_base.exists():
                        for subdir in models_base.iterdir():
                            if subdir.is_dir() and not subdir.name.startswith('.'):
                                for child in subdir.iterdir():
                                    if child.is_dir() and child.name.lower() == (model_name + '-' + version_key).lower():
                                        model_dir = child
                                        break
                                if model_dir.exists():
                                    break
                
                if not model_dir.exists():
                    model_status["versions"][version_key] = {
                        "status": "not_downloaded",
                        "label": "未下载",
                        "ready": False
                    }
                else:
                    # 路径存在，检查是否有权重文件
                    if model_dir.is_file():
                        # 如果路径本身是文件（如 LoRA 的 .safetensors 文件），直接检查
                        has_weights = (
                            model_dir.suffix in ['.safetensors', '.bin', '.pt', '.ckpt'] or
                            model_dir.name in ['model_index.json', 'config.json', 'pytorch_model.bin.index.json']
                        )
                    else:
                        # 如果是目录，递归检查
                        has_weights = any(
                            f.suffix in ['.safetensors', '.bin', '.pt', '.ckpt'] or
                            f.name in ['model_index.json', 'config.json', 'pytorch_model.bin.index.json']
                            for f in model_dir.rglob('*')
                            if f.is_file()
                        )
                    
                    if has_weights:
                        model_status["versions"][version_key] = {
                            "status": "ready",
                            "label": "已就绪",
                            "ready": True
                        }
                        any_ready = True
                    else:
                        model_status["versions"][version_key] = {
                            "status": "incomplete",
                            "label": "文件缺失",
                            "ready": False
                        }
            
            # 设置模型整体状态
            if any_ready:
                model_status["ready"] = True
                model_status["status"] = "ready"
            elif model_status["versions"]:
                # 检查是否所有版本都未下载
                all_not_downloaded = all(v["status"] == "not_downloaded" for v in model_status["versions"].values())
                if all_not_downloaded:
                    model_status["status"] = "not_downloaded"
                else:
                    model_status["status"] = "incomplete"
            
            status[model_name] = model_status
        
        return status
    
    def get_disk_space(self) -> Dict[str, Any]:
        """获取磁盘空间信息"""
        import shutil
        
        # 获取各目录
        models_dir = self._path_resolver.get_models_dir()
        loras_dir = self._path_resolver.get_loras_dir()
        outputs_dir = self._path_resolver.get_outputs_dir()
        
        # 获取磁盘使用情况
        def get_dir_info(path: Path) -> Dict[str, Any]:
            if not path.exists():
                return {"exists": False, "size": 0, "size_human": "0 B"}
            
            # 计算目录大小
            total_size = 0
            for f in path.rglob("*"):
                if f.is_file():
                    total_size += f.stat().st_size
            
            # 获取磁盘剩余空间
            usage = shutil.disk_usage(str(path))
            
            return {
                "exists": True,
                "size": total_size,
                "size_human": _human_readable_size(total_size),
                "free": usage.free,
                "free_human": _human_readable_size(usage.free),
                "total": usage.total,
                "total_human": _human_readable_size(usage.total),
                "percent_used": round((usage.total - usage.free) / usage.total * 100, 1)
            }
        
        return {
            "models": get_dir_info(models_dir),
            "loras": get_dir_info(loras_dir),
            "outputs": get_dir_info(outputs_dir),
        }


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

