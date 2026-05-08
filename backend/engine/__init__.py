# 引擎层 — 延迟导入避免循环依赖
def get_engine_registry():
    from backend.engine.engine_registry import EngineRegistry
    return EngineRegistry

def get_danqing_image_engine():
    from backend.engine.danqing_image_engine import DanQingImageEngine
    return DanQingImageEngine

def get_danqing_video_engine():
    from backend.engine.danqing_video_engine import DanQingVideoEngine
    return DanQingVideoEngine

__all__ = ["get_engine_registry", "get_danqing_image_engine", "get_danqing_video_engine"]
