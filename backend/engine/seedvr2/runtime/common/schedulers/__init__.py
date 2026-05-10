"""SeedVR2 内部推理仅注册单步 Euler（Studio 超分路径）。"""

from .seedvr2_euler_scheduler import SeedVR2EulerScheduler

__all__ = ["SCHEDULER_REGISTRY", "SeedVR2EulerScheduler", "try_import_external_scheduler"]

SCHEDULER_REGISTRY: dict[str, type] = {
    "seedvr2_euler": SeedVR2EulerScheduler,
    "SeedVR2EulerScheduler": SeedVR2EulerScheduler,
}


def try_import_external_scheduler(scheduler_object_path: str):
    raise RuntimeError(
        f"External scheduler {scheduler_object_path!r} is not supported for SeedVR2 in DanQing."
    )
