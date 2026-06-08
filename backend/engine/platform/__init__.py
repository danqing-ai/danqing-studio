"""Platform session + backend detection (engine v3)."""

from backend.engine.platform.info import PlatformInfo
from backend.engine.platform.session import PlatformSession, platform_from_runtime

__all__ = ["PlatformInfo", "PlatformSession", "platform_from_runtime"]
