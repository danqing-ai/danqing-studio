"""SeedVR2 文本侧：仅固定正嵌入，走与注册表 ``encoder_type`` 无关的独立路径。"""

from backend.engine.seedvr2.text_encoders.positive import SeedVR2PositiveEmbeddings

__all__ = ["SeedVR2PositiveEmbeddings"]
