"""L3 model contracts — ``TransformerBase`` and DiT dispatch stem."""

from backend.engine.common.model.base import TransformerBase
from backend.engine.common.model.dit_stem import DelegatingDiTStem, dispatch_dit_implementation

__all__ = ["TransformerBase", "DelegatingDiTStem", "dispatch_dit_implementation"]
