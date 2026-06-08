"""TwoStageInference — LM→DiT 两阶段编排 (L2).

ACE-Step ``generate_waveform`` 在启用 5Hz LM 时经本模块编排扩写 + DiT 扩散。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from backend.engine.inference._runtime import raise_if_cancelled


@dataclass
class TwoStageBundle:
    """两阶段编排专用 bundle。"""

    stage1_fn: Callable[[], Any]
    stage2_fn: Callable[[Any], Any]
    cancel_token: Any | None = None


class TwoStageInference:
    """LM 扩写 → DiT 扩散 两阶段编排（ACE-Step ``generate_waveform``）。"""

    def run(self, bundle: TwoStageBundle) -> dict[str, Any]:
        stage1_result = bundle.stage1_fn()
        raise_if_cancelled(bundle.cancel_token)
        stage2_result = bundle.stage2_fn(stage1_result)
        return {
            "latents": stage2_result,
            "stage1": stage1_result,
        }
