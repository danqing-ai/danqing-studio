"""Optional LM log-prob quality scoring (lightweight PMI-style heuristic)."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PmiAssessment:
    enabled: bool
    score: Optional[float]
    mean_logprob: Optional[float]
    note: str

    def quality_bonus(self) -> float:
        if not self.enabled or self.score is None:
            return 0.0
        if self.score >= 0.75:
            return 4.0
        if self.score >= 0.55:
            return 1.5
        if self.score < 0.35:
            return -6.0
        return 0.0


def pmi_scoring_enabled() -> bool:
    return os.environ.get("ACESTEP_PMI_SCORING", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
