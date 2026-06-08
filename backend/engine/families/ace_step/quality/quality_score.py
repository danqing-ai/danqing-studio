"""Lightweight ACE-Step output quality heuristics (no upstream LM PMI dependency)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class AceStepQualityAssessment:
    score: float
    grade: str
    warnings: tuple[str, ...]

    def as_metadata(self) -> dict[str, Any]:
        return {
            "quality_score": round(self.score, 1),
            "quality_grade": self.grade,
            "quality_warnings": list(self.warnings),
        }


def assess_generation_quality(
    *,
    hum_ratio: float = 0.0,
    mains_acf: float = 0.0,
    latent_cos: float = 0.0,
    latent_diff: float = 0.0,
    lm_expanded: bool = False,
    near_silence: bool = False,
    pmi_bonus: float = 0.0,
) -> AceStepQualityAssessment:
    """Score 0–100 from signals already collected during inference."""
    score = 78.0
    warnings: list[str] = []

    if near_silence or (latent_cos >= 0.95 and latent_diff < 0.10):
        score -= 45.0
        warnings.append("latent_near_silence")
    if hum_ratio > 0.35:
        score -= 18.0
        warnings.append("high_hum_ratio")
    elif hum_ratio > 0.22:
        score -= 8.0
        warnings.append("moderate_hum_ratio")
    if mains_acf > 0.45:
        score -= 12.0
        warnings.append("mains_hum")
    elif mains_acf > 0.32:
        score -= 5.0
    if lm_expanded:
        score += 2.0
    if pmi_bonus:
        score += pmi_bonus
        if pmi_bonus < 0:
            warnings.append("low_lm_pmi")

    score = max(0.0, min(100.0, score))
    if score >= 72.0:
        grade = "good"
    elif score >= 55.0:
        grade = "fair"
    else:
        grade = "poor"
    return AceStepQualityAssessment(score=score, grade=grade, warnings=tuple(warnings))


def quality_log_message(assessment: AceStepQualityAssessment) -> Optional[str]:
    if assessment.grade == "good" and not assessment.warnings:
        return None
    warn = ", ".join(assessment.warnings) if assessment.warnings else "none"
    return (
        f"生成质量评估: {assessment.score:.0f}/100 ({assessment.grade}); "
        f"signals={warn}"
    )
