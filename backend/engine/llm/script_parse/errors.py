"""Script parse pipeline errors."""
from __future__ import annotations


class ScriptParseQualityError(RuntimeError):
    """Expand failed parse quality gate (critical issues)."""

    def __init__(self, message: str, *, quality_issues: list[dict] | None = None) -> None:
        super().__init__(message)
        self.quality_issues = list(quality_issues or [])
