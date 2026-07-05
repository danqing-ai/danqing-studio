"""Orchestrator for script_parse 4-pass pipeline."""
from __future__ import annotations

from typing import Any, Callable

from backend.engine.llm.script_parse.beat_plan import run_beat_plan
from backend.engine.llm.script_parse.decompose import run_script_decompose
from backend.engine.llm.script_parse.finalize import run_finalize
from backend.engine.llm.script_parse.schemas import DecomposeResult, ExpandResult, ScriptArtifact
from backend.engine.llm.script_parse.shot_spec import run_shot_spec

ProgressFn = Callable[[str, str], None]


def merge_beat_shots(
    existing: list[dict[str, Any]],
    new_shots: list[dict[str, Any]],
    beat_index: int,
) -> list[dict[str, Any]]:
    """Replace shots for one narrative beat and re-order."""
    kept = [s for s in existing if int(s.get("narrative_beat_index", -1)) != int(beat_index)]
    merged = kept + [dict(s) for s in new_shots]
    merged.sort(
        key=lambda s: (
            int(s.get("narrative_beat_index", 0)),
            int(s.get("order", 0)),
            str(s.get("id", "")),
        )
    )
    for i, row in enumerate(merged):
        row["order"] = i
    return merged


def _run_expand_core(
    *,
    script: ScriptArtifact,
    target_duration_sec: float,
    segment_duration_sec: float,
    max_clip_sec: float,
    locale: str,
    chat_fn: Callable[..., Any],
    think_apply: Callable[[str], str],
    token_budget: Callable[[int], int],
    on_progress: ProgressFn | None = None,
    beat_indices: list[int] | None = None,
) -> ExpandResult:
    phases: list[str] = []
    llm_calls = 0

    beat_plan, n = run_beat_plan(
        script=script,
        target_duration_sec=target_duration_sec,
        segment_duration_sec=segment_duration_sec,
        max_clip_sec=max_clip_sec,
        locale=locale,
        chat_fn=chat_fn,
        think_apply=think_apply,
        token_budget=token_budget,
        on_progress=on_progress,
        beat_indices=beat_indices,
    )
    llm_calls += n
    phases.append("beat_plan")

    specs, n = run_shot_spec(
        script=script,
        beat_plan=beat_plan,
        locale=locale,
        chat_fn=chat_fn,
        think_apply=think_apply,
        token_budget=token_budget,
        on_progress=on_progress,
        beat_indices=beat_indices,
    )
    llm_calls += n
    phases.append("shot_spec")

    result = run_finalize(
        script=script,
        beat_plan=beat_plan,
        specs=specs,
        target_duration_sec=target_duration_sec,
        max_clip_sec=max_clip_sec,
        locale=locale,
        chat_fn=chat_fn,
        think_apply=think_apply,
        token_budget=token_budget,
        on_progress=on_progress,
    )
    llm_calls += result.llm_calls
    phases.extend(result.phases)
    return ExpandResult(
        shots=result.shots,
        characters=result.characters,
        scenes=result.scenes,
        character_anchor=result.character_anchor,
        style_anchor=result.style_anchor,
        llm_calls=llm_calls,
        phases=phases,
        validation_warnings=result.validation_warnings,
        quality_issues=result.quality_issues,
    )


def run_decompose(
    *,
    script_text: str,
    title: str = "",
    locale: str = "zh",
    chat_fn: Callable[..., Any],
    think_apply: Callable[[str], str],
    token_budget: Callable[[int], int],
    on_progress: ProgressFn | None = None,
) -> DecomposeResult:
    return run_script_decompose(
        script_text=script_text,
        title=title,
        locale=locale,
        chat_fn=chat_fn,
        think_apply=think_apply,
        token_budget=token_budget,
        on_progress=on_progress,
    )


def run_expand(
    *,
    script: ScriptArtifact,
    target_duration_sec: float = 60.0,
    segment_duration_sec: float = 5.0,
    max_clip_sec: float = 10.0,
    locale: str = "zh",
    chat_fn: Callable[..., Any],
    think_apply: Callable[[str], str],
    token_budget: Callable[[int], int],
    on_progress: ProgressFn | None = None,
    beat_indices: list[int] | None = None,
) -> ExpandResult:
    indices = [int(i) for i in (beat_indices or []) if int(i) >= 0] or None
    return _run_expand_core(
        script=script,
        target_duration_sec=target_duration_sec,
        segment_duration_sec=segment_duration_sec,
        max_clip_sec=max_clip_sec,
        locale=locale,
        chat_fn=chat_fn,
        think_apply=think_apply,
        token_budget=token_budget,
        on_progress=on_progress,
        beat_indices=indices,
    )


def run_expand_beat(
    *,
    script: ScriptArtifact,
    beat_index: int,
    target_duration_sec: float = 60.0,
    segment_duration_sec: float = 5.0,
    max_clip_sec: float = 10.0,
    locale: str = "zh",
    chat_fn: Callable[..., Any],
    think_apply: Callable[[str], str],
    token_budget: Callable[[int], int],
    on_progress: ProgressFn | None = None,
    existing_shots: list[dict[str, Any]] | None = None,
) -> ExpandResult:
    known = {b.index for b in script.beats}
    if int(beat_index) not in known:
        raise ValueError(f"unknown beat_index {beat_index}")

    partial = _run_expand_core(
        script=script,
        target_duration_sec=target_duration_sec,
        segment_duration_sec=segment_duration_sec,
        max_clip_sec=max_clip_sec,
        locale=locale,
        chat_fn=chat_fn,
        think_apply=think_apply,
        token_budget=token_budget,
        on_progress=on_progress,
        beat_indices=[int(beat_index)],
    )

    if existing_shots:
        merged = merge_beat_shots(existing_shots, partial.shots, int(beat_index))
        return partial.model_copy(update={"shots": merged})
    return partial


def run_full_parse(
    *,
    script_text: str,
    title: str = "",
    target_duration_sec: float = 60.0,
    segment_duration_sec: float = 5.0,
    max_clip_sec: float = 10.0,
    locale: str = "zh",
    chat_fn: Callable[..., Any],
    think_apply: Callable[[str], str],
    token_budget: Callable[[int], int],
    on_progress: ProgressFn | None = None,
) -> tuple[DecomposeResult, ExpandResult]:
    decomposed = run_decompose(
        script_text=script_text,
        title=title,
        locale=locale,
        chat_fn=chat_fn,
        think_apply=think_apply,
        token_budget=token_budget,
        on_progress=on_progress,
    )
    expanded = run_expand(
        script=decomposed.artifact,
        target_duration_sec=target_duration_sec,
        segment_duration_sec=segment_duration_sec,
        max_clip_sec=max_clip_sec,
        locale=locale,
        chat_fn=chat_fn,
        think_apply=think_apply,
        token_budget=token_budget,
        on_progress=on_progress,
    )
    return decomposed, expanded
