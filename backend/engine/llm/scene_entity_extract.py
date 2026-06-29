"""Round-2 LLM extraction of reusable scene/location entities from analyzed beats."""
from __future__ import annotations

import hashlib
from typing import Any, Callable

from pydantic import ValidationError

from backend.engine.llm.chat_invoke import invoke_text_chat
from backend.engine.llm.json_output import extract_json_object
from backend.engine.llm.prompts.locale import scene_entity_json_user_locale_block
from backend.engine.llm.prompts.system import SCENE_ENTITY_SYSTEM
from backend.engine.llm.schemas.long_video import SceneEntityPayloadSchema, SceneLookSchema
from backend.engine.llm.storyboard import normalize_storyboard_locale
from backend.engine.llm.storyboard_scenes import SceneLook, StoryboardScene, normalize_storyboard_scene_roster


def _stable_id(prefix: str, key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _scene_look_body(look: SceneLookSchema, *, locale: str) -> str:
    env = look.environment.strip()
    dress = look.set_dressing.strip()
    if locale == "zh":
        if dress:
            return f"环境：{env} | 置景：{dress}"
        return f"环境：{env}"
    if dress:
        return f"Environment: {env} | Set dressing: {dress}"
    return f"Environment: {env}"


def scenes_from_entity_json(text: str, *, locale: str) -> list[StoryboardScene]:
    try:
        payload = SceneEntityPayloadSchema.model_validate(extract_json_object(text))
    except ValidationError as exc:
        raise ValueError(f"scene entity JSON schema invalid: {exc}") from exc

    loc = normalize_storyboard_locale(locale)
    default_label = "默认" if loc == "zh" else "default"
    scenes: list[StoryboardScene] = []
    for row in payload.scenes:
        name = row.name.strip()
        if not name:
            raise ValueError("scene entity JSON has an empty scene name")
        looks: list[SceneLook] = []
        for index, lk in enumerate(row.looks):
            label = lk.label.strip() or default_label
            body = _scene_look_body(lk, locale=loc)
            looks.append(
                SceneLook(
                    id=_stable_id("slook", f"{name}|{label}|{index}|{body[:64]}"),
                    label=label,
                    body=body,
                )
            )
        scenes.append(
            StoryboardScene(
                id=_stable_id("scene", name),
                name=name,
                looks=looks,
                default_look_id=looks[0].id,
            )
        )
    if not scenes:
        raise ValueError("scene entity JSON contained no scenes")
    return normalize_storyboard_scene_roster(scenes)


def run_scene_entity_extract(
    *,
    synopsis: str,
    beat_sheet: list[str],
    locale: str = "zh",
    chat_fn: Callable[..., Any],
    think_apply: Callable[[str], str] | None = None,
    token_budget: Callable[[int], int] | None = None,
) -> tuple[list[StoryboardScene], int]:
    """Extract scene roster via LLM (round 2). Invalid JSON or schema → raise ValueError."""
    loc = normalize_storyboard_locale(locale)
    apply_think = think_apply or (lambda t: t)
    budget = token_budget or (lambda b: b)

    if not beat_sheet:
        return [], 0

    beat_lines = "\n".join(f"- {b.strip()}" for b in beat_sheet if b.strip())
    user = (
        f"Synopsis:\n{(synopsis or '').strip()[:1200]}\n\n"
        f"Storyboard beats ({len(beat_sheet)}):\n{beat_lines}\n\n"
        "Return deduplicated location entities as JSON."
        + scene_entity_json_user_locale_block(loc)
    )
    resp = invoke_text_chat(
        chat_fn,
        system=SCENE_ENTITY_SYSTEM,
        user=user,
        max_tokens=budget(1400),
        think_apply=apply_think,
    )
    return scenes_from_entity_json(resp, locale=loc), 1
