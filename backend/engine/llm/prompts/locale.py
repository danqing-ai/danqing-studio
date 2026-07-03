"""User-message locale and task constraints (never append to system prompts)."""
from __future__ import annotations


def chapter_json_user_locale_block(locale: str) -> str:
    """Language constraint for JSON chapter analyze user messages."""
    from backend.engine.llm.storyboard import normalize_storyboard_locale

    loc = normalize_storyboard_locale(locale)
    if loc == "zh":
        return (
            "\n\n## Output language\n"
            "Write every JSON string value in Simplified Chinese (简体中文). "
            "Keep character names in the script's Chinese form."
        )
    return "\n\n## Output language\nWrite every JSON string value in English."


def scene_entity_json_user_locale_block(locale: str) -> str:
    from backend.engine.llm.storyboard import normalize_storyboard_locale

    loc = normalize_storyboard_locale(locale)
    if loc == "zh":
        return "\n\n## Output language\nWrite every JSON string value in Simplified Chinese (简体中文)."
    return "\n\n## Output language\nWrite every JSON string value in English."


def storyboard_user_locale_block(locale: str) -> str:
    """Language, pronoun, and anchor-format constraints for storyboard/chapter user messages."""
    from backend.engine.llm.storyboard import (
        normalize_storyboard_locale,
        storyboard_anchor_format_rule,
    )

    loc = normalize_storyboard_locale(locale)
    pronoun_rule = (
        "Each [Visual]/[Motion]/[Beat] is sent to image/video models alone — "
        "name every character explicitly using proper names from the input. "
        "Never start with or rely on standalone pronouns (她/他/they/she/he)."
    )
    anchor_fmt = storyboard_anchor_format_rule(loc)
    if loc == "zh":
        lang = (
            "Output language: Simplified Chinese (简体中文) ONLY for every "
            "[Synopsis], [Mood], [Anchor], [Beat], [Visual], and [Motion] line. "
            "Keep character names in Chinese script as in the input (do not romanize)."
        )
    else:
        lang = "Output language: English ONLY for every labeled block."
    return f"\n\n## Output constraints\n{lang}\n{pronoun_rule}\n{anchor_fmt}"


def scene_entity_user_locale_block(locale: str) -> str:
    from backend.engine.llm.storyboard import normalize_storyboard_locale

    loc = normalize_storyboard_locale(locale)
    if loc == "zh":
        return "\n\n## Output language\nSimplified Chinese (简体中文) ONLY for [SceneRoster] blocks."
    return "\n\n## Output language\nEnglish ONLY for [SceneRoster] blocks."


def enhance_user_locale_hint(text: str) -> str:
    """Optional polish hint when input is already detailed (user message only)."""
    raw = (text or "").strip()
    if not raw:
        return ""
    if any("\u4e00" <= ch <= "\u9fff" for ch in raw):
        if len(raw) < 60:
            return (
                "\n\n（输入较短：保留原意与关键词，最多补充少量画面细节，"
                "禁止堆砌形容词或重复用词。）"
            )
        return "\n\n（输入已够详细：只做轻微润色，禁止加长或重复用词。）"
    if len(raw) < 80:
        return (
            "\n\n(Input is short: preserve intent and keywords; add at most a few visual cues; "
            "no filler loops.)"
        )
    if len(raw) >= 80:
        return "\n\n(Input is already detailed: light polish only; do not lengthen or repeat phrases.)"
    return ""
