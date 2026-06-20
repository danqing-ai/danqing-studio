"""VLM auto-caption for LoRA datasets (person name + scene, or style description)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

_STYLE_AUTO_CAPTION_INSTRUCTION = """You caption ONE image for DreamBooth **style** LoRA training.
Describe the visual style: art medium, rendering, color palette, line work, texture, mood, composition.
Do NOT identify people by name. Use concise comma-separated phrases.
Ignore any watermarks, logos, or text overlaid on the image.
Use Chinese if the artwork is Chinese-centric; otherwise English.
Output ONLY the style description — no quotes, headings, or labels."""

_STYLE_AUTO_CAPTION_RETRY_INSTRUCTION = """Describe this image's visual style in 3-8 short comma-separated phrases for LoRA training.
Focus on: art medium, color palette, rendering technique, mood.
Ignore watermarks and text. No names, no punctuation-only output.
Use Chinese if appropriate. Output ONLY the style phrases."""

_LEGACY_TEMPLATE_MARKERS = (
    "a photo of",
    "photo of",
    "sks",
    "dreambooth",
    "portrait of",
)

_GARBAGE_SCENE_RE = re.compile(r"^[\s!！?？.。,，、…\-_=~#@*]+$")
_PUNCT_RUN_RE = re.compile(r"([!！?？.。,，、…])\1{5,}")
_BANNED_BEAUTY_TEXTURE_TERMS = (
    "光滑",
    "磨皮",
    "无瑕",
    "零瑕疵",
    "细腻皮肤",
    "smooth skin",
    "flawless skin",
    "poreless",
    "airbrushed",
    "beauty retouch",
    "over-retouched",
    "plastic skin",
    "waxy skin",
)


def _count_meaningful_chars(text: str) -> int:
    return sum(1 for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def is_usable_scene_caption(text: str, *, min_meaningful: int = 2) -> bool:
    scene = (text or "").strip()
    if not scene:
        return False
    if _GARBAGE_SCENE_RE.match(scene):
        return False
    if _PUNCT_RUN_RE.search(scene):
        return False
    meaningful = _count_meaningful_chars(scene)
    if meaningful < min_meaningful:
        return False
    compact = scene.replace(" ", "")
    if len(compact) >= 8 and meaningful / max(len(compact), 1) < 0.15:
        return False
    return True


def build_concept_auto_caption_instruction(subject_name: str) -> str:
    subject = (subject_name or "").strip()
    subject_line = (
        f"Training trigger word: {subject}\nDo NOT include the trigger word in your output."
        if subject
        else "No trigger word is configured. Do NOT identify or name the person in your output."
    )
    return f"""You caption ONE photo for DreamBooth person/face LoRA training.
{subject_line}

Describe only what is visible using concise comma-separated phrases:
- shot type (特写/胸像/半身/全身, or close-up, bust, half-body, full-body)
- image orientation (竖版/横版/正方形, or portrait/landscape/square) — include if notable
- clothing, accessories, hairstyle
- expression, pose, gaze direction
- background and environment
- lighting (natural light, studio, golden hour, etc.)

Rules for special cases:
- If multiple people appear, describe ONLY the most prominent/central person; mention "多人合照" briefly but do NOT describe other people in detail.
- Ignore any text, watermarks, logos, or UI elements overlaid on the image.
- If the photo appears to be a selfie or mirror shot, note it (自拍/镜前照 or selfie/mirror shot).
- Do NOT describe skin texture or beauty-retouching qualities such as 光滑皮肤/无瑕肌肤/smooth skin/flawless skin/poreless/airbrushed.
- Do NOT turn natural skin details such as moles, freckles, acne, or blemishes into labels.

Use Chinese phrases if the subject name is Chinese; otherwise English.
Never output repeated punctuation, filler characters, or refusal text.
If the image is unclear, give a brief best-effort description (e.g. 半身人像, 户外照片).
Output ONLY the scene description — no quotes, headings, labels, or the subject name."""


def build_style_auto_caption_instruction() -> str:
    return _STYLE_AUTO_CAPTION_INSTRUCTION


def build_style_auto_caption_retry_instruction() -> str:
    return _STYLE_AUTO_CAPTION_RETRY_INSTRUCTION


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _looks_like_legacy_template(text: str) -> bool:
    low = (text or "").strip().lower()
    return any(marker in low for marker in _LEGACY_TEMPLATE_MARKERS)


def resolve_lora_subject_name(meta: dict[str, Any]) -> str:
    """Explicit trigger used for concept LoRA captions; dataset names are not identities."""
    trigger = str(meta.get("trigger_word") or "").strip()
    return trigger if trigger and not _looks_like_legacy_template(trigger) else ""


_CONCEPT_AUTO_CAPTION_RETRY_INSTRUCTION = """Describe this photo in 3-8 short comma-separated phrases for LoRA training.
Do NOT identify, infer, or include any person's name.
Include: shot type, clothing, background, lighting.
Do NOT describe skin texture or beauty-retouching qualities such as 光滑皮肤, smooth skin, flawless skin, poreless, or airbrushed.
Use Chinese if appropriate. No punctuation-only output, no exclamation marks, no filler.
Output ONLY the description phrases."""


def build_concept_auto_caption_retry_instruction() -> str:
    return _CONCEPT_AUTO_CAPTION_RETRY_INSTRUCTION


def clean_scene_caption(raw: str, *, subject_name: str = "") -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    if ":" in text.splitlines()[0]:
        first = text.splitlines()[0]
        key, val = first.split(":", 1)
        if key.strip().upper() in {"DESCRIPTION", "SCENE", "CAPTION", "OUTPUT", "STYLE"}:
            text = val.strip()
    text = text.strip().strip('"\'')
    # Strip common English VLM preambles
    text = re.sub(
        r"^(the scene (is|shows)|description:|the image (shows|depicts|displays)|this (photo|image) (shows|depicts))\s*",
        "", text, flags=re.I,
    ).strip()
    # Strip common Chinese VLM preambles
    text = re.sub(
        r"^(图片(展示了|显示了|描绘了)|照片(展示了|显示了|描绘了)|画面(展示了|显示了|中有|中有))\s*",
        "", text,
    ).strip()
    # Strip colon-separated Chinese labels
    text = re.sub(r"^(描述|场景|画面描述|风格描述)\s*[：:]\s*", "", text).strip()

    subject = (subject_name or "").strip()
    if subject:
        for prefix in (f"{subject}，", f"{subject},", f"{subject}、", f"{subject}:"):
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
                break
        text = re.sub(rf"^{re.escape(subject)}[\s\-—:：]+", "", text).strip()
        if text == subject:
            text = ""
    return text.strip()


def _drop_banned_beauty_texture_phrases(text: str) -> str:
    phrases = [p.strip() for p in re.split(r"[，,、;；]+", text or "") if p.strip()]
    if not phrases:
        return text
    kept = [
        p
        for p in phrases
        if not any(term in p.lower() for term in _BANNED_BEAUTY_TEXTURE_TERMS)
    ]
    sep = "，" if _has_cjk(text) else ", "
    return sep.join(kept)


def normalize_scene_caption(raw: str, *, subject_name: str = "") -> str:
    """Clean VLM output and reject punctuation-only / garbage captions."""
    text = clean_scene_caption(raw, subject_name=subject_name)
    if not text:
        return ""
    text = re.sub(r"^[\s!！?？.。,，、…\-_=~#@*]+", "", text)
    text = re.sub(r"[\s!！?？.。,，、…\-_=~#@*]+$", "", text)
    text = re.sub(r"([!！?？.。,，、])\1{2,}", r"\1", text)
    text = text.strip()
    if (subject_name or "").strip():
        text = _drop_banned_beauty_texture_phrases(text).strip()
        if not text:
            return ""
    if not is_usable_scene_caption(text):
        return ""
    if len(text) > 200:
        for sep in ("，", ","):
            if sep in text[:200]:
                text = text[:200].rsplit(sep, 1)[0].strip()
                break
        else:
            text = text[:200].strip()
    return text


def compose_person_caption(subject_name: str, scene: str) -> str:
    subject = (subject_name or "").strip()
    scene = normalize_scene_caption(scene, subject_name=subject)
    if not subject:
        return scene
    if not scene:
        return subject
    sep = "，" if _has_cjk(subject) else ", "
    return f"{subject}{sep}{scene}"


def caption_dataset_image(
    path: Path,
    model_dir: Path,
    *,
    audit_kind: str = "concept",
    subject_name: str = "",
    analyze_fn: Callable[[Path, str], str] | None = None,
) -> str:
    if analyze_fn is not None:
        def _analyze(image_path: Path, instruction: str, *, temperature: float = 0.2) -> str:
            return analyze_fn(image_path, instruction)

        kind = (audit_kind or "concept").strip().lower()
        if kind == "style":
            raw = _analyze(path, build_style_auto_caption_instruction())
            scene = normalize_scene_caption(raw)
            if not scene:
                raw_retry = _analyze(path, build_style_auto_caption_retry_instruction(), temperature=0.1)
                scene = normalize_scene_caption(raw_retry)
            return scene or "style reference"

        subject = (subject_name or "").strip()
        raw = _analyze(path, build_concept_auto_caption_instruction(subject))
        scene = normalize_scene_caption(raw, subject_name=subject)
        if not scene:
            raw_retry = _analyze(path, build_concept_auto_caption_retry_instruction(), temperature=0.1)
            scene = normalize_scene_caption(raw_retry, subject_name=subject)
        return compose_person_caption(subject, scene)

    caps = caption_dataset_images_batch(
        [path],
        model_dir,
        audit_kind=audit_kind,
        subject_name=subject_name,
    )
    if not caps:
        raise RuntimeError(f"VLM auto-caption returned no caption for {path.name}")
    cap = str(caps[0] or "").strip()
    if not cap:
        raise RuntimeError(f"VLM auto-caption returned empty caption for {path.name}")
    return cap


def caption_dataset_images_batch(
    paths: list[Path],
    model_dir: Path,
    *,
    audit_kind: str = "concept",
    subject_name: str = "",
    batch_analyze_fn: Callable[..., list[str]] | None = None,
) -> list[str]:
    """Caption many images with one VLM load per pass (primary + optional retry pass)."""
    if not paths:
        return []

    from backend.engine.llm.vision import analyze_image_files_batch

    analyze_batch = batch_analyze_fn or (
        lambda image_paths, instruction, max_tokens=128, temperature=0.2: analyze_image_files_batch(
            image_paths,
            model_dir,
            instruction=instruction,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    )

    kind = (audit_kind or "concept").strip().lower()
    if kind == "style":
        raw_list = analyze_batch(paths, build_style_auto_caption_instruction())
        style_captions: list[str | None] = [None] * len(paths)
        retry_paths_style: list[Path] = []
        retry_indices_style: list[int] = []
        for idx, raw in enumerate(raw_list):
            scene = normalize_scene_caption(raw)
            if scene:
                style_captions[idx] = scene
            else:
                retry_paths_style.append(paths[idx])
                retry_indices_style.append(idx)
        if retry_paths_style:
            retry_inst = build_style_auto_caption_retry_instruction()
            raw_retry = analyze_batch(retry_paths_style, retry_inst, max_tokens=128, temperature=0.1)
            for idx, raw in zip(retry_indices_style, raw_retry, strict=False):
                style_captions[idx] = normalize_scene_caption(raw) or "style reference"
        return [cap if cap is not None else "style reference" for cap in style_captions]

    subject = (subject_name or "").strip()
    primary_inst = build_concept_auto_caption_instruction(subject)
    raw_list = analyze_batch(paths, primary_inst, max_tokens=128, temperature=0.2)

    captions: list[str | None] = [None] * len(paths)
    retry_paths: list[Path] = []
    retry_indices: list[int] = []

    for idx, raw in enumerate(raw_list):
        scene = normalize_scene_caption(raw, subject_name=subject)
        if scene:
            captions[idx] = compose_person_caption(subject, scene)
        else:
            retry_paths.append(paths[idx])
            retry_indices.append(idx)

    if retry_paths:
        retry_inst = build_concept_auto_caption_retry_instruction()
        raw_retry = analyze_batch(retry_paths, retry_inst, max_tokens=128, temperature=0.1)
        for idx, raw in zip(retry_indices, raw_retry, strict=False):
            captions[idx] = compose_person_caption(subject, raw)

    missing = [
        paths[i].name
        for i, cap in enumerate(captions)
        if cap is None or not str(cap).strip()
    ]
    if missing:
        preview = ", ".join(missing[:5])
        more = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        raise RuntimeError(
            f"VLM auto-caption failed for {len(missing)}/{len(paths)} image(s): {preview}{more}. "
            "Retry auto-caption or edit captions manually."
        )
    return [str(cap).strip() for cap in captions]


