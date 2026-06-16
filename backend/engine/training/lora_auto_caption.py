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
    subject = (subject_name or "subject").strip()
    return f"""You caption ONE photo for DreamBooth person/face LoRA training.
Training subject name: {subject}
Do NOT include the person's name in your output.

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
    """Best-effort subject name for concept LoRA captions."""
    default = str(meta.get("default_prompt") or "").strip()
    trigger = str(meta.get("trigger_word") or "").strip()
    name = str(meta.get("name") or "").strip()

    for candidate in (default, trigger, name):
        if candidate and not _looks_like_legacy_template(candidate):
            return candidate
    return default or trigger or name or "subject"


_CONCEPT_AUTO_CAPTION_RETRY_INSTRUCTION = """Describe this photo in 3-8 short comma-separated phrases for LoRA training.
Subject name is already stored separately — do NOT include any person's name.
Include: shot type, clothing, background, lighting.
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
        if text == subject:
            text = ""
    return text.strip()


def normalize_scene_caption(raw: str, *, subject_name: str = "") -> str:
    """Clean VLM output and reject punctuation-only / garbage captions."""
    text = clean_scene_caption(raw, subject_name=subject_name)
    if not text:
        return ""
    text = re.sub(r"^[\s!！?？.。,，、…\-_=~#@*]+", "", text)
    text = re.sub(r"[\s!！?？.。,，、…\-_=~#@*]+$", "", text)
    text = re.sub(r"([!！?？.。,，、])\1{2,}", r"\1", text)
    text = text.strip()
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
        return scene or "a photo"
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

        subject = (subject_name or "subject").strip()
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
    return caps[0] if caps else (subject_name.strip() or "a photo")


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

    subject = (subject_name or "subject").strip()
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

    fallback = subject or "a photo"
    return [cap if cap is not None else fallback for cap in captions]


# ---------------------------------------------------------------------------
# Face-anchor generation (consistent facial-feature descriptor for LoRA)
# ---------------------------------------------------------------------------

_FACE_ANCHOR_INSTRUCTION = """You describe the FACIAL FEATURES of the person in this photo for DreamBooth face LoRA training.
Describe ONLY these aspects (use short comma-separated phrases, 4-8 phrases total):
- face shape (鹅蛋脸/圆脸/方脸/瓜子脸, or oval/round/square/heart)
- eyes (大眼/细长眼/双眼皮, or big eyes, almond eyes, double eyelids)
- nose (高鼻梁/小巧鼻, or high bridge, small nose)
- lips (薄唇/厚唇/自然唇, or thin/full/natural lips)
- skin (白皙/小麦色/光滑, or fair/tanned/smooth skin)
- eyebrows (弯眉/平眉/浓眉, or arched/straight/thick brows)

Do NOT describe: clothing, accessories, hairstyle, background, lighting, pose.
Do NOT include the person's name.
Use Chinese phrases if the subject name is Chinese; otherwise English.
Output ONLY the facial feature phrases — no quotes, headings, or labels."""


def _clean_face_anchor(raw: str) -> str:
    """Clean VLM face-anchor output into a compact comma-separated string."""
    text = (raw or "").strip()
    if not text:
        return ""
    # Strip common VLM prefixes
    text = re.sub(r"^(description|output|facial features|face features)\s*[:：]\s*", "", text, flags=re.I).strip()
    text = text.strip('"\'')
    # Remove line breaks, join with comma
    text = re.sub(r"\s*[\n\r]+\s*", "，", text)
    # Normalize separators
    text = re.sub(r"[,，、;；]+", "，", text)
    text = text.strip("，,。.")
    # Truncate if too long
    if len(text) > 120:
        for sep in ("，", ","):
            if sep in text[:120]:
                text = text[:120].rsplit(sep, 1)[0].strip()
                break
        else:
            text = text[:120].strip()
    return text


def generate_face_anchor(
    paths: list[Path],
    model_dir: Path,
    *,
    subject_name: str = "",
    sample_count: int = 5,
    batch_analyze_fn: Callable[..., list[str]] | None = None,
) -> str:
    """Analyze a sample of dataset images with VLM and produce a face_anchor string.

    The face anchor is a consistent facial-feature descriptor injected into every
    training caption, helping the LoRA learn identity-specific facial features
    rather than variable elements (clothing, scene, etc.).

    Returns a clean comma-separated string, or ``""`` on failure.
    """
    if not paths:
        return ""
    # Sample up to sample_count images (evenly spaced)
    n = len(paths)
    if n <= sample_count:
        sample = list(paths)
    else:
        step = max(1, n // sample_count)
        sample = [paths[i] for i in range(0, n, step)][:sample_count]

    from backend.engine.llm.vision import analyze_image_files_batch

    analyze_batch = batch_analyze_fn or (
        lambda image_paths, instruction, max_tokens=64, temperature=0.15: analyze_image_files_batch(
            image_paths,
            model_dir,
            instruction=instruction,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    )

    inst = _FACE_ANCHOR_INSTRUCTION
    try:
        raw_list = analyze_batch(sample, inst, max_tokens=64, temperature=0.15)
    except Exception:
        return ""

    # Collect all phrases from VLM responses
    all_phrases: list[str] = []
    for raw in raw_list:
        cleaned = _clean_face_anchor(raw)
        if cleaned:
            for sep in ("，", ","):
                if sep in cleaned:
                    all_phrases.extend(p.strip() for p in cleaned.split(sep) if p.strip())
                    break
            else:
                all_phrases.append(cleaned)

    if not all_phrases:
        return ""

    # Keep phrases that appear in >=40% of responses (consensus features).
    # Require at least 2 occurrences to filter VLM hallucinations on single images.
    from collections import Counter

    counts = Counter(all_phrases)
    threshold = max(2, round(len(sample) * 0.4))
    consensus = [p for p, c in counts.most_common() if c >= threshold]
    if len(consensus) < 3:
        # Fall back: take top phrases with count >= 2, or most common if very few
        consensus = [p for p, c in counts.most_common(6) if c >= 2]
        if len(consensus) < 2:
            consensus = [p for p, _ in counts.most_common(6)]

    if not consensus:
        return ""

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for p in consensus:
        low = p.lower()
        if low not in seen:
            seen.add(low)
            deduped.append(p)

    return "，".join(deduped[:8])
