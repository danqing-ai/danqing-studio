"""
LLMService — load/run/unload LLM models via mlx-lm.
Does NOT use shared ModelCache (avoids conflict with DiT models).
Each request: load → infer → unload (load-on-demand, release-immediately).
"""

from __future__ import annotations

import gc
import logging
import secrets
import threading
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx_lm
from mlx_lm.sample_utils import make_sampler

from backend.core.contracts import (
    ChatChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatDeltaChoice,
    ChatMessage,
    DeltaMessage,
    EnhanceRequest,
    EnhanceResponse,
)
from backend.core.interfaces import AppSettings
from backend.core.model_registry import ModelEntry, ModelRegistry
from backend.engine.llm.lyrics_sanitize import lyric_line_has_annotations, sanitize_lyrics_output
from backend.engine.llm.prompt_sanitize import (
    prompt_enhance_quality_ok,
    sanitize_enhanced_prompt,
)
from backend.engine.llm.think_parse import extract_final_llm_content
from backend.engine.llm.vision import (
    IMAGE_TO_PROMPT_INSTRUCTION,
    VIDEO_FRAME_TO_PROMPT_INSTRUCTION,
    analyze_image_file,
    describe_image_file,
    mlx_vlm_importable,
    vision_weights_ready,
)
from backend.utils.path_utils import PathResolver

logger = logging.getLogger(__name__)

_SETTINGS_DEFAULTS = AppSettings()
DEFAULT_LLM_MODEL_ID = _SETTINGS_DEFAULTS.default_model_llm
DEFAULT_VLM_MODEL_ID = _SETTINGS_DEFAULTS.default_model_vlm


def _is_valid_llm_entry(entry: ModelEntry | None) -> bool:
    if entry is None or entry.media != "llm":
        return False
    return bool(entry.actions & {"chat", "enhance"})


def _is_valid_vlm_entry(entry: ModelEntry | None) -> bool:
    if entry is None or entry.media != "llm":
        return False
    return "describe" in entry.actions


def _pick_first_llm(registry: ModelRegistry) -> str | None:
    for mid in sorted(registry.all()):
        if _is_valid_llm_entry(registry.get(mid)):
            return mid
    return None


def _pick_first_vlm(registry: ModelRegistry) -> str | None:
    for mid in sorted(registry.all()):
        if _is_valid_vlm_entry(registry.get(mid)):
            return mid
    return None


def _coerce_llm_model_id(preferred: str, registry: ModelRegistry) -> str:
    candidate = (preferred or "").strip()
    if candidate and _is_valid_llm_entry(registry.get(candidate)):
        return candidate
    fallback = DEFAULT_LLM_MODEL_ID
    if _is_valid_llm_entry(registry.get(fallback)):
        return fallback
    picked = _pick_first_llm(registry)
    return picked or fallback


def _coerce_vlm_model_id(preferred: str, registry: ModelRegistry) -> str:
    candidate = (preferred or "").strip()
    if candidate and _is_valid_vlm_entry(registry.get(candidate)):
        return candidate
    fallback = DEFAULT_VLM_MODEL_ID
    if _is_valid_vlm_entry(registry.get(fallback)):
        return fallback
    picked = _pick_first_vlm(registry)
    return picked or fallback


def normalize_app_llm_settings(settings: AppSettings, registry: ModelRegistry) -> bool:
    """Align saved LLM/VLM defaults with registry; return True if settings changed."""
    changed = False
    coerced_llm = _coerce_llm_model_id(settings.default_model_llm, registry)
    if settings.default_model_llm != coerced_llm:
        if (settings.default_model_llm or "").strip():
            logger.warning(
                "default_model_llm %r not in registry; using %r",
                settings.default_model_llm,
                coerced_llm,
            )
        settings.default_model_llm = coerced_llm
        changed = True

    coerced_vlm = _coerce_vlm_model_id(settings.default_model_vlm, registry)
    if settings.default_model_vlm != coerced_vlm:
        if (settings.default_model_vlm or "").strip():
            logger.warning(
                "default_model_vlm %r not in registry; using %r",
                settings.default_model_vlm,
                coerced_vlm,
            )
        settings.default_model_vlm = coerced_vlm
        changed = True
    return changed


def resolve_llm_model_id(settings: AppSettings, registry: ModelRegistry) -> str:
    return _coerce_llm_model_id(settings.default_model_llm, registry)


def resolve_vlm_model_id(settings: AppSettings, registry: ModelRegistry) -> str:
    return _coerce_vlm_model_id(settings.default_model_vlm, registry)

ENHANCE_IMAGE_SYSTEM_PROMPT = """You are a prompt engineer for AI image models (Flux, Z-Image, Qwen-Image).
Rewrite the user's idea into one vivid, comma-separated description. Keep their subject, names, and intent.

Language: Chinese in → Chinese out; English in → English out.
If the input is already detailed, lightly polish only — do not lengthen.

Add at most a few cues for lighting, composition, color, texture, and mood.
Length cap: ~120 Chinese characters or ~80 English words.
Never repeat the same word or phrase; never loop filler at the end.
Do not write "Okay", explanations, or quotes. Output ONLY the enhanced prompt."""

ENHANCE_VIDEO_SYSTEM_PROMPT = """You are a professional prompt engineer for AI video generation.
Given a user's brief, rewrite it into a detailed prompt for image-to-video or text-to-video models.
Include subject, scene, lighting, style, camera movement, motion dynamics, pacing, and temporal mood.
For LTX audio-video models, hint ambient sound rhythm and dialogue pacing without writing looping lines.
Distinguish static scene description from continuing motion the camera can follow.
Match the user's language: Chinese input → Chinese output; English input → English output.
Keep it concise: one paragraph, at most ~120 Chinese characters or ~80 English words.
CRITICAL: Never repeat the same phrase or word. No filler loops.
Output ONLY the enhanced prompt text, without explanation or quotation marks."""

LONG_VIDEO_OPENING_SYSTEM_PROMPT = """You polish the FIRST segment prompt for a multi-pass long LTX audio-video generation.
Output ONE paragraph for Pass0 text-to-video. Must include:
1) CharacterAnchor: 2-3 sentences fixing appearance, wardrobe, palette, camera distance.
2) SceneBeat: this segment's action, camera move, and sound mood.
Match input language. Max ~180 Chinese characters or ~120 English words.
No markdown. Output ONLY the prompt."""

LONG_VIDEO_PLAN_SYSTEM_PROMPT = """You plan a timed long-video beat sheet for LTX A/V generation.
Output format ONLY:
[Anchor] <2-3 sentences: fixed character/scene identity>
[Beat 1] <one sentence plot beat for segment 1 (~opening)>
[Beat 2] <one sentence for segment 2>
... exactly N beats total as requested.
Budget: compact=quick arc; standard=mid climax; epic=slower build + late climax.
Match user language. No extra commentary."""

LONG_VIDEO_PLAN_SHOT_SYSTEM_PROMPT = """You plan a keyframe storyboard for segmented image-to-video generation.
Each [Beat] is one KEYFRAME moment (a still frame), not a transition clip.
Output format ONLY:
[Anchor]
<Cast roster — blocks separated by a line containing only --- >
One block per character LOOK (same person may have multiple looks if wardrobe changes):
【角色·<姓名>·<装扮名>】<固定发型、服饰、体型、肤色>
---
【角色·<姓名>·<另一装扮名>】<…>   (only when brief implies outfit change)
---
【画风】<全片统一的色调、镜头、胶片感>
[Beat 1] <scene/pose only; name every visible character>
[Beat 2] ...
... exactly N beats as requested.
When a beat uses a non-default look, tag it: <姓名>（<装扮名>）… e.g. 小明（晚礼服）走上红毯.
Each [Beat] must name visible characters (never 她/他/she/he alone). Do not invent outfits without a matching 【角色·…·装扮名】 block in [Anchor].
Budget: compact=quick arc; standard=mid climax; epic=slower build + late climax.
Match brief language. No markdown."""

LONG_VIDEO_EXPAND_SYSTEM_PROMPT = """Expand beat sheet lines into full LTX audio-video prompts.
Output format ONLY:
[Opening] <full Pass0 prompt with CharacterAnchor + SceneBeat>
[Segment 1] <extend pass 1 prompt: continue motion + restate anchor keywords>
[Segment 2] ...
Each segment is one paragraph; include motion, camera, ambient audio mood.
Never copy-paste identical text across segments. Match user language."""

LONG_VIDEO_EXPAND_SHOT_SYSTEM_PROMPT = """Expand story beats into prompts for keyframe + image-to-video workflow.
For each shot index N output BOTH blocks:
[Visual N] <scene-only: composition, pose, lighting — NO wardrobe/hair repeat>
[Motion N] <I2V: camera move + action; may repeat character names>
Preserve outfit tags from beats (e.g. 赵今麦（地府）) in Visual/Motion when present.
Downstream code appends [Anchor] character-look blocks (--- separated) before T2I.
Name every on-screen character. Forbidden: standalone 她/他/she/he.
Match brief language. No extra commentary."""

LONG_VIDEO_CONTINUITY_SYSTEM_PROMPT = """Fix continuity across long-video segment prompts.
Keep [Opening] and [Segment N] labels. Smooth transitions, restore missing anchor keywords,
remove repetition loops. Match user language. Output ONLY the revised script."""

LONG_VIDEO_CONTINUITY_SHOT_SYSTEM_PROMPT = """Fix continuity across keyframe storyboard prompts.
Keep every [Visual N] and [Motion N] label.
Replace standalone pronouns (她/他/she/he) with correct character names using Anchor + Beats.
[Visual N] must stay scene-only (composition/pose/lighting) — do NOT embed full Anchor; code appends reference blocks.
Remove repetition loops. Match user language. Output ONLY the revised Visual/Motion script."""

ENHANCE_AUDIO_BRIEF_SYSTEM_PROMPT = """You are a music producer writing briefs for AI music generation (ACE-Step).
Given a user's music idea, expand it into a clear, vivid description covering genre, mood, tempo feel, instrumentation,
vocal style, and emotional arc. Match the user's language when the input is Chinese.
Keep it concise: one short paragraph. Never repeat the same phrase or word. No filler loops.
Output ONLY the enhanced brief text, without explanation or quotation marks."""

LYRICS_SYSTEM_PROMPT = """# ACE-Step lyrics

Reply with **only** a lyric script. No title, planning, markdown fences, or text before/after the script.

Infer structure and language from the examples below. Match the music description language (Chinese description → Chinese example shape; English → English shape). Use the instrumental example when the request has no vocals.

## Vocal · Chinese

```
[Verse 1]
清晨的风吹过旧街角
你的笑容还在心头绕

[Chorus]
我们是今夜的星光
照亮这片无尽的天

[Outro]
慢慢沉入安静的夜
```

## Vocal · English

```
[Verse 1]
Walking down the empty street at dawn
City lights fading one by one

[Chorus]
We are the stars tonight
Shining through the endless sky

[Outro]
Fade into the quiet night
```

## Instrumental

```
[Instrumental]
```

## Counter-example (invalid — never resemble this)

```
[Verse 1]
青峰云海间 (5 chars) - "Green peaks, cloud sea"
Here is the chorus:
[Chorus]
We are the stars tonight
```
"""

DESCRIBE_NODE_SYSTEM_PROMPT = """You are a creative studio assistant writing short notes on canvas nodes.
Given metadata about a generated asset (title, prompt, model, dimensions, lineage), write a concise note (2-4 sentences)
that helps the artist remember what this node is and how to iterate next.
Be specific about style, subject, and suggested next steps. Match the user's language when metadata is Chinese.
Output ONLY the note text, without quotes or headings."""


class LLMService:
    """Load → infer → unload MLX LLM models on demand.

    Does NOT participate in the shared ModelCache (max_entries=1) used by
    image/video/audio engines.  Each request loads the model, runs inference,
    and immediately releases GPU memory to avoid OOM conflicts.
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        path_resolver: PathResolver,
        default_model_id: str = DEFAULT_LLM_MODEL_ID,
        vision_model_id: str = DEFAULT_VLM_MODEL_ID,
        llm_think_enabled: bool = False,
    ):
        self._registry = model_registry
        self._path_resolver = path_resolver
        self._model_id = default_model_id
        self._vision_model_id = vision_model_id
        self._llm_think_enabled = bool(llm_think_enabled)
        self._generation_lock = threading.Lock()

    def apply_model_settings(
        self,
        *,
        default_model_id: str | None = None,
        vision_model_id: str | None = None,
        llm_think_enabled: bool | None = None,
    ) -> None:
        if default_model_id is not None:
            coerced = _coerce_llm_model_id(default_model_id, self._registry)
            self._registry.require(coerced)
            self._model_id = coerced
        if vision_model_id is not None:
            coerced = _coerce_vlm_model_id(vision_model_id, self._registry)
            self._registry.require(coerced)
            self._vision_model_id = coerced
        if llm_think_enabled is not None:
            self._llm_think_enabled = bool(llm_think_enabled)
        if not self._is_thinking_model(self._model_id):
            self._llm_think_enabled = False

    # ------------------------------------------------------------------
    # Public status
    # ------------------------------------------------------------------

    def get_model_info(self) -> dict[str, Any]:
        """Return current LLM model id, availability, and resolved path."""
        entry = self._registry.get(self._model_id)
        return {
            "model_id": self._model_id,
            "name": self._registry_display_name(entry, self._model_id),
            "available": self.is_available(),
            "think_supported": self._is_thinking_model(self._model_id),
            "think_enabled": self._llm_think_enabled,
        }

    def get_vision_model_info(self) -> dict[str, Any]:
        entry = self._registry.get(self._vision_model_id)
        return {
            "model_id": self._vision_model_id,
            "name": self._registry_display_name(entry, self._vision_model_id),
            "available": self.is_vision_available(),
            "mlx_vlm_installed": mlx_vlm_importable(),
        }

    def is_vision_available(self) -> bool:
        if not mlx_vlm_importable():
            return False
        try:
            path = self._resolve_vision_model_path()
            return vision_weights_ready(path)
        except Exception:
            return False

    def is_available(self) -> bool:
        """True when the model directory exists and contains weight files."""
        try:
            path = self._resolve_model_path()
            if not path.is_dir():
                return False
            # Accept either single model.safetensors or any sharded safetensors/bin files
            return (
                (path / "model.safetensors").is_file()
                or any(f.suffix == ".safetensors" for f in path.rglob("*") if f.is_file())
                or any(f.suffix == ".bin" for f in path.rglob("*") if f.is_file())
            )
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Chat completion
    # ------------------------------------------------------------------

    def chat_completion(
        self,
        request: ChatCompletionRequest,
        *,
        enable_thinking: bool | None = None,
    ) -> ChatCompletionResponse:
        """Run a single-turn non-streaming chat completion (sync, for asyncio.to_thread)."""
        thinking = self._resolve_enable_thinking(enable_thinking)
        think_active = self._think_is_active(thinking)
        with self._generation_lock:
            model, tokenizer = self._load_model()
            try:
                messages = self._apply_think_mode_to_messages(request.messages)
                prompt = self._build_chat_prompt(
                    tokenizer,
                    messages,
                    enable_thinking=thinking,
                )
                result = mlx_lm.generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    verbose=False,
                    **self._generation_kwargs(request, think_active=think_active),
                )
                content = extract_final_llm_content(result, think_enabled=think_active)
                return self._format_response(content, request.model)
            finally:
                self._unload_model(model, tokenizer)

    # ------------------------------------------------------------------
    # Chat completion (SSE streaming)
    # ------------------------------------------------------------------

    async def chat_completion_stream(self, request: ChatCompletionRequest):
        """Yield SSE lines in OpenAI format: data: {json}\n\n ... data: [DONE]\n\n"""
        import asyncio

        with self._generation_lock:
            model, tokenizer = self._load_model()
            try:
                messages = self._apply_think_mode_to_messages(request.messages)
                thinking = self._resolve_enable_thinking(None)
                prompt_text = self._build_chat_prompt(
                    tokenizer,
                    messages,
                    enable_thinking=thinking,
                )
            except Exception:
                self._unload_model(model, tokenizer)
                raise

        think_active = self._think_is_active(thinking)

        # Release the lock while streaming so other requests can queue.
        # The model stays loaded until we explicitly unload.
        try:
            chunk_id = f"chatcmpl-{secrets.token_hex(12)}"
            created = int(time.time())

            def _generate():
                return list(
                    mlx_lm.stream_generate(
                        model,
                        tokenizer,
                        prompt=prompt_text,
                        **self._generation_kwargs(request, think_active=think_active),
                    )
                )

            responses = await asyncio.to_thread(_generate)

            for resp in responses:
                chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatDeltaChoice(
                            index=0,
                            delta=DeltaMessage(content=resp.text),
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

            # Final chunk with finish_reason
            last_resp = responses[-1] if responses else None
            finish_reason = last_resp.finish_reason if last_resp else "stop"
            final_chunk = ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=request.model,
                choices=[
                    ChatDeltaChoice(
                        index=0,
                        delta=DeltaMessage(),
                        finish_reason=finish_reason,
                    )
                ],
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            self._unload_model(model, tokenizer)

    # ------------------------------------------------------------------
    # Prompt enhancement
    # ------------------------------------------------------------------

    @staticmethod
    def _enhance_system_prompt(target_action: str) -> str:
        action = (target_action or "image_create").strip().lower()
        if action in ("video", "video_create", "animate", "video_generation"):
            return ENHANCE_VIDEO_SYSTEM_PROMPT
        if action in ("long_video_opening",):
            return LONG_VIDEO_OPENING_SYSTEM_PROMPT
        if action in ("audio", "audio_create", "music", "audio_generation"):
            return ENHANCE_AUDIO_BRIEF_SYSTEM_PROMPT
        return ENHANCE_IMAGE_SYSTEM_PROMPT

    def enhance_prompt(self, request: EnhanceRequest) -> EnhanceResponse:
        """Enhance a creative brief for image, video, or audio generation."""
        system_prompt = self._enhance_system_prompt(request.target_action)
        action = (request.target_action or "image_create").strip().lower()
        raw_prompt = (request.prompt or "").strip()
        user_content = raw_prompt
        if action in ("image_create", "create", "image") and len(raw_prompt) >= 60:
            if any("\u4e00" <= ch <= "\u9fff" for ch in raw_prompt):
                user_content += "\n\n（输入已够详细：只做轻微润色，禁止加长或重复用词。）"
            elif len(raw_prompt) >= 80:
                user_content += "\n\n(Input is already detailed: light polish only; do not lengthen or repeat phrases.)"
        user_content = self._apply_think_mode_to_text(user_content)

        think_active = self._think_is_active(self._resolve_enable_thinking(None))
        attempts = (
            (0.65, self._token_budget(200, think_active)),
            (0.45, self._token_budget(160, think_active)),
            (0.35, self._token_budget(140, think_active)),
        )
        last_clean = ""
        for temperature, max_tokens in attempts:
            internal = ChatCompletionRequest(
                model=self._model_id,
                messages=[
                    ChatMessage(role="system", content=system_prompt),
                    ChatMessage(role="user", content=user_content),
                ],
                temperature=temperature,
                top_p=0.9,
                max_tokens=max_tokens,
                stream=False,
            )
            result = self.chat_completion(internal)
            cleaned = sanitize_enhanced_prompt(
                result.choices[0].message.content,
                think_enabled=think_active,
            )
            if prompt_enhance_quality_ok(cleaned):
                return EnhanceResponse(enhanced_prompt=cleaned)
            last_clean = cleaned

        fallback = sanitize_enhanced_prompt(raw_prompt, think_enabled=think_active)
        if prompt_enhance_quality_ok(last_clean):
            return EnhanceResponse(enhanced_prompt=last_clean)
        return EnhanceResponse(enhanced_prompt=fallback or last_clean)

    def generate_long_video_storyboard(
        self,
        request: "LongVideoStoryboardRequest",
    ) -> "LongVideoStoryboardResponse":
        from backend.core.contracts import (
            LongVideoPlanDTO,
            LongVideoStoryboardRequest,
            LongVideoStoryboardResponse,
            LongVideoStoryboardShotDTO,
            LongVideoCharacterDTO,
            LongVideoCharacterLookDTO,
            LongVideoShotCastLookDTO,
        )
        from backend.engine.common.long_video.plan import build_shot_plan
        from backend.engine.families.ltx.long_video_plan import LongVideoPlan, build_long_video_plan
        from backend.engine.llm.storyboard_cast import (
            format_character_roster,
            parse_character_roster,
            roster_to_dtos,
        )
        from backend.engine.llm.storyboard import (
            apply_storyboard_anchor_locale,
            apply_storyboard_output_locale,
            build_structured_shots,
            coalesce_dual_pairs,
            expand_batches_for_plan,
            expand_batches_for_shot_count,
            merge_expand_batches,
            normalize_storyboard_locale,
            normalize_character_anchor,
            parse_dual_shot_script,
            parse_expand_script,
            parse_plan_script,
            plan_to_dto,
            shot_plan_to_dto,
            storyboard_language_rule,
            storyboard_language_user_suffix,
            storyboard_quality_ok,
            storyboard_shot_pairs_ok,
            dual_pairs_from_beats,
        )

        raw = (request.prompt or "").strip()
        if not raw:
            raise RuntimeError("long_video_storyboard requires a non-empty prompt")

        locale = normalize_storyboard_locale(getattr(request, "locale", None))
        lang_rule = storyboard_language_rule(locale)
        lang_suffix = storyboard_language_user_suffix(locale)

        plan = build_long_video_plan(
            target_duration_sec=request.target_duration_sec,
            initial_duration_sec=request.initial_duration_sec,
            segment_extend_sec=request.segment_extend_sec,
            reference_duration_sec=request.reference_duration_sec,
        )
        shot_plan = build_shot_plan(
            target_duration_sec=request.target_duration_sec,
            segment_duration_sec=request.segment_duration_sec,
        )
        expected_beats = shot_plan.shot_count if request.use_shot_plan else plan.total_segments
        llm_calls = 0
        think_active = self._think_is_active(self._resolve_enable_thinking(None))

        if request.use_shot_plan:
            plan_user = (
                f"Brief: {raw}\n"
                f"Keyframe shots: {shot_plan.shot_count} "
                f"(~{shot_plan.segment_duration_sec}s I2V clip per edge)\n"
                f"Total target duration: {shot_plan.target_duration_sec}s\n"
                f"Narrative budget: {shot_plan.narrative_budget}\n"
                f"Write exactly {expected_beats} [Beat] lines after [Anchor] — one per keyframe."
            )
            plan_system = LONG_VIDEO_PLAN_SHOT_SYSTEM_PROMPT
            expand_system = LONG_VIDEO_EXPAND_SHOT_SYSTEM_PROMPT
            continuity_system = LONG_VIDEO_CONTINUITY_SHOT_SYSTEM_PROMPT
        else:
            plan_user = (
                f"Brief: {raw}\n"
                f"Segments: {plan.total_segments} (1 opening + {plan.extend_pass_count} extends)\n"
                f"Durations (sec): {list(plan.segment_durations_sec)}\n"
                f"Narrative budget: {plan.narrative_budget}\n"
                f"Write exactly {expected_beats} [Beat] lines after [Anchor]."
            )
            plan_system = LONG_VIDEO_PLAN_SYSTEM_PROMPT
            expand_system = LONG_VIDEO_EXPAND_SYSTEM_PROMPT
            continuity_system = LONG_VIDEO_CONTINUITY_SYSTEM_PROMPT
        plan_system = f"{plan_system}\n\n{lang_rule}"
        expand_system = f"{expand_system}\n\n{lang_rule}"
        continuity_system = f"{continuity_system}\n\n{lang_rule}"
        if request.style_positive.strip():
            plan_user += f"\nStyle: {request.style_positive.strip()}"
        plan_user += lang_suffix

        plan_resp = self.chat_completion(
            ChatCompletionRequest(
                model=self._model_id,
                messages=[
                    ChatMessage(role="system", content=plan_system),
                    ChatMessage(role="user", content=self._apply_think_mode_to_text(plan_user)),
                ],
                temperature=0.55,
                top_p=0.9,
                max_tokens=self._token_budget(600, think_active),
                stream=False,
            )
        )
        llm_calls += 1
        plan_text = sanitize_enhanced_prompt(
            plan_resp.choices[0].message.content,
            think_enabled=think_active,
        )
        character_anchor, beat_sheet = parse_plan_script(plan_text, expected_beats=expected_beats)
        if len(character_anchor.strip()) < 12:
            character_anchor = raw[:240].strip()
        roster, style_anchor = parse_character_roster(character_anchor, locale=locale)
        character_anchor = apply_storyboard_anchor_locale(
            format_character_roster(roster, style_anchor, locale=locale) if roster else character_anchor,
            beat_sheet=beat_sheet[:expected_beats],
            locale=locale,
        )
        if roster and not style_anchor:
            _, style_anchor = parse_character_roster(character_anchor, locale=locale)
        character_dtos = roster_to_dtos(roster)

        expand_expected = shot_plan.shot_count if request.use_shot_plan else plan.extend_pass_count

        segment_batches: list[list[str]] = []
        dual_pairs: list[tuple[str, str]] = []
        opening_parts: list[str] = []
        expand_batches = (
            expand_batches_for_shot_count(shot_plan.shot_count)
            if request.use_shot_plan
            else expand_batches_for_plan(plan)
        )
        for start, count in expand_batches:
            batch_beats = beat_sheet[start : start + count]
            expand_user = (
                f"Anchor:\n{character_anchor}\n\nBeats:\n"
                + "\n".join(f"- {beat_sheet[i]}" for i in range(start, min(start + count, len(beat_sheet))))
                + f"\n\nExpand shots {start + 1}..{start + count}. "
                f"[Visual N] = scene-only; Anchor blocks are appended automatically before T2I."
                + lang_suffix
            )
            expand_resp = self.chat_completion(
                ChatCompletionRequest(
                    model=self._model_id,
                    messages=[
                        ChatMessage(role="system", content=expand_system),
                        ChatMessage(role="user", content=self._apply_think_mode_to_text(expand_user)),
                    ],
                    temperature=0.6,
                    top_p=0.9,
                    max_tokens=self._token_budget(1400 if request.use_shot_plan else 900, think_active),
                    stream=False,
                )
            )
            llm_calls += 1
            expand_text = sanitize_enhanced_prompt(
                expand_resp.choices[0].message.content,
                think_enabled=think_active,
            )
            try:
                batch_dual = parse_dual_shot_script(
                    expand_text,
                    expected_shots=count,
                    fallback=batch_beats,
                )
                dual_pairs.extend(batch_dual)
                segment_batches.append([m for _, m in batch_dual])
            except ValueError:
                try:
                    opening, segs = parse_expand_script(
                        expand_text,
                        expected_segments=count,
                        fallback=batch_beats,
                    )
                except ValueError:
                    segs = list(batch_beats)
                    if len(segs) < count:
                        segs = dual_pairs_from_beats(
                            beat_sheet[start : start + count],
                            count,
                            character_anchor=character_anchor,
                        )
                        segs = [m for _, m in segs]
                    opening = ""
                opening_parts.append(opening)
                segment_batches.append(segs)

        if not dual_pairs and beat_sheet and request.use_shot_plan:
            dual_pairs = dual_pairs_from_beats(
                beat_sheet,
                shot_plan.shot_count,
                character_anchor=character_anchor,
            )

        if request.use_shot_plan and dual_pairs:
            dual_pairs = coalesce_dual_pairs(
                dual_pairs,
                beat_sheet,
                shot_plan.shot_count,
                character_anchor=character_anchor,
            )

        if dual_pairs:
            opening_prompt = dual_pairs[0][0] if dual_pairs else ""
            segment_prompts = [m for _, m in dual_pairs]
        else:
            opening_prompt, segment_prompts = merge_expand_batches(opening_parts, segment_batches)
        if not opening_prompt and beat_sheet:
            opening_prompt = beat_sheet[0]
        if request.use_shot_plan and shot_plan.shot_count > 0:
            from backend.engine.llm.storyboard import _pad_strings

            beat_sheet = _pad_strings(beat_sheet, shot_plan.shot_count, label="beat sheet")

        quality_plan = plan
        if request.use_shot_plan:
            quality_plan = LongVideoPlan(
                target_duration_sec=shot_plan.target_duration_sec,
                initial_duration_sec=shot_plan.segment_duration_sec,
                segment_extend_sec=shot_plan.segment_duration_sec,
                reference_duration_sec=request.reference_duration_sec,
                extend_pass_count=expand_expected,
                total_segments=expected_beats,
                segment_durations_sec=shot_plan.segment_durations_sec,
                narrative_budget=shot_plan.narrative_budget,
            )

        def _quality_ok() -> bool:
            if request.use_shot_plan:
                return (
                    len(character_anchor.strip()) >= 12
                    and storyboard_shot_pairs_ok(
                        dual_pairs,
                        shot_count=shot_plan.shot_count,
                        beat_sheet=beat_sheet[:expected_beats],
                        character_anchor=character_anchor,
                        characters=character_dtos,
                        style_anchor=style_anchor,
                        locale=locale,
                    )
                )
            return storyboard_quality_ok(
                character_anchor=character_anchor,
                opening_prompt=opening_prompt,
                segment_prompts=segment_prompts,
                beat_sheet=beat_sheet[:expected_beats],
                plan=quality_plan,
                min_segment_prompts=0,
            )

        if not _quality_ok():
            if request.use_shot_plan:
                pairs_for_cont = dual_pairs or [
                    (opening_prompt, segment_prompts[i] if i < len(segment_prompts) else beat_sheet[i])
                    for i in range(expand_expected)
                ]
                beat_block = "\n".join(
                    f"[Beat {i + 1}] {b}" for i, b in enumerate(beat_sheet[:expand_expected])
                )
                cont_user = (
                    f"Anchor:\n{character_anchor}\n\nBeats (cast per shot):\n{beat_block}\n\n"
                    + "\n".join(
                        f"[Visual {i + 1}] {v}\n[Motion {i + 1}] {m}"
                        for i, (v, m) in enumerate(pairs_for_cont)
                    )
                    + lang_suffix
                )
            else:
                cont_user = (
                    f"Anchor:\n{character_anchor}\n\nOpening:\n{opening_prompt}\n\nSegments:\n"
                    + "\n".join(f"[Segment {i+1}] {p}" for i, p in enumerate(segment_prompts))
                    + lang_suffix
                )
            cont_resp = self.chat_completion(
                ChatCompletionRequest(
                    model=self._model_id,
                    messages=[
                        ChatMessage(role="system", content=continuity_system),
                        ChatMessage(role="user", content=self._apply_think_mode_to_text(cont_user)),
                    ],
                    temperature=0.45,
                    top_p=0.9,
                    max_tokens=self._token_budget(1600 if request.use_shot_plan else 1000, think_active),
                    stream=False,
                )
            )
            llm_calls += 1
            cont_text = sanitize_enhanced_prompt(
                cont_resp.choices[0].message.content,
                think_enabled=think_active,
            )
            if request.use_shot_plan:
                dual_pairs = parse_dual_shot_script(
                    cont_text,
                    expected_shots=expand_expected,
                    fallback=beat_sheet[:expand_expected],
                )
                dual_pairs = coalesce_dual_pairs(
                    dual_pairs,
                    beat_sheet,
                    shot_plan.shot_count,
                    character_anchor=character_anchor,
                )
                opening_prompt = dual_pairs[0][0] if dual_pairs else opening_prompt
                segment_prompts = [m for _, m in dual_pairs]
            else:
                opening_prompt, segment_prompts = parse_expand_script(
                    cont_text,
                    expected_segments=expand_expected,
                    fallback=beat_sheet[:expand_expected],
                )

        def _localized_shots(shot_dicts: list[dict]) -> list[dict]:
            return apply_storyboard_output_locale(
                shot_dicts,
                beat_sheet=beat_sheet,
                locale=locale,
            )

        def _shot_build_kwargs() -> dict:
            return {
                "character_anchor": character_anchor,
                "opening_prompt": opening_prompt,
                "segment_prompts": segment_prompts,
                "beat_sheet": beat_sheet,
                "target_duration_sec": request.target_duration_sec,
                "segment_duration_sec": request.segment_duration_sec,
                "dual_pairs": dual_pairs or None,
                "characters": character_dtos,
                "style_anchor": style_anchor,
                "locale": locale,
            }

        def _to_shot_dtos(shot_dicts: list[dict]) -> list[LongVideoStoryboardShotDTO]:
            out: list[LongVideoStoryboardShotDTO] = []
            for s in shot_dicts:
                cast = [
                    LongVideoShotCastLookDTO(**row)
                    for row in (s.get("cast_looks") or [])
                    if isinstance(row, dict)
                ]
                out.append(
                    LongVideoStoryboardShotDTO(
                        id=str(s.get("id", "")),
                        order=int(s.get("order", 0)),
                        visual_prompt=str(s.get("visual_prompt", "")),
                        motion_prompt=str(s.get("motion_prompt", "")),
                        scene_prompt=str(s.get("scene_prompt", "")),
                        cast_looks=cast,
                    )
                )
            return out

        def _to_character_dtos() -> list[LongVideoCharacterDTO]:
            items: list[LongVideoCharacterDTO] = []
            for row in character_dtos:
                looks = [
                    LongVideoCharacterLookDTO(**lk)
                    for lk in (row.get("looks") or [])
                    if isinstance(lk, dict)
                ]
                items.append(
                    LongVideoCharacterDTO(
                        id=str(row.get("id", "")),
                        name=str(row.get("name", "")),
                        default_look_id=str(row.get("default_look_id", "")),
                        looks=looks,
                    )
                )
            return items

        if not _quality_ok():
            shot_dicts = _localized_shots(build_structured_shots(**_shot_build_kwargs()))
            if request.use_shot_plan and any(
                str(s.get("visual_prompt", "")).strip() for s in shot_dicts
            ):
                dto = LongVideoPlanDTO(**plan_to_dto(plan))
                return LongVideoStoryboardResponse(
                    character_anchor=character_anchor,
                    opening_prompt=opening_prompt,
                    segment_prompts=segment_prompts,
                    segment_count=expand_expected,
                    plan=dto,
                    beat_sheet=beat_sheet[:expected_beats],
                    llm_calls=llm_calls,
                    shots=_to_shot_dtos(shot_dicts),
                    characters=_to_character_dtos(),
                    style_anchor=style_anchor,
                )
            raise RuntimeError(
                "long_video_storyboard quality check failed after Plan/Expand/Continuity"
            )

        shot_dicts = _localized_shots(build_structured_shots(**_shot_build_kwargs()))
        dto = LongVideoPlanDTO(**plan_to_dto(plan))
        return LongVideoStoryboardResponse(
            character_anchor=character_anchor,
            opening_prompt=opening_prompt,
            segment_prompts=segment_prompts,
            segment_count=expand_expected,
            plan=dto,
            beat_sheet=beat_sheet[:expected_beats],
            llm_calls=llm_calls,
            shots=_to_shot_dtos(shot_dicts),
            characters=_to_character_dtos(),
            style_anchor=style_anchor,
        )

    # ------------------------------------------------------------------
    # Lyrics generation
    # ------------------------------------------------------------------

    def _lyrics_chat_request(
        self,
        user_msg: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model=self._model_id,
            messages=[
                ChatMessage(role="system", content=LYRICS_SYSTEM_PROMPT),
                ChatMessage(role="user", content=user_msg),
            ],
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens,
            stream=False,
        )

    @staticmethod
    def _lyrics_quality_ok(text: str) -> bool:
        cleaned = (text or "").strip()
        if not cleaned:
            return False

        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        if not lines:
            return False

        if len(lines) == 1 and lines[0].lower().strip("[]") == "instrumental":
            return True

        section_tags = [ln for ln in lines if ln.startswith("[") and ln.endswith("]")]
        if not section_tags:
            return False

        lyric_lines = [ln for ln in lines if not (ln.startswith("[") and ln.endswith("]"))]
        if not lyric_lines:
            return False

        total_chars = sum(len(ln) for ln in lyric_lines)
        if total_chars < 8:
            return False
        if len(lyric_lines) < 2 and total_chars < 16:
            return False

        for ln in lyric_lines:
            if lyric_line_has_annotations(ln):
                return False
            words = ln.split()
            if len(words) >= 6 and len(set(w.lower() for w in words)) / len(words) < 0.35:
                return False
            if len(ln) > 120:
                return False
        return True

    @staticmethod
    def _build_lyrics_user_message(prompt: str, style: str | None = None) -> str:
        parts = ["## Music description", (prompt or "").strip()]
        if (style or "").strip():
            parts.extend(["", "## Style", style.strip()])
        return "\n".join(parts)

    def generate_lyrics(self, prompt: str, style: str | None = None) -> str:
        """Generate ACE-Step formatted lyrics from a music description."""
        user_msg = self._with_no_think_suffix(self._build_lyrics_user_message(prompt, style))

        attempts = (
            (0.65, 420),
            (0.5, 360),
            (0.4, 512),
        )
        last_raw = ""
        for temp, max_tokens in attempts:
            result = self.chat_completion(
                self._lyrics_chat_request(user_msg, temperature=temp, max_tokens=max_tokens),
                enable_thinking=False,
            )
            last_raw = result.choices[0].message.content.strip()
            cleaned = sanitize_lyrics_output(last_raw, think_enabled=False)
            if self._lyrics_quality_ok(cleaned):
                return cleaned

        fallback = sanitize_lyrics_output(last_raw, think_enabled=False)
        if fallback and self._lyrics_quality_ok(fallback):
            return fallback

        raise RuntimeError(
            "LLM returned no usable lyrics. Check Settings → Default LLM Model or try again."
        )

    # ------------------------------------------------------------------
    # Vision: image/video → generation prompt & reference analysis
    # ------------------------------------------------------------------

    def image_to_prompt(
        self,
        asset_context: dict[str, Any],
        *,
        image_path: Path,
        media: str = "image",
    ) -> tuple[str, bool]:
        """Reverse-engineer a generation prompt from a reference image or video keyframe."""
        if not self.is_vision_available():
            raise RuntimeError(
                "Vision model not available. Install a VLM from Models page and set Settings → Default VLM Model."
            )
        meta = asset_context.get("metadata") or {}
        metadata_hint = self._metadata_hint_lines(asset_context, meta)
        instruction = (
            VIDEO_FRAME_TO_PROMPT_INSTRUCTION
            if media == "video"
            else IMAGE_TO_PROMPT_INSTRUCTION
        )
        with self._generation_lock:
            prompt = analyze_image_file(
                image_path,
                self._resolve_vision_model_path(),
                instruction=instruction,
                metadata_hint=metadata_hint,
                max_tokens=384,
            )
        return self._clean_vlm_prompt(prompt), True

    def analyze_reference(
        self,
        asset_context: dict[str, Any],
        *,
        image_path: Path,
        question: str,
    ) -> tuple[str, bool]:
        """Answer a creative question about a reference image (style, palette, subject, etc.)."""
        if not self.is_vision_available():
            raise RuntimeError(
                "Vision model not available. Install a VLM from Models page and set Settings → Default VLM Model."
            )
        meta = asset_context.get("metadata") or {}
        metadata_hint = self._metadata_hint_lines(asset_context, meta)
        instruction = (
            "You are a creative director analyzing a reference image for an artist.\n"
            f"Task: {question.strip()}\n"
            "Be specific and actionable for the next generation. "
            "Match Chinese if the user question is Chinese. Output ONLY the analysis text."
        )
        with self._generation_lock:
            answer = analyze_image_file(
                image_path,
                self._resolve_vision_model_path(),
                instruction=instruction,
                metadata_hint=metadata_hint,
                max_tokens=320,
            )
        return answer.strip(), True

    # ------------------------------------------------------------------
    # Canvas node description (metadata-based; text LLM)
    # ------------------------------------------------------------------

    def describe_node(
        self,
        asset_context: dict[str, Any],
        *,
        image_path: Path | None = None,
        prefer_vision: bool = True,
    ) -> tuple[str, bool]:
        """Generate a canvas node note. Uses VLM for image/video when available."""
        meta = asset_context.get("metadata") or {}
        kind = str(asset_context.get("kind") or "image")
        metadata_hint = self._metadata_hint_lines(asset_context, meta)

        if (
            prefer_vision
            and image_path is not None
            and kind in ("image", "video")
            and self.is_vision_available()
        ):
            with self._generation_lock:
                try:
                    note = describe_image_file(
                        image_path,
                        self._resolve_vision_model_path(),
                        metadata_hint=metadata_hint,
                    )
                    return note.strip(), True
                except Exception as exc:
                    logger.warning("Vision describe failed, using text LLM: %s", exc)

        note = self._describe_node_from_metadata(metadata_hint)
        return note, False

    @staticmethod
    def _clean_vlm_prompt(text: str) -> str:
        t = text.strip()
        if t.lower().startswith("prompt:"):
            t = t.split(":", 1)[1].strip()
        return t

    @staticmethod
    def _metadata_hint_lines(asset_context: dict[str, Any], meta: dict[str, Any]) -> str:
        lines = [
            f"Kind: {asset_context.get('kind', 'image')}",
            f"Title: {meta.get('title') or ''}",
            f"Prompt: {meta.get('prompt') or ''}",
            f"Model: {meta.get('model') or ''}",
            f"Size: {asset_context.get('width') or meta.get('width')}x"
            f"{asset_context.get('height') or meta.get('height')}",
            f"Source action: {asset_context.get('source_action') or ''}",
            f"Relation: {asset_context.get('relation_type') or ''}",
        ]
        if asset_context.get("duration_seconds"):
            lines.append(f"Duration (s): {asset_context.get('duration_seconds')}")
        if asset_context.get("parent_asset_id"):
            lines.append(f"Parent asset: {asset_context.get('parent_asset_id')}")
        if asset_context.get("relation_type"):
            lines.append(f"Lineage relation: {asset_context.get('relation_type')}")
        return "\n".join(lines)

    def _describe_node_from_metadata(self, metadata_hint: str) -> str:
        user_msg = f"Asset metadata:\n{metadata_hint}\n\nWrite a canvas node note:"
        internal = ChatCompletionRequest(
            model=self._model_id,
            messages=[
                ChatMessage(role="system", content=DESCRIBE_NODE_SYSTEM_PROMPT),
                ChatMessage(role="user", content=user_msg),
            ],
            temperature=0.6,
            max_tokens=256,
            stream=False,
        )
        result = self.chat_completion(internal)
        return result.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Model lifecycle (load-on-demand, release-immediately)
    # ------------------------------------------------------------------

    def _resolve_vision_model_path(self) -> Path:
        entry = self._registry.require(self._vision_model_id)
        versions = entry.raw.get("versions") or {}
        default_ver = next(
            (v for v in versions.values() if v.get("default")),
            next(iter(versions.values()), None),
        )
        if default_ver is None:
            raise RuntimeError(
                f"No versions defined for vision model {self._vision_model_id!r} in registry"
            )
        local_path = default_ver.get("local_path")
        if not local_path:
            raise RuntimeError(
                f"No local_path for default version of {self._vision_model_id!r}"
            )
        return self._path_resolver.resolve_registry_local_path(local_path)

    def _resolve_model_path(self) -> Path:
        entry = self._registry.require(self._model_id)
        versions = entry.raw.get("versions") or {}
        default_ver = next(
            (v for v in versions.values() if v.get("default")),
            next(iter(versions.values()), None),
        )
        if default_ver is None:
            raise RuntimeError(
                f"No versions defined for LLM model {self._model_id!r} in registry"
            )
        local_path = default_ver.get("local_path")
        if not local_path:
            raise RuntimeError(
                f"No local_path for default version of {self._model_id!r}"
            )
        return self._path_resolver.resolve_registry_local_path(local_path)

    def _load_model(self) -> tuple[Any, Any]:
        """Load the LLM model + tokenizer into GPU memory."""
        model_path = self._resolve_model_path()
        logger.info("Loading LLM model from %s", model_path)
        model, tokenizer = mlx_lm.load(str(model_path))
        logger.info("LLM model loaded successfully")
        return model, tokenizer

    def _unload_model(self, model: Any, tokenizer: Any) -> None:
        """Release model from GPU memory."""
        del model
        del tokenizer
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass
        logger.info("LLM model unloaded")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _registry_display_name(entry: Any, fallback: str) -> Any:
        """Registry ``name`` may be a bilingual dict or plain string."""
        if entry is None:
            return fallback
        raw_name = entry.raw.get("name")
        return raw_name if raw_name is not None else fallback

    def _generation_kwargs(
        self,
        request: ChatCompletionRequest,
        *,
        think_active: bool = False,
    ) -> dict[str, Any]:
        """Sampling kwargs for mlx-lm >= 0.31 (``sampler`` instead of ``temp``)."""
        max_tokens = request.max_tokens or 512
        if think_active:
            max_tokens = self._token_budget(max_tokens, True)
        return {
            "max_tokens": max_tokens,
            "sampler": make_sampler(
                temp=request.temperature,
                top_p=request.top_p,
            ),
        }

    @staticmethod
    def _token_budget(base: int, think_active: bool) -> int:
        if not think_active:
            return base
        return min(max(base + 768, base * 3), 8192)

    def _think_is_active(self, thinking: bool | None) -> bool:
        return bool(thinking) if self._is_thinking_model(self._model_id) else False

    @staticmethod
    def _is_thinking_model(model_id: str) -> bool:
        return "thinking" in (model_id or "").lower()

    def _resolve_enable_thinking(self, override: bool | None) -> bool | None:
        if not self._is_thinking_model(self._model_id):
            return None
        if override is not None:
            return override
        return self._llm_think_enabled

    def _apply_think_mode_to_text(self, text: str) -> str:
        if not self._is_thinking_model(self._model_id):
            return text
        if self._llm_think_enabled:
            return self._with_think_suffix(text)
        return self._with_no_think_suffix(text)

    def _apply_think_mode_to_messages(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        if not self._is_thinking_model(self._model_id):
            return messages
        last_user = max((i for i, m in enumerate(messages) if m.role == "user"), default=-1)
        if last_user < 0:
            return messages
        msg = messages[last_user]
        new_content = (
            self._with_think_suffix(msg.content)
            if self._llm_think_enabled
            else self._with_no_think_suffix(msg.content)
        )
        if new_content == msg.content:
            return messages
        patched = list(messages)
        patched[last_user] = ChatMessage(role=msg.role, content=new_content)
        return patched

    @staticmethod
    def _with_no_think_suffix(text: str) -> str:
        body = (text or "").rstrip()
        if not body or "/no_think" in body or "/think" in body:
            return body
        return f"{body} /no_think"

    @staticmethod
    def _with_think_suffix(text: str) -> str:
        body = (text or "").rstrip()
        if not body or "/think" in body or "/no_think" in body:
            return body
        return f"{body} /think"

    @staticmethod
    def _build_chat_prompt(
        tokenizer,
        messages: list[ChatMessage],
        *,
        enable_thinking: bool | None = None,
    ) -> str:
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        try:
            return tokenizer.apply_chat_template(msg_dicts, **template_kwargs)
        except TypeError:
            if enable_thinking is not None:
                try:
                    return tokenizer.apply_chat_template(
                        msg_dicts,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    pass
        except Exception:
            pass
        # Fallback: simple concatenation for tokenizers without chat template
        parts: list[str] = []
        for m in messages:
            if m.role == "system":
                parts.append(f"<|system|>\n{m.content}</s>")
            elif m.role == "user":
                parts.append(f"<|user|>\n{m.content}</s>")
            elif m.role == "assistant":
                parts.append(f"<|assistant|>\n{m.content}</s>")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    @staticmethod
    def _format_response(text: str, model_name: str) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            id=f"chatcmpl-{secrets.token_hex(12)}",
            created=int(time.time()),
            model=model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
            usage={},
        )
