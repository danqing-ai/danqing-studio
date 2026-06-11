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
from backend.core.model_registry import ModelRegistry
from backend.engine.llm.lyrics_sanitize import sanitize_lyrics_output
from backend.engine.llm.prompt_sanitize import prompt_enhance_quality_ok, sanitize_enhanced_prompt
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

DEFAULT_LLM_MODEL_ID = "qwen3-4b-thinking-2507"
DEFAULT_VLM_MODEL_ID = "qwen3-vl-4b-instruct"


def resolve_llm_model_id(settings: AppSettings, registry: ModelRegistry) -> str:
    preferred = (getattr(settings, "default_model_llm", "") or "").strip()
    if preferred:
        if registry.get(preferred):
            return preferred
        logger.warning("Unknown default_model_llm %r; using %s", preferred, DEFAULT_LLM_MODEL_ID)
    return DEFAULT_LLM_MODEL_ID


def resolve_vlm_model_id(settings: AppSettings, registry: ModelRegistry) -> str:
    preferred = (getattr(settings, "default_model_vlm", "") or "").strip()
    if preferred:
        if registry.get(preferred):
            return preferred
        logger.warning("Unknown default_model_vlm %r; using %s", preferred, DEFAULT_VLM_MODEL_ID)
    return DEFAULT_VLM_MODEL_ID

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
Match the user's language: Chinese input → Chinese output; English input → English output.
Keep it concise: one paragraph, at most ~120 Chinese characters or ~80 English words.
CRITICAL: Never repeat the same phrase or word. No filler loops.
Output ONLY the enhanced prompt text, without explanation or quotation marks."""

ENHANCE_AUDIO_BRIEF_SYSTEM_PROMPT = """You are a music producer writing briefs for AI music generation (ACE-Step).
Given a user's music idea, expand it into a clear, vivid description covering genre, mood, tempo feel, instrumentation,
vocal style, and emotional arc. Match the user's language when the input is Chinese.
Keep it concise: one short paragraph. Never repeat the same phrase or word. No filler loops.
Output ONLY the enhanced brief text, without explanation or quotation marks."""

LYRICS_SYSTEM_PROMPT = """You write singable lyrics for ACE-Step music generation.

Format:
- Section tags on their own line: [Intro], [Verse], [Verse 1], [Chorus], [Bridge], [Outro]
- 2–4 short lines per section; use only sections that fit a ~30–90s song
- Typical flow: [Verse 1] → [Chorus] → [Verse 2] → [Chorus] → [Outro] (skip optional sections when unnecessary)
- If the description asks for instrumental or no vocals: output only `[Instrumental]` and stop

Line rules:
- English: 4–10 words per line, one complete sung phrase
- Chinese: 5–12 characters per line, natural rhythm, each line self-contained
- Match the language and theme of the music description

Anti-loop (critical):
- Never repeat a word twice in a row
- Do not reuse the same line or filler hook ("la la", "oh oh") more than once
- After [Outro] lines, STOP — no extra sections or commentary

Output ONLY lyrics with section tags. No title, quotes, or explanation.

Example (English):
[Verse 1]
Walking down the empty street at dawn
City lights fading one by one

[Chorus]
We are the stars tonight
Shining through the endless sky

[Outro]
Fade into the quiet night

Example (Chinese):
[Verse 1]
清晨的风吹过旧街角
你的笑容还在心头绕

[Chorus]
我们是今夜的星光
照亮这片无尽的天

[Outro]
慢慢沉入安静的夜"""

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
    ):
        self._registry = model_registry
        self._path_resolver = path_resolver
        self._model_id = default_model_id
        self._vision_model_id = vision_model_id
        self._generation_lock = threading.Lock()

    def apply_model_settings(
        self,
        *,
        default_model_id: str | None = None,
        vision_model_id: str | None = None,
    ) -> None:
        if default_model_id:
            self._registry.require(default_model_id)
            self._model_id = default_model_id
        if vision_model_id:
            self._registry.require(vision_model_id)
            self._vision_model_id = vision_model_id

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

    def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Run a single-turn non-streaming chat completion (sync, for asyncio.to_thread)."""
        with self._generation_lock:
            model, tokenizer = self._load_model()
            try:
                prompt = self._build_chat_prompt(tokenizer, request.messages)
                result = mlx_lm.generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    verbose=False,
                    **self._generation_kwargs(request),
                )
                return self._format_response(result, request.model)
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
                prompt_text = self._build_chat_prompt(tokenizer, request.messages)
            except Exception:
                self._unload_model(model, tokenizer)
                raise

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
                        **self._generation_kwargs(request),
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

        attempts = (
            (0.65, 200),
            (0.45, 160),
            (0.35, 140),
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
            cleaned = sanitize_enhanced_prompt(result.choices[0].message.content)
            if prompt_enhance_quality_ok(cleaned):
                return EnhanceResponse(enhanced_prompt=cleaned)
            last_clean = cleaned

        fallback = sanitize_enhanced_prompt(raw_prompt)
        if prompt_enhance_quality_ok(last_clean):
            return EnhanceResponse(enhanced_prompt=last_clean)
        return EnhanceResponse(enhanced_prompt=fallback or last_clean)

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
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        lyric_lines = [ln for ln in lines if not ln.startswith("[")]
        if len(lyric_lines) < 4:
            return False
        for ln in lyric_lines:
            words = ln.split()
            if len(words) >= 6 and len(set(w.lower() for w in words)) / len(words) < 0.35:
                return False
        return True

    def generate_lyrics(self, prompt: str, style: str | None = None) -> str:
        """Generate ACE-Step formatted lyrics from a music description."""
        user_msg = f"Music description: {prompt}"
        if style:
            user_msg += f"\nStyle/Genre: {style}"
        user_msg += (
            "\n\nWrite ACE-Step lyrics with section tags. "
            "Match the description language. End with [Outro] (or [Instrumental] if no vocals) and stop."
        )

        attempts = (
            (0.65, 420),
            (0.5, 360),
        )
        last_raw = ""
        for temp, max_tokens in attempts:
            result = self.chat_completion(
                self._lyrics_chat_request(user_msg, temperature=temp, max_tokens=max_tokens),
            )
            last_raw = result.choices[0].message.content.strip()
            cleaned = sanitize_lyrics_output(last_raw)
            if self._lyrics_quality_ok(cleaned):
                return cleaned

        return sanitize_lyrics_output(last_raw)

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

    @staticmethod
    def _generation_kwargs(request: ChatCompletionRequest) -> dict[str, Any]:
        """Sampling kwargs for mlx-lm >= 0.31 (``sampler`` instead of ``temp``)."""
        return {
            "max_tokens": request.max_tokens or 512,
            "sampler": make_sampler(
                temp=request.temperature,
                top_p=request.top_p,
            ),
        }

    @staticmethod
    def _build_chat_prompt(tokenizer, messages: list[ChatMessage]) -> str:
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
        try:
            return tokenizer.apply_chat_template(
                msg_dicts,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
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
