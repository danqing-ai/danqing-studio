"""OpenAI-style multimodal chat message helpers."""

from __future__ import annotations

import base64
import re
import tempfile
from pathlib import Path
from typing import Any
from backend.core.contracts import ChatMessage
from backend.persistence.asset_store import SQLiteAssetStore

_ASSET_URL_RE = re.compile(
    r"^(?:https?://[^/]+)?/api/assets/([^/?#]+)(?:/(file|thumbnail))?$"
)


def message_has_image(message: ChatMessage) -> bool:
    return any(part.get("type") == "image_url" for part in _content_parts(message.content))


def messages_have_images(messages: list[ChatMessage]) -> bool:
    return any(message_has_image(m) for m in messages)


def flatten_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    chunks: list[str] = []
    for part in content:
        if part.get("type") == "text":
            text = str(part.get("text") or "").strip()
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def flatten_messages_for_text_llm(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Collapse multipart text-only messages to plain strings for mlx-lm."""
    out: list[ChatMessage] = []
    for msg in messages:
        if isinstance(msg.content, str):
            out.append(msg)
            continue
        if message_has_image(msg):
            raise ValueError("Text LLM messages must not contain image_url parts")
        out.append(ChatMessage(role=msg.role, content=flatten_text_content(msg.content)))
    return out


def extract_vision_instruction(messages: list[ChatMessage]) -> str:
    """Build VLM instruction from system + user text parts (OpenAI messages → prompt)."""
    system_parts: list[str] = []
    user_parts: list[str] = []
    for msg in messages:
        text = flatten_text_content(msg.content).strip()
        if not text:
            continue
        if msg.role == "system":
            system_parts.append(text)
        elif msg.role == "user":
            user_parts.append(text)
    chunks: list[str] = []
    if system_parts:
        chunks.append("\n\n".join(system_parts))
    if user_parts:
        chunks.append("\n\n".join(user_parts))
    instruction = "\n\n".join(chunks).strip()
    if not instruction:
        raise ValueError("Vision request requires text in system or user messages")
    return instruction


def _content_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    parts: list[dict[str, Any]] = []
    for part in content or []:
        if isinstance(part, dict):
            parts.append(part)
            continue
        part_type = getattr(part, "type", None)
        if part_type == "text":
            parts.append({"type": "text", "text": getattr(part, "text", "")})
        elif part_type == "image_url":
            image_url = getattr(part, "image_url", None)
            url = getattr(image_url, "url", "") if image_url is not None else ""
            parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


def iter_image_urls(messages: list[ChatMessage]) -> list[str]:
    urls: list[str] = []
    for msg in messages:
        for part in _content_parts(msg.content):
            if part.get("type") != "image_url":
                continue
            image_url = part.get("image_url") or {}
            url = str(image_url.get("url") or "").strip()
            if url:
                urls.append(url)
    return urls


def _decode_data_url(url: str) -> Path:
    if "," not in url:
        raise ValueError("Invalid data URL for image_url")
    header, payload = url.split(",", 1)
    if "base64" not in header:
        raise ValueError("image_url data URLs must be base64-encoded")
    raw = base64.b64decode(payload, validate=False)
    fd, tmp_name = tempfile.mkstemp(suffix=".img", prefix="dq_vlm_url_")
    import os

    os.close(fd)
    tmp_path = Path(tmp_name)
    tmp_path.write_bytes(raw)
    return tmp_path


def _parse_asset_ref(url: str) -> tuple[str, bool] | None:
    url = url.strip()
    if url.startswith("asset:"):
        return url[len("asset:") :].strip(), False
    m = _ASSET_URL_RE.match(url)
    if not m:
        return None
    return m.group(1), m.group(2) == "thumbnail"


def resolve_image_url(
    store: SQLiteAssetStore,
    url: str,
) -> tuple[Path, dict[str, Any] | None, bool]:
    """Resolve image_url to local path. Returns (path, asset_row|None, is_temp)."""
    url = url.strip()
    if url.startswith("data:"):
        return _decode_data_url(url), None, True

    asset_ref = _parse_asset_ref(url)
    if asset_ref is None:
        path = Path(url)
        if path.is_file():
            return path, None, False
        raise FileNotFoundError(f"Unsupported or missing image_url: {url}")

    asset_id, want_thumbnail = asset_ref
    row = store.get_asset_record(asset_id)
    if not row:
        raise FileNotFoundError(f"asset not found: {asset_id}")

    kind = str(row.get("kind") or "")
    image_path: Path | None = None
    if kind in ("image", "video"):
        try:
            if want_thumbnail or kind == "video":
                image_path = store.get_thumbnail_path(asset_id)
                if image_path is None and kind == "video":
                    image_path = store.get_file_path(asset_id)
            else:
                image_path = store.get_file_path(asset_id)
        except FileNotFoundError:
            image_path = None
        if image_path is None and kind == "image":
            image_path = store.get_file_path(asset_id)

    if image_path is None or not image_path.is_file():
        raise FileNotFoundError(f"asset has no readable image preview: {asset_id}")
    return image_path, row, False


def resolve_message_images(
    store: SQLiteAssetStore,
    messages: list[ChatMessage],
) -> list[tuple[Path, dict[str, Any] | None, bool]]:
    resolved: list[tuple[Path, dict[str, Any] | None, bool]] = []
    for url in iter_image_urls(messages):
        resolved.append(resolve_image_url(store, url))
    if not resolved:
        raise ValueError("Vision request requires at least one image_url in messages")
    return resolved
