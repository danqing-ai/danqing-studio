"""Optional MLX-VLM helpers for canvas node vision description."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

VISION_DESCRIBE_PROMPT = """You are a creative studio assistant. Describe this visual asset in 2-4 concise sentences
for an artist's canvas board note. Cover subject, style, lighting, composition, and one concrete next-step suggestion.
Match Chinese if the scene or any overlaid text is primarily Chinese. Output ONLY the note text."""

IMAGE_TO_PROMPT_INSTRUCTION = """You are an expert AI art prompt engineer. Analyze this image and write a detailed English
prompt suitable for text-to-image generation models (Flux, SDXL, etc.).
Include subject, composition, lighting, color palette, art style, mood, camera angle, and fine details.
Output ONLY the prompt text — no quotes, headings, or explanation."""

VIDEO_FRAME_TO_PROMPT_INSTRUCTION = """You are an expert AI video prompt engineer. This image is a keyframe or reference
for video generation. Write a detailed English prompt describing the scene plus implied motion, camera movement, and
temporal atmosphere suitable for image-to-video models.
Output ONLY the prompt text — no quotes, headings, or explanation."""


def mlx_vlm_importable() -> bool:
    try:
        import mlx_vlm  # noqa: F401

        return True
    except ImportError:
        return False


def vision_weights_ready(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    return (
        (model_dir / "model.safetensors").is_file()
        or any(f.suffix == ".safetensors" for f in model_dir.rglob("*") if f.is_file())
        or any(f.suffix == ".bin" for f in model_dir.rglob("*") if f.is_file())
    )


def analyze_image_file(
    image_path: Path,
    model_dir: Path,
    *,
    instruction: str,
    metadata_hint: str = "",
    max_tokens: int = 256,
    temperature: float = 0.4,
) -> str:
    """Run a single-image VLM task with a custom instruction. Raises RuntimeError on failure."""
    if not image_path.is_file():
        raise RuntimeError(f"Vision analyze image not found: {image_path}")
    if not vision_weights_ready(model_dir):
        raise RuntimeError(f"Vision model weights not found under {model_dir}")

    try:
        import mlx.core as mx
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config
    except ImportError as exc:
        raise RuntimeError(
            "mlx-vlm is not installed. Install with: pip install mlx-vlm"
        ) from exc

    prompt = instruction.strip()
    if metadata_hint.strip():
        prompt += f"\n\nOptional metadata context:\n{metadata_hint.strip()}"

    model, processor = load(str(model_dir))
    config = load_config(str(model_dir))
    images = [str(image_path)]
    formatted = apply_chat_template(processor, config, prompt, num_images=len(images))
    output = generate(
        model,
        processor,
        formatted,
        images,
        max_tokens=max_tokens,
        temperature=temperature,
        verbose=False,
    )
    del model
    del processor
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass

    text = _coerce_vlm_output_text(output)
    if not text:
        raise RuntimeError("Vision model returned empty output")
    return text


def _coerce_vlm_output_text(output: Any) -> str:
    """mlx-vlm 0.3+ returns GenerationResult; older versions may return str."""
    if isinstance(output, str):
        return output.strip()
    text = getattr(output, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    draft = getattr(output, "draft_text", None)
    if isinstance(draft, str) and draft.strip():
        return draft.strip()
    return str(output).strip()


def describe_image_file(
    image_path: Path,
    model_dir: Path,
    *,
    metadata_hint: str = "",
    max_tokens: int = 256,
) -> str:
    """Canvas node note via VLM."""
    return analyze_image_file(
        image_path,
        model_dir,
        instruction=VISION_DESCRIBE_PROMPT,
        metadata_hint=metadata_hint,
        max_tokens=max_tokens,
    )
