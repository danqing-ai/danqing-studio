from __future__ import annotations

"""SeedVR2 超分产物：结果容器 + 张量转 PIL（模型族内）。"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import numpy as np
import PIL.Image

from .weights_mlx import ModelConfig

if TYPE_CHECKING:
    from .job_mlx import SeedVR2UpscaleRuntime


@dataclass
class GeneratedImage:
    image: PIL.Image.Image
    model_config: ModelConfig
    seed: int
    prompt: str
    steps: int
    guidance: float | None
    precision: mx.Dtype
    quantization: int
    generation_time: float
    lora_paths: list[str] | None = None
    lora_scales: list[float] | None = None
    height: int | None = None
    width: int | None = None
    controlnet_image_path: str | Path | None = None
    controlnet_strength: float | None = None
    image_path: str | Path | None = None
    image_paths: list[str] | list[Path] | None = None
    image_strength: float | None = None
    masked_image_path: str | Path | None = None
    depth_image_path: str | Path | None = None
    redux_image_paths: list[str] | list[Path] | None = None
    redux_image_strengths: list[float] | None = None
    concept_heatmap: Any = None
    negative_prompt: str | None = None
    init_metadata: dict | None = None


class ImageUtil:
    @staticmethod
    def to_image(
        *,
        seed: int,
        prompt: str,
        runtime: SeedVR2UpscaleRuntime,
        quantization: int,
        decoded_latents: mx.array,
        generation_time: float,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        controlnet_image_path: str | Path | None = None,
        image_path: str | Path | None = None,
        image_paths: list[str] | list[Path] | None = None,
        redux_image_paths: list[str] | list[Path] | None = None,
        redux_image_strengths: list[float] | None = None,
        image_strength: float | None = None,
        masked_image_path: str | Path | None = None,
        depth_image_path: str | Path | None = None,
        concept_heatmap: Any = None,
        negative_prompt: str | None = None,
        init_metadata: dict | None = None,
    ) -> GeneratedImage:
        normalized = ImageUtil._denormalize(decoded_latents)
        normalized_numpy = ImageUtil._to_numpy(normalized)
        image = ImageUtil._numpy_to_pil(normalized_numpy)
        return GeneratedImage(
            image=image,
            model_config=runtime.model_config,
            seed=seed,
            steps=runtime.num_inference_steps,
            prompt=prompt,
            guidance=runtime.guidance,
            precision=runtime.precision,
            quantization=quantization,
            generation_time=generation_time,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            height=runtime.height,
            width=runtime.width,
            image_path=image_path,
            image_paths=image_paths,
            image_strength=image_strength,
            controlnet_image_path=controlnet_image_path,
            controlnet_strength=runtime.controlnet_strength,
            masked_image_path=masked_image_path,
            depth_image_path=depth_image_path,
            redux_image_paths=redux_image_paths,
            redux_image_strengths=redux_image_strengths,
            concept_heatmap=concept_heatmap,
            negative_prompt=negative_prompt,
            init_metadata=init_metadata,
        )

    @staticmethod
    def _denormalize(images: mx.array) -> mx.array:
        return mx.clip((images / 2 + 0.5), 0, 1)

    @staticmethod
    def _to_numpy(images: mx.array) -> np.ndarray:
        if len(images.shape) == 5:
            images = mx.squeeze(images, axis=2)
        images = mx.transpose(images, (0, 2, 3, 1))
        images = mx.array.astype(images, mx.float32)
        return np.array(images)

    @staticmethod
    def _numpy_to_pil(images: np.ndarray) -> PIL.Image.Image:
        images = (images * 255).round().astype("uint8")
        pil_images = [PIL.Image.fromarray(image) for image in images]
        return pil_images[0]
