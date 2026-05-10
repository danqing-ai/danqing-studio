"""SeedVR2 超分编排 — 与 ``ImagePipeline.run_upscale`` 对齐的正式管线类。

将「扁平 bundle 加载 + 3D VAE + MM-DiT 单步去噪」收拢为单一编排层，替代
``nn.Module`` 外壳 ``SeedVR2`` + ``SeedVR2Initializer``，避免双入口与隐式 ``nn.Module`` 状态。
数值子模块仍位于 ``seedvr2.runtime``（DiT / VAE / 权重映射器）。
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from backend.engine.seedvr2.config import ModelConfig
from backend.engine.seedvr2.runtime.callbacks.callback_registry import CallbackRegistry
from backend.engine.seedvr2.runtime.common.config.config import Config
from backend.engine.seedvr2.runtime.common.vae.tiling_config import TilingConfig
from backend.engine.seedvr2.runtime.common.vae.vae_util import VAEUtil
from backend.engine.seedvr2.runtime.common.weights.loading.weight_applier import WeightApplier
from backend.engine.seedvr2.runtime.common.weights.loading.weight_loader import WeightLoader
from backend.engine.seedvr2.runtime.dit import SeedVR2Transformer
from backend.engine.seedvr2.runtime.seedvr2_pkg.latent_creator.seedvr2_latent_creator import (
    SeedVR2LatentCreator,
)
from backend.engine.seedvr2.runtime.seedvr2_pkg.variants.upscale.seedvr2_util import SeedVR2Util
from backend.engine.seedvr2.runtime.seedvr2_pkg.weights.seedvr2_weight_definition import (
    SeedVR2WeightDefinition,
)
from backend.engine.seedvr2.runtime.utils.generated_image import GeneratedImage
from backend.engine.seedvr2.runtime.utils.image_util import ImageUtil
from backend.engine.seedvr2.runtime.utils.metadata_reader import MetadataReader
from backend.engine.seedvr2.runtime.utils.scale_factor import ScaleFactor
from backend.engine.seedvr2.runtime.vae3d import SeedVR2VAE
from backend.engine.seedvr2.text_encoders import SeedVR2PositiveEmbeddings


class SeedVR2UpscalePipeline:
    """从 bundle 构建并执行 SeedVR2 超分（Studio 唯一热路径）。"""

    __slots__ = (
        "model_config",
        "vae",
        "transformer",
        "tiling_config",
        "callbacks",
        "bits",
    )

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        vae: SeedVR2VAE,
        transformer: SeedVR2Transformer,
        tiling_config: TilingConfig,
        callbacks: CallbackRegistry,
        bits: int | None,
    ) -> None:
        self.model_config = model_config
        self.vae = vae
        self.transformer = transformer
        self.tiling_config = tiling_config
        self.callbacks = callbacks
        self.bits = bits

    @classmethod
    def from_bundle(
        cls,
        bundle_path: str | Path,
        model_config: ModelConfig,
        *,
        quantize: int | None = None,
    ) -> SeedVR2UpscalePipeline:
        path = str(bundle_path)
        weight_definition = SeedVR2WeightDefinition.resolve(model_config)
        weights = WeightLoader.load(weight_definition=weight_definition, model_path=path)
        vae = SeedVR2VAE()
        transformer = SeedVR2Transformer(**(model_config.transformer_overrides or {}))
        bits = WeightApplier.apply_and_quantize(
            weights=weights,
            models={"transformer": transformer, "vae": vae},
            quantize_arg=quantize,
            weight_definition=weight_definition,
        )
        return cls(
            model_config=model_config,
            vae=vae,
            transformer=transformer,
            tiling_config=TilingConfig(),
            callbacks=CallbackRegistry(),
            bits=bits,
        )

    def generate_image(
        self,
        seed: int,
        image_path: str | Path,
        resolution: int | ScaleFactor,
        softness: float = 0.0,
    ) -> GeneratedImage:
        processed_image, true_height, true_width = SeedVR2Util.preprocess_image(
            image_path=image_path,
            resolution=resolution,
            softness=softness,
        )

        config = Config(
            width=true_width,
            height=true_height,
            guidance=1.0,
            num_inference_steps=1,
            image_path=image_path,
            scheduler="seedvr2_euler",
            model_config=self.model_config,
        )

        initial_latent = VAEUtil.encode(
            vae=self.vae, image=processed_image, tiling_config=self.tiling_config
        )
        static_condition = SeedVR2LatentCreator.create_condition(encoded_latent=initial_latent)
        latents = SeedVR2LatentCreator.create_noise_latents(
            seed=seed,
            height=initial_latent.shape[-2],
            width=initial_latent.shape[-1],
        )

        txt_pos = SeedVR2PositiveEmbeddings.load()

        ctx = self.callbacks.start(seed=seed, prompt="", config=config)
        ctx.before_loop(latents)

        for t in config.time_steps:
            model_input = mx.concatenate([latents, static_condition], axis=1)

            noise = self.transformer(
                txt=txt_pos,
                vid=model_input,
                timestep=config.scheduler.timesteps[t],
            )

            latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)

            ctx.in_loop(t, latents)

            mx.eval(latents)

        ctx.after_loop(latents)

        decoded = VAEUtil.decode(vae=self.vae, latent=latents, tiling_config=self.tiling_config)
        decoded = decoded[:, :, :true_height, :true_width]
        style = processed_image[:, :, :true_height, :true_width]
        decoded = SeedVR2Util.apply_color_correction(decoded, style)

        init_metadata = MetadataReader.read_all_metadata(image_path) if image_path else None

        return ImageUtil.to_image(
            seed=seed,
            prompt="",
            config=config,
            quantization=self.bits,
            decoded_latents=decoded,
            generation_time=config.time_steps.format_dict["elapsed"],
            init_metadata=init_metadata,
        )
