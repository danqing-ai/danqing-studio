from __future__ import annotations

"""SeedVR2 超分：单次运行时上下文 + 编排管线（Studio 唯一热路径）。

- **注册表**决定模型 id、动作等；**``ModelConfig``**（``weights_mlx``）描述 3B/7B bundle 元数据。
- **``SeedVR2UpscaleRuntime``** 只描述本次调用的尺寸、步数、guidance、调度器。
"""

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from typing import TYPE_CHECKING, Any, Callable

import mlx.core as mx

if TYPE_CHECKING:
    from .result_mlx import GeneratedImage

from backend.engine.common.bundle_weights import WeightApplier
from backend.engine.common.scale_factor import ScaleFactor
from backend.engine.common.vae.mlx_tiling import TilingConfig, VAEUtil
from .dit_mlx import SeedVR2DiT
from .embed_mlx import SeedVR2PositiveEmbeddings
from .preprocess_mlx import SeedVR2LatentCreator, SeedVR2Util
from .schedule_mlx import SCHEDULER_REGISTRY, SeedVR2EulerScheduler
from .vae_mlx import SeedVR2VAE
from .weights_mlx import ModelConfig, load_flat_bundle

logger = logging.getLogger(__name__)


@dataclass
class SeedVR2UpscaleRuntime:
    """对齐画幅后的单次推理上下文；调度器按 ``scheduler_key`` 惰性构造。"""

    model_config: ModelConfig
    height: int
    width: int
    num_inference_steps: int = 1
    guidance: float = 1.0
    image_path: Path | str | None = None
    scheduler_key: str = "seedvr2_euler"

    _scheduler: SeedVR2EulerScheduler | None = field(default=None, repr=False)
    _time_steps: range | None = field(default=None, repr=False)

    @classmethod
    def from_aligned_hw(
        cls,
        *,
        model_config: ModelConfig,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance: float,
        image_path: Path | str | None,
        scheduler_key: str = "seedvr2_euler",
    ) -> SeedVR2UpscaleRuntime:
        if width % 16 != 0 or height % 16 != 0:
            logger.warning("Width and height should be multiples of 16; rounding down.")
        h = 16 * (height // 16)
        w = 16 * (width // 16)
        return cls(
            model_config=model_config,
            height=h,
            width=w,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
            image_path=image_path,
            scheduler_key=scheduler_key,
        )

    @property
    def precision(self) -> mx.Dtype:
        return ModelConfig.precision

    @property
    def num_train_steps(self) -> int | None:
        return self.model_config.num_train_steps

    @property
    def image_seq_len(self) -> int:
        return (self.height // 16) * (self.width // 16)

    @property
    def init_time_step(self) -> int:
        return 0

    @property
    def time_steps(self) -> range:
        if self._time_steps is None:
            self._time_steps = range(self.init_time_step, self.num_inference_steps)
        return self._time_steps

    @property
    def controlnet_strength(self) -> float | None:
        return None

    @property
    def scheduler(self) -> SeedVR2EulerScheduler:
        if self._scheduler is None:
            registered = SCHEDULER_REGISTRY.get(self.scheduler_key)
            if registered is None:
                raise NotImplementedError(
                    f"The scheduler {self.scheduler_key!r} is not available for SeedVR2. "
                    f"Supported: {sorted(SCHEDULER_REGISTRY.keys())}"
                )
            self._scheduler = registered(self)
            if hasattr(self._scheduler, "set_image_seq_len") and self.model_config.requires_sigma_shift:
                self._scheduler.set_image_seq_len(self.image_seq_len)
        return self._scheduler


class _UpscaleDenoiseCtx:
    __slots__ = ()

    def before_loop(self, latents, *, canny_image=None, depth_image=None) -> None:
        return None

    def in_loop(self, t: int, latents, time_steps=None) -> None:
        return None

    def after_loop(self, latents) -> None:
        return None


def _resolve_eval_fn(dit: Any) -> Callable[..., None]:
    fn = mx.eval
    dit_ctx = getattr(dit, "ctx", None)
    if dit_ctx is not None and hasattr(dit_ctx, "eval"):
        fn = dit_ctx.eval
    return fn


def _resolve_array_fn(dit: Any) -> Callable[..., Any]:
    fn = mx.array
    dit_ctx = getattr(dit, "ctx", None)
    if dit_ctx is not None and hasattr(dit_ctx, "array"):
        fn = dit_ctx.array
    return fn


def _resolve_seeded_randn_fn(dit: Any) -> Callable[..., Any] | None:
    dit_ctx = getattr(dit, "ctx", None)
    if dit_ctx is not None and hasattr(dit_ctx, "seeded_randn"):
        return dit_ctx.seeded_randn
    return None


class SeedVR2UpscalePipeline:
    """从 bundle 构建并执行 SeedVR2 超分。"""

    __slots__ = (
        "model_config",
        "vae",
        "dit",
        "tiling_config",
        "bits",
        "_bundle_path",
    )

    def __init__(
        self,
        *,
        model_config: ModelConfig,
        vae: SeedVR2VAE,
        dit: SeedVR2DiT,
        tiling_config: TilingConfig,
        bits: int | None,
        bundle_path: str | Path | None = None,
    ) -> None:
        self.model_config = model_config
        self.vae = vae
        self.dit = dit
        self.tiling_config = tiling_config
        self.bits = bits
        self._bundle_path = Path(bundle_path) if bundle_path is not None else None

    @classmethod
    def from_bundle(
        cls,
        bundle_path: str | Path,
        model_config: ModelConfig,
        *,
        quantize: int | None = None,
    ) -> SeedVR2UpscalePipeline:
        path = str(bundle_path)
        weights, weight_definition_cls = load_flat_bundle(path, model_config)
        vae = SeedVR2VAE()
        dit = SeedVR2DiT(**(model_config.transformer_overrides or {}))

        components = {c.name: c for c in weight_definition_cls.get_components()}
        WeightApplier.set_weights(weights, {"transformer": dit, "vae": vae}, components)

        from backend.engine.common.bundle_weights.resolution import QuantizationResolution
        stored_q = weights.meta_data.quantization_level
        bits, warning = QuantizationResolution.resolve(stored=stored_q, requested=quantize)
        if warning:
            print(f"⚠️  {warning}")
        if bits is not None and stored_q is None:
            dit.quantize_runtime(bits=bits)
            vae.quantize_runtime(bits=bits)

        dit.after_load_weights(bundle_root=path)
        return cls(
            model_config=model_config,
            vae=vae,
            dit=dit,
            tiling_config=TilingConfig(),
            bits=bits,
            bundle_path=path,
        )

    def generate_image(
        self,
        seed: int,
        image_path: str | Path,
        resolution: int | ScaleFactor,
        softness: float = 0.0,
    ) -> GeneratedImage:
        from .result_mlx import ImageUtil
        from backend.utils.image_metadata import MetadataReader

        processed_image, true_height, true_width = SeedVR2Util.preprocess_image(
            image_path=image_path,
            resolution=resolution,
            softness=softness,
        )

        runtime = SeedVR2UpscaleRuntime.from_aligned_hw(
            model_config=self.model_config,
            height=true_height,
            width=true_width,
            num_inference_steps=1,
            guidance=1.0,
            image_path=image_path,
            scheduler_key="seedvr2_euler",
        )

        initial_latent = VAEUtil.encode(
            vae=self.vae, image=processed_image, tiling_config=self.tiling_config
        )
        static_condition = SeedVR2LatentCreator.create_condition(encoded_latent=initial_latent)
        latents = SeedVR2LatentCreator.create_noise_latents(
            seed=seed,
            height=initial_latent.shape[-2],
            width=initial_latent.shape[-1],
            seeded_randn_fn=_resolve_seeded_randn_fn(self.dit),
        )

        txt_pos = SeedVR2PositiveEmbeddings.load(bundle_path=self._bundle_path)
        eval_fn = _resolve_eval_fn(self.dit)

        t0 = time.perf_counter()
        ctx = _UpscaleDenoiseCtx()
        ctx.before_loop(latents)

        for t in runtime.time_steps:
            model_input = mx.concatenate([latents, static_condition], axis=1)

            noise = self.dit(
                txt=txt_pos,
                vid=model_input,
                timestep=runtime.scheduler.timesteps[t],
            )

            latents = runtime.scheduler.step(noise=noise, timestep=t, latents=latents)

            ctx.in_loop(t, latents)

            eval_fn(latents)

        ctx.after_loop(latents)

        elapsed_s = time.perf_counter() - t0

        decoded = VAEUtil.decode(vae=self.vae, latent=latents, tiling_config=self.tiling_config)
        decoded = decoded[:, :, :true_height, :true_width]
        style = processed_image[:, :, :true_height, :true_width]
        decoded = SeedVR2Util.apply_color_correction(decoded, style)

        init_metadata = MetadataReader.read_all_metadata(image_path) if image_path else None

        return ImageUtil.to_image(
            seed=seed,
            prompt="",
            runtime=runtime,
            quantization=self.bits,
            decoded_latents=decoded,
            generation_time=elapsed_s,
            init_metadata=init_metadata,
        )


def expected_seedvr2_weight_files(model_key: str) -> tuple[str, ...]:
    if "7b" in model_key.lower():
        return ("seedvr2_ema_7b_fp16.safetensors", "ema_vae_fp16.safetensors")
    return ("seedvr2_ema_3b_fp16.safetensors", "ema_vae_fp16.safetensors")


def validate_seedvr2_bundle(bundle_path: Path, model_key: str) -> None:
    missing = [n for n in expected_seedvr2_weight_files(model_key) if not (bundle_path / n).is_file()]
    if missing:
        raise RuntimeError(
            f"SeedVR2 bundle at {bundle_path} is missing weight file(s): {missing}. "
            "Expected flat directory with `ema_vae_fp16.safetensors` plus "
            "`seedvr2_ema_7b_fp16.safetensors` or `seedvr2_ema_3b_fp16.safetensors` "
            "(see registry `local_path`, e.g. models/Upscaler/seedvr2-7b-fp16)."
        )


def run_seedvr2_upscale(
    *,
    bundle_path: Path,
    model_key: str,
    source_image: Path,
    scale: int,
    softness: float,
    seed: int | None,
    output_png: Path,
    on_log: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """执行 SeedVR2 超分并写出 PNG。由 ``ImageUpscalePipeline`` 在 MLX 路径下调用。"""
    validate_seedvr2_bundle(bundle_path, model_key)

    if scale not in (2, 4):
        raise RuntimeError(f"SeedVR2 upscale scale must be 2 or 4, got {scale!r}")
    if not source_image.is_file():
        raise RuntimeError(f"SeedVR2 upscale source image not found: {source_image}")

    if "7b" in model_key.lower():
        model_config = ModelConfig.seedvr2_7b()
    else:
        model_config = ModelConfig.seedvr2_3b()

    pipeline = SeedVR2UpscalePipeline.from_bundle(bundle_path, model_config)
    resolution = ScaleFactor.parse(f"{int(scale)}x")
    soft = max(0.0, min(1.0, float(softness)))
    sd = int(seed) if seed is not None else random.randint(0, 2 ** 31 - 1)

    if on_log:
        on_log(
            "info",
            " ".join(
                [
                    "seedvr2_upscale backend=backend.engine.families.seedvr2.job_mlx",
                    f"bundle={bundle_path}",
                    f"model_key={model_key}",
                    f"resolution={resolution}",
                    f"softness={soft}",
                    f"seed={sd}",
                ]
            ),
        )

    generated = pipeline.generate_image(
        seed=sd,
        image_path=source_image,
        resolution=resolution,
        softness=soft,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    generated.image.save(str(output_png))

    return {
        "upscale_backend": "backend.engine.families.seedvr2.job_mlx",
        "seed": sd,
        "softness": soft,
        "scale": int(scale),
        "reference_model_name": getattr(model_config, "model_name", ""),
    }
