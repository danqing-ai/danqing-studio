"""Catalog schema v3 constants and field sets."""

from __future__ import annotations

SCHEMA_VERSION_V3 = 3

# Registry ``parameters`` keys consumed by engine / pipeline (not Composer UI schema).
ENGINE_PARAMETER_KEYS = frozenset(
    {
        "vae_scale",
        "text_encoder_out_layers",
        "latent_noise_packed",
        "latent_noise_dtype",
        "noise_sample_fp32",
        "vae_preview_warmup",
        "encoder_type",
        "enable_thinking",
        "step_distill",
        "requires_sigma_shift",
        "supports_guidance",
        "use_mlx_cfg_fusion",
    }
)

CATALOG_MODEL_KEYS = frozenset(
    {
        "name",
        "description",
        "category",
        "media",
        "engine",
        "commercial_use_allowed",
        "recommended",
        "successor",
        "distilled_from",
        "distilled_variant",
        "source",
        "type",
        "base_model",
        "nsfw",
        "metadata",
    }
)

DISTRIBUTION_KEYS = frozenset({"versions", "dependencies"})
