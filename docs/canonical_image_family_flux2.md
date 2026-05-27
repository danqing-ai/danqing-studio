# Canonical Image Family Template — Flux2

Use **`backend/engine/families/flux2/`** as the reference when adding a new **Shape A (DiT + ImagePipeline)** image model.

## File layout

| File | Role |
|------|------|
| `transformer.py` | Public stem — re-exports platform implementation |
| `transformer_mlx.py` | DiT topology, `TransformerBase` subclass, `_param_map` |
| `weights.py` | `remap_<family>_weights` (readable `str.replace` loops) |
| `text_encoder.py` | Thin wrapper → shared `common/text_encoders/qwen3_mlx` |
| `text_encoder_mlx.py` | Backend-specific load/encode hooks (if needed) |
| `vae_mlx.py` | Family VAE codec when `_class_name` is non-generic |
| `lora_mlx.py` | LoRA merge (register in `_IMAGE_LORA_MERGE`) |

**Avoid:** monolithic `transformer.py` (see z_image — scheduled for split), upstream `mlx/` subtrees (see heartmula anti-pattern).

## Registration (five touch points)

1. `default_config/models_registry.json` — `family`, `engine`, `actions`, `parameters`, `backends`, bilingual `name`/`description`
2. `make sync-models-registry`
3. `backend/engine/config/model_configs.py` — dataclass + `FAMILY_CONFIG_MAP` + `IMAGE_FAMILY_REUSE_CONTRACT`
4. `backend/engine/_transformer_registry.py` — `_TRANSFORMER`, `_WEIGHT_REMAP`, `_TEXT_ENCODER`, optional `_IMAGE_LORA_MERGE`
5. `backend/engine/vae_codec_registry.py` — only if VAE `_class_name` is not generic `AutoencoderKL`

## Runtime semantics

Declare behavior on **`ModelConfig`**, not in `ImagePipeline`:

- `latent_noise_dtype`, `noise_sample_fp32`, `passes_guidance_in_kwargs`
- `z_image_noise_layout`, `skip_negative_when_structured_prompt`
- `encoder_step_kwargs` (e.g. qwen_image spatial kwargs)

`FamilyRuntimeContract` reads config flags only.

## Hooks (optional)

Use `TransformerBase` hooks for cross-cutting features:

- `after_load_weights` — LoRA / compile prep
- `prepare_conditioning` — ControlNet-style cond
- `before_denoise` — cache warm-up
- `refine_cfg_noise` / `forward_cfg` — CFG parity

Do **not** add `family ==` branches in `ImagePipeline`.

## Verify

```bash
python -m py_compile backend/engine/families/flux2/*.py
make verify-engine-stack
bin/danqing-generate --model flux2-klein-9b --prompt "test"
```

See also: `docs/engine_new_model_checklist.md`, `docs/dual_platform_architecture.md`.
