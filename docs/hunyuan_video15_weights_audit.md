# HunyuanVideo-1.5 Weights Audit

Audit date: 2026-05-21. Source: [hunyuanvideo-community/hunyuanvideo-15](https://huggingface.co/collections/hunyuanvideo-community/hunyuanvideo-15).

## Diffusers bundle layout (all community checkpoints)

```
guider/
scheduler/
text_encoder/          # Qwen2_5_VLTextModel
text_encoder_2/        # T5EncoderModel (ByT5 glyph)
tokenizer/
tokenizer_2/
transformer/           # HunyuanVideo15Transformer3DModel
vae/                   # AutoencoderKLHunyuanVideo15
model_index.json
```

Diffusers version: `0.36.0.dev0`. Pipeline class: `HunyuanVideo15Pipeline`.

## Available checkpoints

| Model ID | Task | Resolution | Notes |
|----------|------|------------|-------|
| HunyuanVideo-1.5-Diffusers-480p_t2v | T2V | 480p | Base T2V |
| HunyuanVideo-1.5-Diffusers-720p_t2v | T2V | 720p | Higher-res T2V |
| HunyuanVideo-1.5-Diffusers-480p_t2v_distilled | T2V | 480p | CFG-distilled |
| HunyuanVideo-1.5-Diffusers-480p_i2v | I2V | 480p | Base I2V (`in_channels` differs) |
| HunyuanVideo-1.5-Diffusers-720p_i2v | I2V | 720p | Higher-res I2V |
| HunyuanVideo-1.5-Diffusers-480p_i2v_distilled | I2V | 480p | CFG-distilled |
| HunyuanVideo-1.5-Diffusers-720p_i2v_distilled | I2V | 720p | CFG-distilled |
| HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled | I2V | 480p | 8/12-step step-distill |

## SR (1080p super-resolution)

- **Not in official hunyuanvideo-community collection** (as of audit date).
- Community checkpoint: `weizhou03/HunyuanVideo-1.5-Diffusers-1080p-2SR` (third-party).
- Tencent README open-source plan lists SR weights as partially unreleased.
- DanQing registers SR as optional `hunyuan-video-1.5-1080p-sr`; pipeline fails loud if bundle missing.

## Transformer config (480p T2V reference)

- `in_channels`: 65 (32 noise + 32 cond + 1 mask)
- `out_channels`: 32
- `num_layers`: 54, `num_refiner_layers`: 2
- `num_attention_heads`: 16, `attention_head_dim`: 128 → inner_dim 2048
- `text_embed_dim`: 3584 (Qwen2.5-VL), `text_embed_2_dim`: 1472 (ByT5)
- `task_type`: `t2v` | `i2v`

## VAE config

- `latent_channels`: 32
- `spatial_compression_ratio`: 16 → `vae_scale` 16
- `temporal_compression_ratio`: 4 → `temporal_vae_scale` 4
- `scaling_factor`: 1.03682

## MLX status

- No `mlx-community` video weights.
- Implementation: native MLX in `backend/engine/families/hunyuan/` (same pattern as CogVideoX).

## Recommended install paths (DanQing)

```
models/Video/hunyuan-video-1.5-480p-t2v/     # diffusers snapshot
models/Video/hunyuan-video-1.5-480p-i2v/
models/Video/hunyuan-video-1.5-i2v-step-distill/
models/Video/hunyuan-video-1.5-1080p-sr/     # optional community SR
```
