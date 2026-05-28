# Dual-Platform Architecture (MLX / CUDA) — Redesign Plan

## 1. Motivation

### Current Pain Points

| Issue | Scope |
|-------|-------|
| Some DiT transformers import `mlx` / `torch` at top-level in hot paths | flux1, flux2, fibo, cogvideox |
| Qwen inner modules (`QwenTransformer`, ~11 sub-classes) are pure `nn.Module` with **zero RuntimeContext awareness** | qwen |
| Scattered `_mlx.py` files live under individual family directories with no unified pattern | `z_image/text_encoder_mlx.py`, `families/cogvideox/vae_mlx.py` |
| `seedvr2/` is entirely MLX-only (8 `*_mlx.py` modules under `families/seedvr2/`) | seedvr2 |
| Text encoders diverge: `z_image/text_encoder.py` is dual-backend, `flux2/text_encoder.py` accepts `ctx` but ignores it (MLX-only) | flux2, z_image |
| Bundle weight loading (`common/bundle_weights/`) is MLX-only with no CUDA path | bundle_weights |

### Root Cause

The **same functionality is reimplemented multiple times** instead of being shared or abstracted:
- `SelfAttention` exists in `common/attention.py` (ctx-based) but `qwen/transformer.py` reimplements `QwenAttention(nn.Module)`
- `RMSNorm` exists in `common/norm.py` (ctx-based) but `qwen/transformer.py` reimplements `QwenTransformerRMSNorm(nn.Module)`
- `TimestepEmbedding` exists in `common/embeddings.py` (ctx-based) but `qwen` reimplements its own

---

## 2. Design Principles

### 2.1 Two-Tier Strategy

```
┌────────────────────────────────────────────────────────────────┐
│  Tier 1: RuntimeContext                                        │
│  For all DiT Transformers + Pipeline hot paths                 │
│  ctx.Linear, ctx.reshape, ctx.matmul, ctx.attention, ...       │
│  ZERO top-level import mlx / import torch                      │
├────────────────────────────────────────────────────────────────┤
│  Tier 2: Go-Style xxx.py / xxx_mlx.py / xxx_cuda.py            │
│  For components where RuntimeContext cannot express the logic   │
│  xxx.py      = shared logic (the body)                         │
│  xxx_mlx.py  = backend-differentiated hooks only               │
│  xxx_cuda.py = backend-differentiated hooks only               │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 What RuntimeContext Already Covers

RuntimeContext (`backend/engine/runtime/_base.py`) defines 40+ methods covering:

| Category | Methods |
|----------|---------|
| Module factories | `Linear`, `LayerNorm`, `RMSNorm`, `GroupNorm`, `SiLU`, `GELU`, `Embedding`, `Conv1d/2d/3d`, `Sequential`, `ModuleList` |
| Tensor ops | `reshape`, `permute`, `concat`, `stack`, `matmul`, `softmax`, `einsum`, `sin`, `cos`, `exp`, `log`, `sqrt`, `rsqrt`, `silu`, `gelu`, `where`, `broadcast_to`, `repeat`, `meshgrid`, `linspace`, `split`, `flip`, ... |
| Tensor creation | `zeros`, `ones`, `full`, `arange`, `randn`, `array`, `zeros_like`, `ones_like` |
| Advanced | `attention` (SDPA), `interpolate`, `dequantize`, `compile`, `eval` |
| Memory | `clear_cache`, `active_memory_gb` |
| Weight I/O | `load_weights`, `save_weights` |
| DTypes | `float32`, `float16`, `bfloat16`, `int32`, `bool_` |

**All current direct `mlx.*` calls in transformers can be replaced with `ctx.*` — no new RuntimeContext methods are needed.**

| Current Leak | RuntimeContext Equivalent | Leaking Models |
|-------------|--------------------------|----------------|
| `mx.full(shape, val)` | `ctx.full(shape, val)` | flux1, flux2 |
| `mx.repeat(x, n, axis)` | `ctx.repeat(x, n, axis)` | flux1 |
| `mx.reshape(x, shape)` | `ctx.reshape(x, shape)` | flux1, flux2, fibo, cogvideox |
| `mx.array(data)` | `ctx.array(data)` | fibo |
| `mx.sin(x)`, `mx.cos(x)` | `ctx.sin(x)`, `ctx.cos(x)` | cogvideox |
| `nn.Linear(...)`, `nn.RMSNorm(...)` | `ctx.Linear(...)`, `ctx.RMSNorm(...)` | qwen (worst) |

### 2.3 What RuntimeContext CANNOT Express (Go-Style Targets)

RuntimeContext is impractical for:

1. **Full model weight loading** — `mx.load` vs `safetensors.torch.load_file` differ in API and return format
2. **Deep layout differences** — Conv3d NHWC (MLX) vs NCHW (CUDA). Forcing RuntimeContext here means every method becomes `if ctx.backend == "mlx": ...` — defeating the purpose of the abstraction
3. **MLX-specific memory optimizations** — Tiling strategy depends on Metal memory model
4. **HF transformers integration** — `AutoModel.from_pretrained` is torch-only

### 2.4 Go-Style Pattern Convention

```
xxx.py       = shared logic (class body, algorithms, data flow)
xxx_mlx.py   = backend-differentiated hooks only (imported BY xxx.py)
xxx_cuda.py  = backend-differentiated hooks only (imported BY xxx.py)

RULE:  xxx.py is NEVER an empty shell.
       xxx_mlx.py / xxx_cuda.py override ONLY the parts that differ.
```

**Shallow-difference example** (Text Encoder: ~300 lines in `xxx.py`, ~30 lines each in `_mlx.py` / `_cuda.py`):
```
common/text_encoders/
├── qwen3.py          # Body: Qwen3Encoder class, all layers, forward, encode
├── qwen3_mlx.py      # Hook: load_weights() using mx.load
└── qwen3_cuda.py     # Hook: load_weights() using safetensors.torch
```

**Deep-difference example** (CogVideoX VAE: family-local Go-style layout):
```
families/cogvideox/
├── vae.py       # Decode entry (latents → PIL; no top-level mlx/torch)
├── vae_mlx.py   # Full MLX NHWC decoder
└── vae_cuda.py  # Full CUDA NCHW decoder (optional / when added)
```

---

## 3. New Directory Structure

```
backend/engine/
│
├── runtime/                              # UNCHANGED
│   ├── __init__.py
│   ├── _base.py                          # RuntimeContext ABC
│   ├── mlx.py                            # MLXContext
│   └── cuda.py                           # CudaContext
│
├── common/                               # REFACTORED shared layer
│   ├── __init__.py
│   ├── _base.py                          # TransformerBase + ImageTransformer
│   │
│   │── [ctx-based components — zero backend imports]
│   ├── attention.py                      # SelfAttention, CrossAttention, TemporalAttention
│   ├── norm.py                           # RMSNorm, LayerNorm, GroupNorm, AdaLayerNorm
│   ├── embeddings.py                     # TimestepEmbedding, RoPE2D, RoPE3D, PatchEmbed2D/3D
│   ├── activations.py                    # silu, gelu wrappers
│   ├── schedulers.py                     # FlowMatchEulerScheduler etc.
│   ├── cache.py                          # ModelCache
│   ├── pipeline.py                       # Pipeline helper
│   ├── scale_factor.py                   # Scale factor helpers
│   │
│   ├── vae/                              # Go-style: standard VAE + model-specific VAEs
│   │   ├── __init__.py
│   │   ├── autoencoder.py                # Standard SD VAE (ctx-based, merged from current decoder.py + encoder.py)
│   │   ├── common.py                     # Latent scale/shift, pixel ↔ tensor, PIL post-processing
│   │   ├── remap.py                      # Weight key remap (current weight_remap.py)
│   │   │
│   │   ├── tiling.py                     # Interface + dispatch
│   │   ├── tiling_mlx.py                 # MLX tiling (current mlx_tiling.py)
│   │   ├── tiling_cuda.py                # CUDA tiling (no-op) [NEW]
│   │   │
│   │   │   # CogVideoX 3D VAE: implemented under families/cogvideox/ (vae.py + vae_mlx.py)
│   │   │
│   │   ├── qwen_image_decoder.py         # Interface + dispatch
│   │   ├── qwen_image_decoder_mlx.py     # MLX decoder (consolidated from qwen_image/ 9 files)
│   │   └── qwen_image_decoder_cuda.py    # CUDA decoder [NEW]
│   │
│   ├── text_encoders/                    # Go-style: backend-differentiated hooks only
│   │   ├── __init__.py
│   │   ├── factory.py                    # get_encoder(encoder_type, ctx, ...) → encoder
│   │   │
│   │   ├── qwen3.py                      # Body: Qwen3Encoder + all sub-layers (~300 lines, ctx-driven)
│   │   ├── qwen3_mlx.py                  # Hook: load_weights() using mx.load (~30 lines)
│   │   ├── qwen3_cuda.py                 # Hook: load_weights() using safetensors.torch (~30 lines)
│   │   │
│   │   ├── clip.py                       # Body: CLIP encoder
│   │   ├── clip_mlx.py                   # Hook: load_weights
│   │   ├── clip_cuda.py                  # Hook: load_weights
│   │   │
│   │   ├── t5.py                         # Body: T5 encoder
│   │   ├── t5_mlx.py                     # Hook: load_weights
│   │   └── t5_cuda.py                    # Hook: load_weights
│   │
│   └── bundle_weights/                   # Go-style: weight I/O
│       ├── __init__.py
│       ├── base.py                       # Interfaces + definitions (current definitions.py + loaded_weights.py)
│       ├── loader.py                     # Dispatch + shared logic
│       ├── loader_mlx.py                 # mx.load / mx.save_safetensors (current loader.py + applier.py)
│       ├── loader_cuda.py                # safetensors.torch [NEW]
│       ├── remap.py                      # Generic key remap utilities (current bundle_weight_mapping.py)
│       └── resolution.py                # Path resolution
│
├── flux1/                                # FIX: remove top-level mlx import
│   ├── __init__.py
│   ├── transformer.py                    # Flux1Transformer — replace mx.* with ctx.* in forward()
│   └── weights.py                        # remap_flux1_weights
│
├── flux2/                                # FIX: remove top-level mlx import
│   ├── __init__.py
│   ├── transformer.py                    # Flux2Transformer — replace mx.* with ctx.*
│   ├── modules.py                        # Flux2Modulation etc. (ctx-ified) [extracted from transformer.py]
│   └── weights.py                        # remap_flux2_weights
│
├── z_image/                              # ALREADY CLEAN (remove text_encoder_mlx.py → text_encoders/)
│   ├── __init__.py
│   ├── transformer.py                    # ZImageTransformer (already ctx-only)
│   └── weights.py                        # remap_zimage_weights
│
├── qwen/                                 # REWRITE: full ctx migration
│   ├── __init__.py
│   ├── transformer.py                    # QwenImageTransformer(TransformerBase) — thin wrapper
│   ├── modules.py                        # All internal modules (ALL accept ctx: RuntimeContext):
│   │                                     #   QwenTransformer, QwenTransformerBlock,
│   │                                     #   QwenAttention, QwenFeedForward,
│   │                                     #   AdaLayerNormContinuous
│   ├── embeddings.py                     # QwenTimesteps, QwenTimestepEmbedding, QwenTimeTextEmbed (all ctx)
│   └── weights.py                        # remap_qwen_transformer_weights
│
├── fibo/                                 # FIX: remove top-level mlx import
│   ├── __init__.py
│   ├── transformer.py                    # FIBOTransformer — replace mx.* with ctx.*
│   └── weights.py                        # (if needed)
│
├── ltx/                                  # ALREADY CLEAN
│   ├── __init__.py
│   ├── transformer.py                    # LTXTransformer (already ctx-only)
│   └── weights.py                        # remap_ltx_weights
│
├── wan/                                  # ALREADY CLEAN
│   ├── __init__.py
│   ├── transformer.py                    # WanTransformer (already ctx-only)
│   └── weights.py                        # remap_wan_weights
│
├── cogvideox/                            # FIX: remove top-level mlx import
│   ├── __init__.py
│   ├── transformer.py                    # 对外入口（re-export）；实现 ``transformer_mlx.py``
│   ├── weights.py                        # remap_cogvideox_weights
│   ├── vae.py                            # VAE decode entry (latents → PIL)
│   └── vae_mlx.py                        # Full MLX NHWC decoder (+ optional vae_cuda.py)
│
├── seedvr2/                              # ctx-ify + Go-style VAE
│   ├── __init__.py
│   ├── pipeline.py                       # Main pipeline (ctx-ified)
│   ├── dit.py                            # DiT model (ctx-ified, was sv2_dit.py)
│   ├── embed.py                          # Embeddings (ctx-ified)
│   ├── schedule.py                       # Schedule (ctx-ified)
│   ├── preprocess.py                     # Preprocessing (ctx-ified)
│   ├── vae.py                            # VAE interface + dispatch
│   ├── vae_mlx.py                        # MLX VAE (was sv2_vae_net.py)
│   ├── vae_cuda.py                       # CUDA VAE [NEW]
│   ├── job.py                            # Job (ctx-ified)
│   ├── result.py                         # Result (ctx-ified)
│   ├── spec.py                           # Model spec (ctx-ified)
│   └── weights.py                        # Weight schema (ctx-ified)
│
├── config/                               # UNCHANGED
│   ├── __init__.py
│   └── model_configs.py                  # Model config dataclasses
│
├── _transformer_registry.py              # UNCHANGED (update _TEXT_ENCODER → factory.py)
│
├── image_pipeline.py                     # UNCHANGED
├── video_pipeline.py                     # UNCHANGED
├── image_upscale_pipeline.py             # ctx-ify
├── video_upscale_pipeline.py             # ctx-ify
│
├── danqing_image_engine.py               # UNCHANGED
├── danqing_video_engine.py               # UNCHANGED
├── danqing_audio_engine.py               # UNCHANGED
├── engine_registry.py                    # UNCHANGED
└── platform.py                           # UNCHANGED
```

---

## 4. Code Patterns

### 4.1 DiT Transformer (Tier 1 — RuntimeContext only)

Every transformer under `backend/engine/<family>/transformer.py` and all its sub-modules must follow:

```python
# ✅ CORRECT (z_image, ltx, wan already achieve this)
from backend.engine.common._base import TransformerBase
from backend.engine.runtime._base import RuntimeContext

class SomeTransformer(TransformerBase):
    def __init__(self, config, ctx: RuntimeContext):
        self.ctx = ctx
        nn = ctx                              # ctx.Linear, ctx.RMSNorm, ...
        self.attn = SelfAttention(dim, heads, ctx=ctx)
        self.norm = nn.RMSNorm(dim)
        self.proj = nn.Linear(dim, out_dim)

    def forward(self, latents, timestep, txt_embeds=None, sigmas=None):
        ctx = self.ctx
        x = ctx.reshape(latents, (B, N, C))   # All tensor ops via ctx
        x = ctx.matmul(x, w)
        x = ctx.silu(x)
        return self.proj(x)
```

```python
# ❌ FORBIDDEN
import mlx.core as mx
import mlx.nn as nn

class SomeModel(nn.Module):
    def __init__(self):
        self.attn = nn.Linear(...)            # Direct mlx.nn
    def forward(self, x):
        x = mx.reshape(x, (...))              # Direct mlx.core
```

### 4.2 Go-Style Shallow Difference (Tier 2a)

Used for: **Text Encoders** — the model architecture is the same, only weight loading differs.

**`common/text_encoders/qwen3.py`** — the body (~300 lines):

```python
"""
Qwen3 text encoder — shared model logic via RuntimeContext.
Backend-differentiated weight loading is in qwen3_mlx.py / qwen3_cuda.py.
"""
from backend.engine.runtime._base import RuntimeContext


class Qwen3EncoderLayer:
    def __init__(self, hidden_size, num_heads, head_dim, ctx: RuntimeContext):
        nn = ctx; self.ctx = ctx
        self.input_norm = nn.RMSNorm(hidden_size)
        self.self_attn = _Qwen3Attention(hidden_size, num_heads, head_dim, ctx)
        self.post_norm = nn.RMSNorm(hidden_size)
        self.mlp = _Qwen3MLP(hidden_size, intermediate_size, ctx)

    def forward(self, hidden_states, attention_mask, position_embeddings):
        residual = hidden_states
        hidden_states = self.self_attn(self.input_norm(hidden_states), attention_mask, position_embeddings)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_norm(hidden_states))
        return residual + hidden_states


class Qwen3Encoder:
    def __init__(self, ctx: RuntimeContext, model_path: str, **kwargs):
        self.ctx = ctx
        config = self._load_config(model_path)
        nn = ctx

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Qwen3EncoderLayer(config.hidden_size, config.num_heads, config.head_dim, ctx)
                       for _ in range(config.num_layers)]
        self.norm = nn.RMSNorm(config.hidden_size)
        self.rotary_emb = _Qwen3RotaryEmbedding(config, ctx)

        # ── Backend-differentiated weight loading ──
        if ctx.backend == "mlx":
            from .qwen3_mlx import load_weights
        else:
            from .qwen3_cuda import load_weights
        weights = load_weights(model_path, ctx)
        self._assign_weights(weights)

    def encode(self, texts: list[str]) -> tuple:
        ctx = self.ctx
        input_ids = ctx.array(self.tokenizer(texts)["input_ids"], dtype=ctx.int32())
        attention_mask = ctx.array(self.tokenizer(texts)["attention_mask"], dtype=ctx.float32())
        return self._forward(input_ids, attention_mask)

    def _forward(self, input_ids, attention_mask):
        ctx = self.ctx
        B, S = input_ids.shape
        hidden = self.embed_tokens(input_ids)
        pos_emb = self.rotary_emb(hidden, ctx.arange(S))
        causal_mask = self._make_causal_mask(S, hidden.dtype, ctx)
        for layer in self.layers:
            hidden = layer(hidden, causal_mask, pos_emb)
        return self.norm(hidden), attention_mask
```

**`common/text_encoders/qwen3_mlx.py`** — backend hook (~30 lines):

```python
"""MLX safetensors weight loading for Qwen3 encoder."""
from pathlib import Path


def load_weights(model_path: str, ctx) -> dict:
    import mlx.core as mx

    weights: dict = {}
    for sf in sorted(Path(model_path).glob("*.safetensors")):
        weights.update(dict(mx.load(str(sf))))
    return _strip_prefix(weights, "model.")
```

**`common/text_encoders/qwen3_cuda.py`** — backend hook (~30 lines):

```python
"""CUDA safetensors weight loading for Qwen3 encoder."""
from pathlib import Path


def load_weights(model_path: str, ctx) -> dict:
    import safetensors.torch

    weights: dict = {}
    for sf in sorted(Path(model_path).glob("*.safetensors")):
        weights.update(safetensors.torch.load_file(str(sf), device=ctx.device))
    return _strip_prefix(weights, "model.")
```

### 4.3 Go-Style Deep Difference (Tier 2b)

Used for: **VAE decoders** (CogVideoX, Qwen-Image) — layout differences (NHWC vs NCHW) permeate every layer.

**`families/cogvideox/vae.py`** — decode entry + shared PIL post-processing (imports `vae_mlx` for MLX path):

```python
class CogVideoXDecoder:
    """CogVideoX 3D VAE decoder interface."""
    def decode(self, latents_bcthw) -> Any:
        raise NotImplementedError


def create_cogvideox_decoder(ctx, bundle_root: Path, vae_cfg: dict) -> CogVideoXDecoder:
    if ctx.backend == "mlx":
        from .vae_mlx import CogVideoXDecoderMLX
        return CogVideoXDecoderMLX(ctx, bundle_root, vae_cfg)
    elif ctx.backend == "cuda":
        from .vae_cuda import CogVideoXDecoderCuda
        return CogVideoXDecoderCuda(ctx, bundle_root, vae_cfg)
    raise RuntimeError(f"CogVideoX decoder not available for backend: {ctx.backend}")


def decode_latents_to_frames(ctx, latents_bcthw, bundle_root, vae_cfg) -> list:
    """Shared post-processing: latent → pixel → PIL frames."""
    decoder = create_cogvideox_decoder(ctx, bundle_root, vae_cfg)
    # ── Shared latent pre-processing ──
    scaling_factor = float(vae_cfg.get("scaling_factor", 1.15258426))
    shift_factor = vae_cfg.get("shift_factor", None)
    latents = latents_bcthw / scaling_factor
    if shift_factor is not None:
        latents = latents + float(shift_factor)
    # ── Backend-specific decode ──
    pixels = decoder.decode(latents)
    # ── Shared pixel → PIL ──
    return _ncthw_to_pil_frames(ctx, pixels)
```

**`families/cogvideox/vae_mlx.py`** — full MLX NHWC implementation (~700 lines):

```python
"""CogVideoX 3D VAE decoder — MLX NHWC implementation."""
import mlx.core as mx
import mlx.nn as nn


class CogVideoXDecoderMLX(CogVideoXDecoder):
    def __init__(self, ctx, bundle_root, vae_cfg): ...
    def decode(self, latents): ...

# + All NHWC helper classes: SafeConv3d, CausalConv3d, SpatialNorm3D,
#   ResnetBlock3D, MidBlock3D, UpBlock3D, Upsample3D
```

**`families/cogvideox/vae_cuda.py`** — full CUDA NCHW implementation [NEW]:

```python
"""CogVideoX 3D VAE decoder — CUDA NCHW implementation."""
import torch
import torch.nn as nn


class CogVideoXDecoderCuda(CogVideoXDecoder):
    def __init__(self, ctx, bundle_root, vae_cfg): ...
    def decode(self, latents): ...

# + All NCHW helper classes
```

### 4.4 Qwen Transformer Rewrite (Tier 1 — ctx migration)

**Current architecture (broken):**
```
QwenImageTransformer(TransformerBase)   # accepts ctx, never passes it down
  └── QwenTransformer(nn.Module)        # zero ctx awareness
        ├── nn.Linear(...)              # ← direct mlx.nn
        ├── QwenTransformerBlock(nn.Module)
        │     ├── QwenAttention(nn.Module)
        │     ├── QwenFeedForward(nn.Module)
        │     └── AdaLayerNormContinuous(nn.Module)
        ├── QwenTimestepEmbedding(nn.Module)
        └── QwenTimeTextEmbed(nn.Module)
```

**New architecture:**
```
QwenImageTransformer(TransformerBase)   # thin wrapper
  └── QwenTransformer(ctx)              # all via ctx
        ├── ctx.Linear(...)             # ← ctx.Linear
        ├── QwenTransformerBlock(ctx)
        │     ├── QwenAttention(ctx)
        │     ├── QwenFeedForward(ctx)
        │     └── AdaLayerNormContinuous(ctx)
        ├── QwenTimestepEmbedding(ctx)
        └── QwenTimeTextEmbed(ctx)
```

File split:
| File | Content |
|------|---------|
| `qwen/transformer.py` | `QwenImageTransformer(TransformerBase)` — thin wrapper, passes ctx down |
| `qwen/modules.py` | `QwenTransformer`, `QwenTransformerBlock`, `QwenAttention`, `QwenFeedForward`, `AdaLayerNormContinuous` — all accept `ctx: RuntimeContext` |
| `qwen/embeddings.py` | `QwenTimesteps`, `QwenTimestepEmbedding`, `QwenTimeTextEmbed`, `QwenEmbedRope` — all ctx |
| `qwen/weights.py` | `remap_qwen_transformer_weights` — unchanged |

---

## 5. Acceptance Criteria

### 5.1 Zero Leak Check

```bash
# Must return EMPTY — no direct mlx imports outside runtime/ and *_mlx.py
grep -rln "import mlx\|from mlx" backend/engine/ --include="*.py" \
  | grep -v "runtime/" \
  | grep -v "_mlx.py"

# Must return EMPTY — no direct torch imports outside runtime/ and *_cuda.py
grep -rln "import torch\|from torch" backend/engine/ --include="*.py" \
  | grep -v "runtime/" \
  | grep -v "_cuda.py"
```

### 5.2 Per-Model Verification

| Model | Backends | Verification |
|-------|----------|-------------|
| flux1 | mlx, cuda | `danqing-generate --model flux1 --prompt "..."` on both backends |
| flux2 | mlx, cuda | Same, compare PSNR with reference |
| z_image | mlx, cuda | Already dual-backed; benchmark test exists |
| qwen | mlx, cuda | `danqing-generate --model qwen-image --prompt "..."` on both backends |
| fibo | mlx, cuda | Same |
| ltx | mlx, cuda | `danqing-video-generate --model ltx --prompt "..."` on both backends |
| wan | mlx, cuda | Same |
| cogvideox | mlx, cuda | Same; VAE decode verified on both |
| seedvr2 | mlx, cuda | Upscale pipeline verified on both |

### 5.3 Fail Loud Enforcement

- Models registered with `backends: ["mlx", "cuda"]` must execute on both backends
- If CUDA implementation is missing, the model should fail with a clear error — **never silently fall back to MLX**
- Pipeline must not have `if family == ...:` branches — all differences handled by Transformer polymorphism or registry

---

## 6. Migration Phases

| Phase | Scope | Risk | Reward |
|-------|-------|------|--------|
| **0** | Go-style package scaffold: `text_encoders/` + `vae/` interface files | Low (new files only) | Foundation for CUDA implementations |
| **1** | flux1, flux2, fibo, cogvideox: replace `mx.*` → `ctx.*` in `forward()` | Low (mechanical refactor) | 4 models immediately dual-backend ready |
| **2** | qwen: full rewrite — inner modules all accept `ctx` | Medium (requires pixel benchmark) | Worst offender fixed; dual-backend enabled |
| **3** | Text encoder consolidation → `common/text_encoders/` + CUDA hooks | Medium | Eliminate 3 separate text encoder implementations |
| **4** | seedvr2: ctx-ify + Go-style VAE | High (video benchmark complex) | Last MLX-only code eliminated |
| **5** | CI leak checker (zero import script) | Low | Prevents regression |

---

## 7. File Deletion / Consolidation Schedule

### Files to DELETE (after migration):

| File | Reason | Phase |
|------|--------|-------|
| `z_image/text_encoder_mlx.py` | Consolidated into `common/text_encoders/qwen3_mlx.py` | 3 |
| `flux2/text_encoder.py` | Consolidated into `common/text_encoders/qwen3.py` (body) + `qwen3_mlx.py` (hook) | 3 |
| `qwen/text_encoder.py` | Consolidated into `common/text_encoders/qwen3.py` + `qwen3_mlx.py` | 3 |
| `common/vae/cogvideox_decoder.py` (removed) | Replaced by `families/cogvideox/vae.py` + `vae_mlx.py` | done |
| `common/vae/cogvideox_decoder_mlx.py` (removed) | Renamed to `families/cogvideox/vae_mlx.py` | done |
| `common/vae/mlx_tiling.py` | Moved to `common/vae/tiling_mlx.py` | 0 |
| `common/vae/qwen_image/*.py` (9 files) | Consolidated into `common/vae/qwen_image_decoder_mlx.py` | 0 |
| `common/bundle_weights/applier.py` | Merged into `common/bundle_weights/loader_mlx.py` | 0 |
| `common/bundle_weights/loader.py` | Split into `loader_mlx.py` + `loader_cuda.py` | 0 |
| `common/bundle_weights/definitions.py` | Kept as `base.py` | 0 |
| `common/bundle_weights/loaded_weights.py` | Merged into `base.py` | 0 |
| `common/bundle_weight_mapping.py` | Moved to `common/bundle_weights/remap.py` | 0 |
| `seedvr2/sv2_*` (legacy) | `families/seedvr2/*_mlx.py`（MLX 实现 + CI `_mlx` 豁免） | done |

### Files to RENAME:

| Current | New | Phase |
|---------|-----|-------|
| `common/vae/decoder.py` + `encoder.py` | `common/vae/autoencoder.py` | 0 |
| `common/vae/weight_remap.py` | `common/vae/remap.py` | 0 |
| `common/bundle_weight_mapping.py` | `common/bundle_weights/remap.py` | 0 |
| `seedvr2/sv2_dit.py` | `families/seedvr2/dit_mlx.py` | done |
| `seedvr2/sv2_vae_net.py` | `families/seedvr2/vae_mlx.py` | done |
| `seedvr2/sv2_embed.py` | merged into `families/seedvr2/preprocess_mlx.py` | done |
| `seedvr2/sv2_schedule.py` | merged into `families/seedvr2/job_mlx.py` | done |
| `seedvr2/sv2_preprocess.py` | `families/seedvr2/preprocess_mlx.py` | done |
| `seedvr2/sv2_job.py` | `families/seedvr2/job_mlx.py` (+ schedule/result/video) | done |
| `seedvr2/sv2_result.py` | merged into `families/seedvr2/job_mlx.py` | done |
| `seedvr2/sv2_weight_schema.py` | `families/seedvr2/weights_mlx.py` | done |
| `seedvr2/sv2_model_spec.py` | merged into `families/seedvr2/weights_mlx.py` | done |
| `seedvr2/sv2_dispatch.py` | merged into `families/seedvr2/job_mlx.py` (`run_seedvr2_upscale`) | done |
| `seedvr2/sv2_bundle.py` | merged into `families/seedvr2/weights_mlx.py` (`load_flat_bundle`) | done |

---

## 8. 与当前共识的差异与建议（目录一次做对、少留债）

> 本节在 **§1–§7 原文** 之上做对照：§3 目录树与 §2「两层策略」仍具参考价值，但与后续产品决策 **不完全一致** 处以下文为准，避免双轨技术债。

### 8.1 一致、应保留的部分

- **CI 守门思路**（§5.1）：`mlx`/`torch` 仅出现在 `runtime/` 与 `*_mlx.py` / `*_cuda.py`（或等价命名），与当前共识一致。
- **Go 式三文件组**命名：`stem.py` / `stem_mlx.py` / `stem_cuda.py`；**同 `stem` 一组在族内文件预算中计 1**，与当前共识一致。
- **深差异组件**（VAE NHWC vs NCHW、整栈 HF）用分文件全量实现 + 无后缀文件做 **接口与 dispatch**，与当前共识一致。
- **保留** Engine 契约、任务队列、`ImagePipeline`/`VideoPipeline` **装配阶段语义**（与 §3 中「pipelines / engines UNCHANGED」方向一致；仅 **物理路径** 可迁至 `pipelines/` 子目录，见下）。

### 8.2 差异与修订建议

| 主题 | 本文档原表述（§2–§4） | 当前共识 | **建议（一次做对）** |
|------|------------------------|----------|------------------------|
| **Tier 1** | 所有 DiT 热路径 **必须** 仅用 `RuntimeContext`，「现有 `mx.*` 均可换 `ctx.*`、无需扩展 API」 | **收窄 `RuntimeContext`**：不全能化；DiT 大块可放在 `transformer_mlx.py` / `transformer_cuda.py`，由 `transformer.py` 工厂装配 | **采纳当前共识**：避免 `_base.py` 无限膨胀；若某族仍愿全 ctx，可作为 **族内实现选项**，不作为全仓强制 Tier1。本文 §4.1「禁止 `import mlx`」应对 **无后缀族内文件** 落实；**允许**在 `*_mlx.py` 内 `import mlx`。 |
| **`xxx.py` 职责** | §2.4：正文在 `xxx.py`，`_mlx`/`_cuda` 仅 hook | 曾出现「仅工厂」表述 | **以 §8.5 形态 A/B 为准**：差异小→公共 ctx 实现；差异极大→**对外接口 + dispatch** + 共享 ctx 前后处理；平台整段在 `*_mlx`/`*_cuda`。 |
| **目录** | §3：`flux1/`、`flux2/` 等仍在 `engine/` 根下；`image_pipeline.py` 根下 | `engine/families/<id>/` + `engine/pipelines/` 集中；族内 **折算逻辑单位 ≤8**（同名 `stem` 的 `stem.py`+`stem_mlx.py`+`stem_cuda.py` **计 1**；可缺后缀） | **以当前共识重写 §3**：物理路径采用 `backend/engine/families/<family>/`；`image_pipeline.py` 等迁至 `backend/engine/pipelines/`；**禁止**长期同时保留「根下 family」与「families/」两套路径。 |
| **SeedVR2** | §3 仍列 `pipeline.py`、`dit.py`、`embed.py`… 等多文件 | **折算 ≤8**：当前为 8 个独立 `stem_mlx.py`（各算 1 单位）；若拆 Go 三文件组则一组仍计 1 | **废弃 §3 seedvr2 文件清单的粒度**：以 `families/seedvr2/*_mlx.py` 为准更新示意图；与「几十文件绝不可接受」对齐。 |
| **`common/`** | §3 大量迁入 `common/vae/`、`common/text_encoders/` | **族专属**逻辑优先在 `families/<id>/`；`common/` 仅 **真·跨族** 薄复用 | **单族专用 VAE**（如 CogVideoX）建议最终落在 **`families/cogvideox/`** 三件套，避免 `common/vae/` 再次成为杂项堆。跨族编码器（Qwen3/CLIP/T5）可保留 `common/text_encoders/` 的 Go 式拆分。 |
| **bundle_weights** | §3 拆 `loader_mlx` / `loader_cuda` | 同上，且须满足 CI 后缀规则 | **采纳 §3 方向**；与 `remap` 职责边界写清，避免与 `family/weights.py` 重复维护。 |

### 8.3 单一真相源与后续动作

- **§3 目录树**：实施绿field 时应用 **§8.2 建议** 整体替换，不再在文档内保留两套互相矛盾的树超过一个里程碑。
- **§6 迁移阶段**：可与「先 CI + 再按族合并文件」并行，但 **SeedVR2 / Qwen-Image VAE** 的交付形态以 **§8** 的文件预算为准。
- **与 AGENTS / `model-migration.mdc`**：将「三文件命名 + `families/` + pipelines 子目录 + **族内折算逻辑单位 ≤8**（`stem.py`+`stem_mlx.py`+`stem_cuda.py` 同 stem 计 1）」写入仓库级规则，避免仅存在于本文或仅存在于 Cursor 计划。

### 8.5 三文件语义（最终定稿）：形态 A / 形态 B

**核心结论**（与 Cursor 计划一致）：

- **差异相对小**：`xxx.py` 写 **基于 `RuntimeContext` 的公共实现**；`xxx_mlx.py` / `xxx_cuda.py` 只放 **平台差异化的少量重写函数 / 钩子**。
- **差异特别大**：`xxx.py` 只保留 **对外接口**（`Protocol`/ABC、`create_*`、稳定调用面）+ **dispatch** + 仍能用 **`ctx.*`** 表达的 **共享前后处理**；**各平台整段实现**写在 `xxx_mlx.py` / `xxx_cuda.py`（文件头注明 `implementation_mode=full_platform`）。

| 形态 | **`xxx.py`** | **`xxx_mlx.py` / `xxx_cuda.py`** |
|------|----------------|-----------------------------------|
| **A（默认）** | 公共 ctx 实现（主体，非空壳） | 少量钩子 / 重写函数 |
| **B（兜底）** | **对外接口** + dispatch + 共享 ctx 前后处理 | **平台完整实现**（整网） |

**约定**：优先 **形态 A**；仅当无法抽成少量钩子时采用 **形态 B**。**禁止**第四种布局或 `backends/` 平行包。族内体量按 **折算逻辑单位**（同 `stem` 的 `stem.py`+`stem_mlx.py`+`stem_cuda.py` 计 1）受项目预算约束。CI：`mlx`/`torch` 仅 `runtime/` 与 `*_mlx`/`*_cuda`。

**文本编码（形态 B 已拆）**：`clip_mlx` / `t5_mlx` / `qwen25vl_mlx` 与 `families/z_image/text_encoder_mlx` **不含**字面 `import torch`；PyTorch 前向与 HF 桥接在 `common/text_encoders/{clip,t5,qwen25vl}_cuda.py` 与 `families/z_image/text_encoder_cuda.py`。

### 8.4 小结

| 保留（参考本文） | 以当前共识覆盖（避免技术债） |
|------------------|------------------------------|
| 双后端 CI、fail loud、注册表驱动 | **目录**：`families/` + `pipelines/` 一次到位 |
| Go 三文件命名、**形态 A / 形态 B**（公共实现 vs 对外接口+平台整段） | **RuntimeContext 收窄**；DiT 不必强行全 ctx |
| `common` 内跨族组件的 Go 式拆分（文本编码等） | **族内折算逻辑单位上限**（Go 三文件同 stem 计 1）；SeedVR2 / CogVideoX 目标见项目计划 |
| Pipeline/Engine **职责** 不变 | **物理路径**可迁、**实现** 用工厂接 `_mlx`/`_cuda` |
