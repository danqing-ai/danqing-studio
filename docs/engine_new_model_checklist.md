# Python + MLX Engine Checklist

New model integration policy for DanQing Studio. Keep this checklist aligned with:

- `AGENTS.md`
- `.cursor/rules/model-migration.mdc`
- `.cursor/rules/no-silent-degrade.mdc`
- `docs/dual_platform_architecture.md`
- `docs/canonical_image_family_flux2.md` — **canonical Shape A template**

---

## 0) Plugin shape decision (A / B / C)

Pick **one** shape before writing code:

| Shape | When | Entry | Copy template |
|-------|------|-------|---------------|
| **A — DiT + ImagePipeline** | Standard txt2img / edit with `TransformerBase.forward(latents, t, txt_embeds=…)` | `ImagePipeline.run()` | **flux2** (`docs/canonical_image_family_flux2.md`) |
| **B — Job Pipeline** | Upscale / MM-DiT with non-standard denoise API | `ImageUpscalePipeline` → `families/<family>/stem.py` + `stem_mlx.py` | seedvr2 |
| **C — Generation Facade** | End-to-end audio or whole-stack generator | `MusicPipeline` → `families/<family>/generation.py` | ace_step, heartmula |

**Do not** mirror upstream directory trees (e.g. `families/<family>/mlx/`). HeartMuLa's `mlx/` subtree is allowlisted temporarily only.

```
Need standard image denoise loop?
├─ yes → Shape A (flux2 template)
└─ no
   ├─ upscale / SR job? → Shape B
   └─ audio / full generator? → Shape C
```

---

## 0b) Five registration touch points (Shape A)

Complete **all five** before merging:

- [ ] **`default_config/models_registry.json`** — required: `family`, `engine`, `actions`, `backends`, bilingual `name`/`description`
- [ ] **`make sync-models-registry`**
- [ ] **`backend/engine/config/model_configs.py`** — dataclass, `FAMILY_CONFIG_MAP`, `IMAGE_FAMILY_REUSE_CONTRACT`
- [ ] **`backend/engine/_transformer_registry.py`** — `_TRANSFORMER`, `_WEIGHT_REMAP`, `_TEXT_ENCODER`, optional `_IMAGE_LORA_MERGE`
- [ ] **`backend/engine/vae_codec_registry.py`** — only if VAE `_class_name` ≠ generic `AutoencoderKL`

`family` is **required** in registry JSON (`ModelRegistry.load` fails loud if missing).

---

## 0c) VAE handling (Shape A)

- [ ] **Generic** — `AutoencoderKL` + `common/vae/decoder.py` (no codec registration)
- [ ] **Family codec** — register `_class_name` in `vae_codec_registry.py` (flux2, qwen, wan)
- [ ] **Large VAE subtree** — `common/vae/<name>/` acceptable when mapping table is long (qwen_image)

---

## 1) Three-Layer Boundaries (L1/L2/L3)

- [ ] **L1 Core** (`backend/engine/runtime/`, `backend/engine/common/`):
  - model-agnostic math primitives, schedulers, cfg utilities, shared embeddings/norm/attention
- [ ] **L2 Family** (`backend/engine/families/<family>/`):
  - family topology, family-only conditioning, family weight remap
- [ ] **L3 Contract** (registry + configs + pipeline wiring):
  - model declaration, action contracts, pipeline dispatch and loading
- [ ] Tier-1 hot path uses `ctx.*` by default
- [ ] Tier-2 (`*_mlx.py`, `*_cuda.py`) only when `ctx` cannot express required backend behavior
- [ ] No long-term `if family == "..."` branch in pipeline

---

## 2) Reuse Matrix (Check Before Coding)

- [ ] Attention:
  - prefer `backend/engine/common/attention.py`
  - avoid direct `ctx.attention(...)` in families
- [ ] RoPE / embeddings:
  - prefer `backend/engine/common/embeddings.py`
  - avoid family-local duplicate complex RoPE wrappers
- [ ] Norm / AdaLN / modulation:
  - prefer `backend/engine/common/norm.py`
- [ ] CFG batching:
  - prefer `backend/engine/common/cfg_batch.py`
- [ ] Scheduler:
  - prefer `backend/engine/common/schedulers.py`
- [ ] If common math is close but not exact:
  - extend common helper signatures first
  - only keep family-local math when topology is truly unique

---

## 3) New Model Integration Checklist

### Phase A: Design & Contract

- [ ] Read upstream for math reference only (do not mirror upstream directory tree)
- [ ] Confirm `media`, `actions`, `backends` and fail-loud behavior
- [ ] Pick nearest in-repo reference family:
  - Shape A: **Flux2** (preferred) or Z-Image / Qwen
  - Shape B: seedvr2
  - Shape C: ace_step (`generation.py` facade)
- [ ] Optional scaffold: `python scripts/scaffold_image_family.py --family my_family --class MyFamilyTransformer`
- [ ] Split planned code into:
  - reusable common math
  - family-specific topology/remap

### Phase B: Registry & Config Wiring

- [ ] Update `default_config/models_registry.json`
- [ ] Run `make sync-models-registry`
- [ ] Optional profile shrink: `apply_standard_profile()` in `backend/core/registry_profiles.py` (expanded doc must stay identical)
- [ ] Register bundle components in `backend/core/bundle_manifest.py` (`FAMILY_BUNDLE_CONTRACTS`) when enabling install validation
- [ ] Pipelines call `assert_media_bundle_ready()` from `backend/engine/common/bundle_layout.py` before load
- [ ] Optional graph steps: `pipeline_graph_step` from `backend/engine/pipelines/pipeline_progress.py`
- [ ] Update `backend/engine/config/model_configs.py`
- [ ] Update `backend/engine/_transformer_registry.py`:
  - transformer map
  - weight remap map
  - text encoder map (if needed)
- [ ] Keep pipeline behavior registry-driven

### Phase C: Family Implementation

- [ ] Keep default family layout minimal (`transformer.py`, optional `text_encoder.py`, `weights.py`)
- [ ] Keep family logical units reasonable (see migration rule target)
- [ ] Prefer common modules for attention/norm/embeddings/schedulers before new family-local helpers
- [ ] If nested params exist, flatten `_param_map` and align exactly with remap keys
- [ ] Preserve explicit fail-loud errors on shape/key mismatch

### Phase D: Weights & Dtype Policy

- [ ] Implement `remap_<family>_weights` with auditable mapping rules
- [ ] Validate key-space parity: remap output keys == model param keys
- [ ] No silent key drops or hidden compatibility fallbacks
- [ ] Keep dtype handling explicit and aligned with neighboring families

### Phase E: Validation & Delivery

- [ ] `python -m py_compile <touched files>`
- [ ] `make verify-engine-stack` (runs governance gates + unit tests)
- [ ] Minimal inference path validated (CLI/bench equivalent)
- [ ] Desktop packaging reachability verified for added runtime modules

### Phase E2: Qwen-Image CUDA parity (when `backends` includes `cuda`)

Requires NVIDIA + local `qwen-image` bundle under `./models/` (sync registry first).

- [ ] `make sync-models-registry`
- [ ] Smoke: `bin/danqing-generate --model qwen-image --prompt "a red cube" --runtime cuda --steps 4 --width 512 --height 512 --output /tmp/qwen-cuda.png`
- [ ] Compare against MLX on same seed/steps (visual or `make bench-mflux-case` if reference exists)
- [ ] If DiT diverges: verify `transformer_cuda.py` timestep scaling (`sigma → int(t×1000)`) and packed latent layout vs diffusers pipeline

---

## 4) Implementation Decision Tree

- [ ] If new block is isomorphic to existing common helper:
  - reuse common directly
- [ ] If shape/layout differs but core math is same:
  - extend common helper parameters
- [ ] If difference is backend-only and small:
  - isolate in `*_mlx.py` / `*_cuda.py` hook
- [ ] If topology is uniquely family-specific:
  - keep in family transformer module
  - avoid creating parallel subtree packages

---

## 5) Anti-Patterns (Reject)

- [ ] Family-local copies of generic `SelfAttention`/`RMSNorm` style primitives
- [ ] Parallel subtree under family (`mlx/`, `torch/`, `runtime/`, `common/`) without explicit governance allowance
- [ ] Silent degrade/fallback for load/inference failures
- [ ] Hidden model-specific behavior in API/CLI routes or pipeline branch logic
- [ ] Mirroring upstream project structure instead of DanQing contracts

---

## 6) PR Review Anchors

- [ ] Architecture boundary preserved (L1/L2/L3)
- [ ] Common reuse justified and documented
- [ ] Remap and param map alignment verified
- [ ] Fail-loud behavior preserved
- [ ] Validation commands and key outputs included in PR
