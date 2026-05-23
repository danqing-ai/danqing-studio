# Python + MLX Engine Checklist

New model integration policy for DanQing Studio. Keep this checklist aligned with:

- `AGENTS.md`
- `.cursor/rules/model-migration.mdc`
- `.cursor/rules/no-silent-degrade.mdc`
- `docs/dual_platform_architecture.md`

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
- [ ] Pick nearest in-repo reference family (e.g. Flux2 / Z-Image / Wan)
- [ ] Split planned code into:
  - reusable common math
  - family-specific topology/remap

### Phase B: Registry & Config Wiring

- [ ] Update `default_config/models_registry.json`
- [ ] Run `make sync-models-registry`
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
