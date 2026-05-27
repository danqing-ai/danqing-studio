# DanQing-Studio UX-First Optimization Plan

> **Version**: v3.1 — LRU-aligned implementation  
> **Date**: 2026-05-24  
> **Constraint**: **ModelCache LRU + TTL** — no turbo resident / pin / startup preload.  
> **Core insight**: Users care about **clarity during a run** and **honest wait labels**, not only total time.

---

## 1. UX metrics (revised under LRU)

| Metric | Definition | Target (realistic) |
|--------|------------|-------------------|
| **Hot-cache TTFP** | Model already loaded → first `preview` SSE | First preview within a few denoise steps |
| **Cold TTFP** | Includes model load + text encode | Dominated by load (10–30s for large weights); show `loading_model` phase |
| **Clarity evolution** | Step previews during denoise | `preview_interval_steps` default **2** |
| **Queue transparency** | After submit | `queue_position` + `estimated_wait_seconds` on create page |

**Not pursued**: sub-1s cold TTFP, turbo always resident, latent snapshot / CFG resume, interpolate-before-VAE “fast decode”.

---

## 2. Preview capability (`models_registry.json`)

Per diffusion image model (in `parameters`):

| Field | Values | Default |
|-------|--------|---------|
| `preview_mode` | `stream` \| `none` | `stream` (except `seedvr2` → `none`) |
| `preview_interval_steps` | int | `2` |
| `preview_max_edge` | int px | `512` |

**Status column meaning**

| Label | Meaning |
|-------|---------|
| `inference_ready` | Model runs end-to-end via `ImagePipeline` |
| `preview_wired` | `preview_mode=stream` + SSE `preview` + Living Canvas |

---

## 3. Implemented (2026-05)

### Backend

- [`ProgressEvent`](backend/core/contracts.py): `phase`, `preview_asset_id`
- [`ImagePipeline`](backend/engine/pipelines/image_pipeline.py): step preview decode → ephemeral asset (replaces previous preview asset per task)
- SSE [`GET /api/tasks/{id}/stream`](backend/api/routes/tasks.py): `event: preview`
- Phases: `encoding`, `loading_model`, `denoising`, `decoding`, `saving`

### Frontend

- [`LivingCanvas.vue`](frontend/src/components/create/LivingCanvas.vue) + [`ImageCreateView.vue`](frontend/src/views/ImageCreateView.vue): `onPreview`, phase labels, queue ETA, memory bar, turbo hint
- Duplicate SSE avoided: create page calls `tasksStore.closeTaskLogStream` before opening its stream
- **Enhance** button (optional P3): after `z-image-turbo` / `flux1-schnell` completes → one-click second task with `z-image` / `flux1-dev`

---

## 4. Deferred

| Item | Reason |
|------|--------|
| Video frame-by-frame preview | Needs per-family VAE chunk validation |
| Parameter snapshot / 2s slider feedback | Incompatible with CFG math + LRU reload |
| Turbo resident manager | Conflicts with ModelCache policy |
| Fake VAE via latent interpolate | Quality risk |

---

## 5. Acceptance (manual)

1. `z-image`, 28 steps, model warm: ≥10 `preview` updates; final via `done` + `result`.
2. Cold run: progress shows **正在加载模型** before denoise.
3. Queued task: create page shows position / ETA.
4. Failed task: error text from API `error` field.

---

## 6. Related

- Implementation checklist (LRU): see plan *UX 提升清单（LRU 卸载前提下）* in Cursor plans.
- Personal-tool plan: [`danqing-studio-personal-tool-optimization-plan.md`](danqing-studio-personal-tool-optimization-plan.md) — queue/serial alignment noted.
