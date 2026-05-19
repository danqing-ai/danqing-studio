# AGENTS.md — DanQing Studio v4 (丹青工作室)

Operational guide for contributors and coding agents. User-facing overview: [README.md](README.md) (English default, bilingual).

---

## What this is

DanQing Studio — plugin-style **image / video** generation on **MLX** (Apple Silicon) and **CUDA** (NVIDIA). Split stack: FastAPI + Vue 3 SPA + CLI + SQLite. Models are declared in JSON and implemented under `backend/engine/families/`.

---

## Quick reference

| Item | Location |
|------|----------|
| API entry | `backend/main.py` |
| Frontend entry | `frontend/index.html` → Vite → `frontend/src/` |
| Production UI build | `out/frontend/dist/` (`frontend/vite.config.ts`) |
| Launch / stop | `bin/launch.sh`, `bin/stop.sh` |
| Model registry | `config/models_registry.json` |
| Family configs | `backend/engine/config/model_configs.py` |
| Transformer registry | `backend/engine/_transformer_registry.py` |
| Contracts | `backend/core/contracts.py` |
| Media interfaces | `backend/core/media_interfaces.py` (`IImageEngine`, `IVideoEngine`) |
| Runtime | `backend/engine/runtime/` (`MLXContext`, `CudaContext`) |
| Task kinds | `backend/core/task_kinds.py` (do not hardcode kind strings) |
| Cursor rules | `.cursor/rules/model-migration.mdc`, `.cursor/rules/no-silent-degrade.mdc` |
| Dual-platform design | `docs/dual_platform_architecture.md` |
| Desktop | `desktop/`, `make pack-macos-desktop` |

### Hardcoded paths

- Models: `./models/`
- LoRAs: `./models/Lora/`
- Outputs: `./outputs/`
- Config: `config/.app_config.json`, `config/presets.json`, `config/models_registry.json`
- DB: `db/studio.db` (WAL; no runtime `ALTER` — reset DB + `outputs/` if schema drifts)

### Environment

- **Python** 3.11+, venv at `.venv/`
- **Key packages**: `fastapi`, `uvicorn`, `mlx`, `Pillow`, `transformers`, `safetensors` (+ `torch` when using CUDA)
- **Benchmark venv**: `tests/benchmark/venv/` via `make bench-setup` (isolated from app venv)
- **Platform detect**: `backend/engine/platform.py` — `["mlx"]` on darwin arm64 with mlx; `["cuda"]` when `torch.cuda.is_available()`; **fail loud** if a model’s `backends` entry is not satisfied

---

## Governance anchors (must follow)

Three product-level constraints (also in `.cursor/rules/*.mdc`):

1. **Layering / plugin model** — Reuse `backend/engine/common/`; new models = registry + `model_configs` + `families/<family>/` + `_transformer_registry`. **No** `family == …` branches in `ImagePipeline` / `VideoPipeline` for business logic.
2. **Contract API + CLI** — Routes/CLI only through contracts + `IImageEngine` / `IVideoEngine`. Per-model behavior via registry `actions` / `parameters` and Transformer polymorphism + Hooks — **not** `model_id` switches in `backend/api/routes/` or `backend/cli/`.
3. **RuntimeContext** — Hot paths use `ctx.*`; literal `import mlx` / `import torch` only in `backend/engine/runtime/` or `*_mlx.py` / `*_cuda.py` (or dynamic import). CI: `make check-engine-imports`. Missing backend for a declared capability → **explicit error**, no silent fallback.

| Dimension | Acceptance |
|-----------|------------|
| Plugin | New image family: JSON + `model_configs` + `families/<family>/` + registry; no new Pipeline `family` branches |
| Family size | `families/<family>/` ≤ **8 logical units**; `stem.py` + `stem_mlx.py` + `stem_cuda.py` = **1** unit |
| API/CLI | Extend contracts + route/CLI first; REST and CLI stay aligned |
| Dual platform | Multi-`backends` models must run on each declared runtime or fail loud |

### Fail loud (default)

No silent degradation on generation, load, or registry parse failures. Use `RuntimeError` / clear HTTP errors + `t(key, locale)` + task logs/SSE. Documented user-consent toggles are the **only** allowed fallback (off by default). See `.cursor/rules/no-silent-degrade.mdc`.

---

## Architecture

### Layering

```
REST API / CLI
    ↓  contracts + interfaces only
TaskScheduler (global single queue, serial worker)
    ↓  IImageEngine / IVideoEngine / (audio placeholder)
DanQingImageEngine / DanQingVideoEngine / DanQingAudioEngine
    ↓  Pipeline + RuntimeContext
ImagePipeline / VideoPipeline / …
    ↓  RuntimeContext + TransformerBase + common/
V3TaskStore + SQLiteAssetStore
```

### Modular directories

| Path | Role |
|------|------|
| `backend/api/routes/` | REST by media (`images`, `videos`, `tasks`, `assets`, `registry`, …) |
| `backend/cli/` | `bin/danqing-*`, mirrors REST |
| `backend/scheduler/` | `TaskScheduler` |
| `backend/engine/pipelines/` | Assembly lines |
| `backend/engine/families/<family>/` | Per-family transformer, weights, text encoder |
| `backend/engine/runtime/` | MLX / CUDA contexts |
| `backend/engine/common/` | VAE, schedulers, encoders, `TransformerBase` |
| `backend/persistence/` | SQLite stores |
| `backend/core/` | Interfaces, contracts, container, i18n |
| `backend/services/` | Settings, download |

### Replaceable components

- **RuntimeContext** — `MLXContext` / `CudaContext`
- **Scheduler** — `FlowMatchEulerScheduler`, `LinearScheduler`, …
- **VAEDecoder** — `scaling_factor`, `shift_factor`, `pytorch_compatible`
- **TextEncoder** — T5, Qwen3, CLIP, … (by `config.encoder_type`)
- **ModelCache** — LRU in engine layer

### Model as plugin

Every DiT on `ImagePipeline` / `VideoPipeline` **extends `TransformerBase`** (`backend/engine/common/_base.py`): unified `forward` / `load_weights` / `parameters`, optional Hooks, default `refine_cfg_noise`.

**Pipeline.run()** (registry-driven, no family branches):

1. Resolve model → family + `ModelConfig`
2. Text encoder from `config.encoder_type`
3. Scheduler from registry `parameters.scheduler`
4. Initial latents from `config.vae_scale`
5. Denoise loop: `model(latents, t, txt_embeds=…, sigmas=…)`; CFG + optional `refine_cfg_noise`
6. VAE decode (flags from weights/config, not `family` string)
7. `asset_store.create_from_file`

**Invariant**: Prefer changing registry, config, transformer, remap, registry wiring only. Pipeline changes = new **registry-driven** switches or polymorphic call sites only.

### Transformer hooks

| Hook | When | Typical use |
|------|------|-------------|
| `after_load_weights(bundle_root)` | After `load_weights` | LoRA merge |
| `prepare_conditioning(request, bundle)` | After text encode | ControlNet cond dict |
| `before_denoise(latents, timesteps, sigmas, **cond)` | Before denoise loop | Inject control signal |
| `step_callback(step_idx, latents, noise_pred)` | Each step | Dynamic strength / logging |

`refine_cfg_noise` is **polymorphic** (not a hook), called when registry `enable_cfg_renorm` and CFG apply.

### Pipeline rules

- Scalars/enums in **registry + `model_configs`**, not `if family` in Pipeline.
- Shape/operator math in **Transformer** methods.
- Hooks for **optional cross-cutting** features only.
- Do not register `actions` the model cannot serve; `IImageEngine.supports` rejects at entry.
- Weight-key / config flags for variants (e.g. flux2-style VAE prep), not `family` strings.

### Family layout (migration)

Default per family: `transformer.py`, `text_encoder.py` (if needed), `weights.py`. Dual platform: Go-style `stem.py` / `stem_mlx.py` / `stem_cuda.py` (one logical unit). See `.cursor/rules/model-migration.mdc`.

**Weights**: Prefer `remap_<family>_weights` registered in `_WEIGHT_REMAP` / `_VIDEO_WEIGHT_REMAP` (Z-Image / Flux2 style). VAE may use long tables. MLX nested `parameters()` → explicit flat `_param_map` matching remap keys.

### Implemented families (indicative)

`flux1`, `flux2`, `z_image`, `fibo`, `qwen` (`qwen_image` in registry), `seedvr2`, `ltx`, `wan`, `cogvideox`, `ace_step`, … — source of truth is `_transformer_registry.py` + `models_registry.json`.

---

## New model checklist

1. **`config/models_registry.json`** — `family`, `engine`, `actions`, `parameters`, `versions`, `backends`, bilingual `name`/`description`
2. **`backend/engine/config/model_configs.py`** — dataclass + `FAMILY_CONFIG_MAP`
3. **`backend/engine/families/<family>/transformer.py`** — `TransformerBase`, inject `RuntimeContext`
4. **`backend/engine/families/<family>/weights.py`** — `remap_<family>_weights` (if needed)
5. **`backend/engine/_transformer_registry.py`** — `_TRANSFORMER`, `_WEIGHT_REMAP`, `_TEXT_ENCODER` as needed

**Verify**

```bash
python -m py_compile <touched files>
bin/danqing-generate --model <id> --prompt "test"
make bench-mflux          # when mflux reference exists
make bench-sanity         # no reference CLI
make test-engine-unit
make check-engine-imports
```

**Benchmark snapshot (2026-05, mflux create)**

| Model | Action | PSNR | Status |
|-------|--------|------|--------|
| flux2-klein-9b | create | 31.9 dB | PASS |
| z-image | create | 28.6 dB | WARN |
| z-image-turbo | create | 16.7 dB | FAIL |

---

## Configuration

### Registry (`schema_version: 2`)

- Top-level `engines`: `danqing-image`, `danqing-video`, `danqing-audio`
- Per model: `media` (`image` \| `video` \| `audio`), **`actions`** (not legacy `capabilities`)
- Image actions: `create`, `rewrite`, `retouch`, `extend`, `upscale`
- Video actions: `create`, `animate`
- `parameters`: typed (`int`, `float`, `enum`, `bool`, `object`)
- `name` / `description`: `{ "zh", "en" }`

### Presets

`config/presets.json` — each preset needs **`applies_to`** and **`media_scope`** (`image` \| `video`). UI filters by tab + scope.

### App settings

`config/.app_config.json` — `language`, `theme`, `default_model`, `mlx_memory_limit`, `queue_image_first`, …

---

## CLI vs REST API

| CLI | REST | Engine |
|-----|------|--------|
| `bin/danqing-generate` | `POST /api/images/generations` | `IImageEngine.generate()` |
| `bin/danqing-edit` | `POST /api/images/edits` | `IImageEngine.edit()` |
| `bin/danqing-upscale` | `POST /api/images/upscales` | `IImageEngine.upscale()` |
| `bin/danqing-video-generate` | `POST /api/videos/generations` | `IVideoEngine.generate()` |
| `bin/danqing-video-edit` | `POST /api/videos/edits` | `IVideoEngine.edit()` |

Engines: `ModelRegistry` + `EngineRegistry` → `DanQingImageEngine` / `DanQingVideoEngine`; runtime from registry `backends` via `_resolve_runtime`.

---

## API endpoints (catalog)

### Images / video

- `POST /api/images/generations` | `edits` | `upscales`
- `POST /api/videos/generations` | `edits`

### Tasks / queue

- `GET /api/tasks` (= `/api/tasks/list`) — `limit`, `offset`, `kind`, `status`, `since`
- `GET /api/tasks/{id}` — `queue_position`, `estimated_*`, `model`, `error`, …
- `GET /api/tasks/{id}/logs` — paginated logs
- `PATCH /api/tasks/{id}` — **`queued` only**: `{ "priority": "normal" \| "high" }`
- `DELETE /api/tasks/{id}` — cancel
- `GET /api/tasks/{id}/stream` — SSE: `log`, `progress`, `status`, `result`, `done`
- `GET /api/queue` — snapshot; `estimated_wait_seconds` on queued items

`queue_image_first` in settings: image tasks dequeue before video.

### Assets / gallery

- `POST /api/assets` | `GET /api/assets` | `GET /api/assets/{id}/file` | `…/thumbnail`
- `POST /api/assets/reconcile` — `{ "dry_run": true }` default
- `GET /api/gallery/images` | `POST /api/gallery/upload` | `DELETE /api/gallery/image?path=asset:{id}`
- Task metadata: steps, guidance, seed, width/height; video `duration_seconds` (ffprobe or `num_frames/fps`)

### Registry / models / system

- `GET /api/registry` — full registry + `_index`
- `GET /api/models` — filter `media`, `action`, `installed`
- `GET /api/models/{id}` | `POST …/install` | `POST /api/models/install-batch` | `DELETE …/versions/{version_key}`
- `GET /api/presets` (read); writes via `/api/settings/presets`
- `GET /api/adapters` — LoRA index; `for_model` filter
- `GET /api/system/health` — `mlx` / `cuda` probe, GPU memory
- `GET /api/system/metrics` | `GET /api/settings/system`

### Audio (placeholder)

- `POST /api/audios/generations` | `edits` — enqueued; **no inference backend** → explicit failure in task log

Route sources: `backend/api/routes/*.py`. Live docs: `http://localhost:7860/docs`.

---

## Running

```bash
./bin/launch.sh          # macOS-oriented; venv + deps + optional frontend build
make start / make stop

source .venv/bin/activate
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 7860

make frontend-dev        # :5173 → proxy /api to :7860
make frontend-build      # → out/frontend/dist/
```

---

## Makefile targets

| Target | Purpose |
|--------|---------|
| `bench-setup` | Benchmark venv |
| `bench-mflux` / `bench-mflux-case ID=` | mflux PSNR |
| `bench-sanity` / `bench-sanity-case ID=` | output sanity |
| `test-engine-unit` | `scripts/test_engine_unit.py` |
| `check-consistency` | registry/routes/i18n |
| `check-engine-imports` | mlx/torch gate |
| `lint` | Python syntax |
| `frontend-install` / `frontend-dev` / `frontend-build` / `frontend-typecheck` | frontend |
| `pack-macos-desktop` | macOS Tauri `.app` / `.dmg` (MLX sidecar) |
| `pack-linux-server` | Linux CUDA `.tar.gz` server bundle (venv + sidecar + archive) |
| `pack-windows-desktop-release` | Windows CUDA NSIS installer (venv + Tauri; on Windows) |
| `pack-windows-server` | Windows CUDA `.zip` headless server (optional) |

Makefile pattern: `pack-<platform>-<product>-<step>` (`desktop` \| `server`; `venv` \| `sidecar` \| `shell` \| `archive`). Legacy names (`desktop-bundle`, `release-linux-cuda`, …) remain as aliases.
| `clean` | `scripts/clean_build.py` |

---

## Frontend (Vue 3 + Vite + TypeScript)

- Locales: `frontend/src/locales/zh.json`, `en.json`
- Vue I18n 9, `legacy: false`; use `useI18n()` in script setup
- Template: `$t('key')`; script: `$tt` from `@/utils/i18n`
- Model/preset names: `$mn`, `$md`, `$mvn`, `$pn` on `globalProperties`
- Storage keys: `@/utils/storage` → `DQ_STORAGE` (`dq-studio.*.v4`)
- API: `@/utils/api` (`export const api`)
- Stores: `@/stores/registry`, `@/stores/tasks` (queue poll + SSE)
- Memory soft warning: `@/composables/memoryHint` (`warnIfRiskyMemory`)
- Navigation: Vue Router `router.push({ name: '…' })` — no `window.DQStudioNav`
- Global task queue: `App.vue` + `TopNav.vue` + `useTasksStore`; `openGlobalTaskQueue()` from `@/utils/appEvents`

### Frontend UI (`@danqing/dq-ui`)

- No Element Plus: templates use `Dq*` only; tokens from `@danqing/dq-tokens` (`--dq-*`).
- Shell layout: `App.vue` → `.dq-app-header` + `.dq-app-main`; `TopNav` is native `<nav class="dq-top-nav-menu">`.
- Governance: `make check-ep-boundary`, `make check-theme-legacy`. See `frontend/DQ-UI.md`.

### App.vue routing

`activePage` is a **local `ref`** synced from `route.name` via `watch` — do not bind cross-module route refs directly.

---

## Backend i18n

- `backend/core/i18n.py` — `t(key, locale, **params)`, `resolve_locale(Accept-Language)`
- Messages: `config/locales/zh.json`, `en.json`
- Registry/presets: bilingual objects in JSON (not legacy `name_en` fields)

---

## Desktop packaging

- Build: `make pack-macos-desktop` or `scripts/build_desktop.sh`
- Artifacts: `out/frontend/dist/`, `out/sidecar/danqing-api/`, `out/desktop/bundle/`, `out/dist/*.tar.gz` (Linux CUDA server)
- macOS default: `DANQING_PYINSTALLER_PROFILE=mlx` (no torch / `*_cuda`)
- CUDA server/desktop: `make pack-linux-server` / `make pack-windows-desktop-release`; CI in `.github/workflows/build-desktop.yml`
- Sidecar env: `DANQING_HTTP_HOST`, `DANQING_HTTP_PORT`, `DANQING_USER_DATA_DIR`
- New engine modules must be reachable from `scripts/build_desktop.py` / PyInstaller hooks

---

## Gotchas

- **First run is slow** — model load into GPU memory
- **DB schema** — no migrations; delete `db/studio.db` (+ outputs) to reset
- **`launch.sh`** — checks/builds frontend; production path is `out/frontend/dist/` (see `backend/main.py::_resolve_frontend_static_dir`)
- **Add model** — full 5-step checklist above, not registry-only
- **OpenAI-compatible API** (if added later) — separate `/v1/...` adapter; must delegate to same contracts/engines; do not replace resource-style `/api/images/*` routes
- **Contributors** — run `make check-consistency` and `make check-engine-imports` before PR

---

## Reference

- [README.md](README.md) — user documentation (EN + 中文)
- [config/models_registry.json](config/models_registry.json)
- [backend/core/interfaces.py](backend/core/interfaces.py)
- [tests/benchmark/README.md](tests/benchmark/README.md)
