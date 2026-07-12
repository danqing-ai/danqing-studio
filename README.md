# DanQing Studio v4 (丹青工作室)

Language: **English** | [中文](README_zh.md)

Plugin-style image and video generation studio with **MLX** (Apple Silicon) and **CUDA** (NVIDIA) backends. Split stack: FastAPI backend, Vue 3 SPA, shared REST API and CLI, SQLite persistence, and full **zh/en** i18n.

| | |
|---|---|
| **Docs for contributors / agents** | [AGENTS.md](AGENTS.md) |
| **Desktop (Tauri 2)** | [desktop/README.md](desktop/README.md) |
| **Engine architecture** | [docs/engine_architecture.md](docs/engine_architecture.md) |
| **Image benchmarks** | [tests/benchmark/README.md](tests/benchmark/README.md) |

---

## Features

- **Dual runtime** — `MLXContext` on Apple Silicon; `CudaContext` when PyTorch CUDA is available (per-model `backends` in the registry).
- **Layered architecture** — REST / CLI → `TaskScheduler` → `DanQing*Engine` → `ImagePipeline` / `VideoPipeline` → `RuntimeContext` → SQLite.
- **Models as plugins** — New families touch registry JSON, config, `families/<family>/`, and `_transformer_registry.py`; the pipeline skeleton stays family-agnostic.
- **Contract-driven API** — Routes and CLI go through `backend/core/contracts.py` and `IImageEngine` / `IVideoEngine`; no per-model branches in route handlers.
- **Global task queue** — One worker, image/video (and audio placeholders) serialized; SSE progress, priority, queue position, persistent logs.
- **Studio UI** — Vue 3 + Vite + TypeScript + `@danqing/dq-ui` + Pinia; macOS-native dark theme; model names and presets are bilingual in the registry.
- **Four modules** — **Create** (image/video tabs filtered by model `actions`), **Gallery** (SQLite `assets`), **Models** (install/delete weights), **Settings** (presets, queue policy, system health).
- **Infinite canvas** (image / video / audio create) — Gallery **grid** and **canvas** views share one asset library; canvas sessions persist layout, lineage edges, and composer state per media type.

### Infinite canvas workflow

In **Create → Canvas view** (toggle at the top of the gallery strip):

1. **Import** — bottom-right **Import works** (`I`), gallery hover **Add to canvas**, or multi-select in grid then switch to canvas.
2. **Iterate** — select a node; the bottom **Composer** fills prompt/model; floating toolbar runs edit / branch / cover workflows.
3. **Generate** — outputs land in the **staging zone** (orange box); press `S` to snap staging beside the selection.
4. **Lineage** — parent→child SVG edges (`E`); session graph (`G`); lineage sidebar (`Y`) — click to focus on canvas, double-click to jump and close.
5. **Sessions** — top-left bar switches/creates/renames canvas sessions (synced via `/api/canvas/sessions`).

| Key | Action |
|-----|--------|
| `I` | Import works picker |
| `S` | Snap staging to selection |
| `R` | Region guides (staging + overlay links) |
| `L` / `G` / `E` | Layers / session graph / lineage edges |
| `Y` | Lineage sidebar |
| `F2` | Rename selected node |
| `Esc` | Close panel → clear selection |
| Space drag | Pan viewport |

Settings → **Auto-add results to canvas** keeps staging placement even when you stay in grid view during generation.

### ControlNet / structural guide (FLUX.1)

Invoke-style **structural conditioning** on image create (FLUX.1 base only, e.g. `flux1-dev`):

1. **Models** — install base `flux1-dev` and a ControlNet bundle (`flux-canny-controlnet`, `flux-depth-controlnet`, `flux-redux`, …). **Depth** also needs the `depth-pro` tool model; **Canny/Depth/Redux preprocess** uses OpenCV (Canny) or **PyTorch** (Depth Pro + SigLIP/Redux) on CPU.
2. **Composer** — advanced → ControlNet model + strength; pick a **structural guide** image (gallery asset). Selecting a controlnet applies registry defaults (e.g. Canny/Depth CFG ≈ 30).
3. **Canvas** — select a node → **Guide branch** or **Use as structural guide**; CTRL overlay syncs with the composer.
4. **Generate** — API sends `structural_guide` (`model_id`, `asset_id`, `type`, `weight`):
   - **Canny / Depth** — preprocess guide → VAE encode → 128-ch patch concat + companion LoRA (`flux1-canny-dev-lora` / `flux1-depth-dev-lora`).
   - **Redux** — SigLIP + redux MLP tokens concat to T5 context (no patch embed).
   - **Fill** (`flux-fill-controlnet`) — inpainting/outpainting only (retouch/extend); not available in text-to-image.

Structural guide cannot combine with reference img2img on the same request. Lineage uses `relation_type: controlnet` when a guide image is bound.

### Studio tabs ↔ model `actions`

Creation tabs only list models that declare the required `action` in the workspace `config/models_registry.json` (seeded from `default_config/`).

#### Image create

| Tab | Required action | API |
|-----|-----------------|-----|
| Text-to-image | `create` | `POST /api/images/generations` |
| Reference-driven edit | `rewrite` | `POST /api/images/edits` (`operation: rewrite`) |
| Instruct edit | `rewrite` | `POST /api/images/edits` (`operation: rewrite`) |
| Inpaint / retouch | `retouch` | `POST /api/images/edits` (`operation: retouch`) |
| Outpaint / extend | `extend` | `POST /api/images/edits` (`operation: extend`) |
| Upscale | `upscale` | `POST /api/images/upscales` |

#### Video create

| Tab | Required action | API |
|-----|-----------------|-----|
| Text-to-video | `create` | `POST /api/videos/generations` |
| Image-to-video | `animate` | `POST /api/videos/edits` |

#### Audio (placeholder)

Audio routes accept tasks but **fail explicitly** in the task log until an inference backend exists.

---

## Requirements

| Platform | Notes |
|----------|--------|
| **macOS (Apple Silicon)** | Primary target; MLX via Metal. `make dev` expects macOS + Python 3.11. |
| **Linux / Windows + NVIDIA** | CUDA path when `torch` + CUDA are installed; not all families ship `*_cuda.py` yet — missing capability **fails loud**, no silent fallback. |
| **Python** | 3.11+ (`.venv/` at repo root) |
| **RAM** | 32 GB+ recommended for large models |
| **Node.js** | For frontend dev/build and desktop packaging |
| **ffmpeg / ffprobe** | Video thumbnails and duration metadata (optional but recommended) |

---

## Quick start

### Install

```bash
git clone <repo-url> DanQing-Studio
cd DanQing-Studio

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run (web)

```bash
# 开发：uvicorn --reload + Vite HMR（一键启停）
make dev
# 或 make start / make stop
```

Open **http://localhost:5800** (Vite, proxies `/api` → :7800) or **http://localhost:7800** — Swagger at **/docs**.

### Dev ports (DanQing family)

Backend **`78xx`**, frontend **`58xx`** — same last two digits = same project. All three repos can run `make dev` at once.

| Project | Backend | Frontend (Vite) |
|---------|---------|-------------------|
| **Studio** | 7800 | 5800 |
| Teams | 7801 | 5801 |
| Mail | 7802 | 5802 |

Override: `DQ_BACKEND_PORT`, `DQ_FRONTEND_PORT` (see `scripts/out_paths.sh`).

### Release

```bash
make pack-macos-desktop     # macOS .app / .dmg → out/desktop/bundle/
make pack-linux-server      # Linux CUDA tar.gz → out/dist/
make pack-windows-desktop-release   # Windows NSIS (on Windows)
```

### CLI examples

```bash
bin/danqing-generate --model flux2-klein-9b --prompt "a cat on a windowsill"
bin/danqing-edit --model <id> --image input.png --prompt "add a hat" --operation rewrite
bin/danqing-upscale --model <id> --image input.png
bin/danqing-video-generate --model <id> --prompt "ocean waves at sunset"
```

Full CLI ↔ REST mapping: [AGENTS.md](AGENTS.md#cli-vs-rest-api).

### Models on disk

| Path | Purpose |
|------|---------|
| `./models/` | Checkpoints (`.safetensors`, `.bin`, diffusers layouts, `model_index.json`) |
| `./models/Lora/` | LoRA weights |
| `./outputs/` | Generated files |
| `./db/studio.db` | Tasks + assets (SQLite WAL) |

Install weights from the **Models** page or `POST /api/models/{id}/install` (progress via download SSE).

### Frontend dev (hot reload)

`make dev` 已同时启动 API（:7800，--reload）与 Vite（:5800）。也可单独：

```bash
make frontend-dev   # Vite on :5800, proxies /api → :7800
```

### Benchmarks (optional)

Uses an isolated venv under `tests/benchmark/venv/`:

```bash
make bench-setup
make bench-eval-smoke   # image model eval (L1 + ImageReward, fast)
make bench-eval         # full prompt matrix
make verify-engine-stack   # governance gates + engine unit tests
```

---

## Project layout

```
DanQing-Studio/
├── backend/
│   ├── api/routes/          # REST (images, videos, tasks, assets, registry, …)
│   ├── cli/                 # bin/danqing-* (mirrors REST)
│   ├── core/                # contracts, interfaces, DI container, i18n
│   ├── engine/
│   │   ├── pipelines/       # ImagePipeline, VideoPipeline, …
│   │   ├── families/        # Per-model transformers (flux1, flux2, z_image, ltx, …)
│   │   ├── runtime/         # MLXContext, CudaContext (only place for mlx/torch imports)
│   │   └── common/          # VAE, schedulers, text encoders, TransformerBase
│   ├── persistence/         # V3TaskStore, SQLiteAssetStore
│   ├── scheduler/           # Global TaskScheduler
│   └── main.py              # FastAPI entry
├── frontend/                # Vue 3 + Vite + TypeScript
├── desktop/                 # Tauri 2 shell
├── bin/                     # danqing-* CLI
├── default_config/          # factory models_registry, presets, locales, workspace.pointer
├── scripts/                 # build, lint gates, desktop packaging
├── tests/benchmark/         # image eval benchmark (L1+L2)
├── docs/                    # engine_architecture.md (single engine doc)
├── models/  outputs/  db/
└── out/                     # Build artifacts (gitignored)
    ├── frontend/dist/       # Vite production build
    ├── sidecar/             # PyInstaller danqing-api
    └── desktop/bundle/      # .app / .dmg
```

---

## Architecture (summary)

```
REST API (FastAPI)  ||  CLI (bin/danqing-*)
         ↓                    ↓
    TaskScheduler  (single global queue, serial worker)
         ↓
DanQingImageEngine / DanQingVideoEngine / DanQingAudioEngine
         ↓
ImagePipeline / VideoPipeline  (registry-driven assembly line)
         ↓
RuntimeContext (MLX | CUDA) + TransformerBase families + common components
         ↓
V3TaskStore + SQLiteAssetStore
```

**Adding a model** (5 steps): registry JSON → `model_configs.py` → `families/<family>/transformer.py` → `weights.py` (`remap_*`) → `_transformer_registry.py`. Details: [AGENTS.md](AGENTS.md#new-model-checklist).

---

## Desktop app

Platform-specific sidecars keep bundles small — **never mix MLX + CUDA in one release**:

| Platform | Profile | Backend | Make target |
|----------|---------|---------|-------------|
| macOS (Apple Silicon) | `mlx` | MLX / Metal | `make pack-macos-desktop` |
| Linux x86_64 server | `cuda` | PyTorch CUDA | `make pack-linux-server` |
| Windows x64 desktop | `cuda` | PyTorch CUDA | `make pack-windows-desktop-release` |

```bash
make pack-macos-desktop          # MLX-only .dmg
make pack-linux-server           # CUDA server .tar.gz
make pack-windows-desktop-release  # CUDA NSIS (on Windows)
```

GitHub tag builds use the same split (`.github/workflows/release.yml`).

See [desktop/README.md](desktop/README.md).

---

## Configuration

**App settings** — `{workspace}/config/.app_config.json`:

```json
{
  "language": "en",
  "theme": "dark",
  "default_model": "flux2-klein-9b",
  "mlx_memory_limit": 120,
  "queue_image_first": true
}
```

**Model registry** — `{workspace}/config/models_registry.json` (`schema_version: 3`; factory copy in `default_config/`): nested `catalog` / `runtime` / `ui` / `distribution`; API returns `CatalogResponse` DTO via `GET /api/registry`.

**Environment** (optional `.env`):

```bash
HF_ENDPOINT=https://hf-mirror.com
HF_HUB_ENABLE_HF_TRANSFER=1
MLX_METAL_DEVICE_ONLY=1
MLX_METAL_MEMORY_LIMIT=120
```

---

## Development

| Command | Purpose |
|---------|---------|
| `make dev` / `make start` / `make stop` | Dev: uvicorn --reload + Vite HMR |
| `make pack-macos-desktop` | macOS desktop release |
| `make pack-linux-server` | Linux server release |
| `make frontend-dev` | Vite dev server |
| `make frontend-build` | Production UI → `out/frontend/dist/` |
| `make frontend-typecheck` | `vue-tsc` |
| `make frontend-canvas-unit` | Canvas edge/staging util self-check |
| `make check-consistency` | Registry / routes / i18n + frontend governance (incl. canvas unit) |
| `make check-engine-imports` | mlx/torch import boundary |
| `make lint` | Python syntax check |
| `make clean` | Remove `out/` build tree |

Backend reload: `python3 -m uvicorn backend.main:app --reload --port 7800`

---

## API overview

| Area | Endpoints |
|------|-----------|
| Images | `POST /api/images/generations`, `edits`, `upscales` |
| Videos | `POST /api/videos/generations`, `edits` |
| Tasks | `GET/PATCH/DELETE /api/tasks/{id}`, `GET …/stream` (SSE), `GET /api/queue` |
| Assets | `GET/POST /api/assets`, `…/file`, `…/thumbnail`, `POST …/reconcile` |
| Models | `GET /api/models`, `POST /api/models/{id}/install`, registry at `GET /api/registry` |
| System | `GET /api/system/health`, `GET /api/settings/system` |

Interactive docs: **http://localhost:7800/docs**

---

## License

MIT

