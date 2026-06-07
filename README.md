# DanQing Studio v4 (丹青工作室)

**English** · [中文](#中文)

Plugin-style image and video generation studio with **MLX** (Apple Silicon) and **CUDA** (NVIDIA) backends. Split stack: FastAPI backend, Vue 3 SPA, shared REST API and CLI, SQLite persistence, and full **zh/en** i18n.

| | |
|---|---|
| **Docs for contributors / agents** | [AGENTS.md](AGENTS.md) |
| **Desktop (Tauri 2)** | [desktop/README.md](desktop/README.md) |
| **Dual-platform engine design** | [docs/dual_platform_architecture.md](docs/dual_platform_architecture.md) |
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
├── docs/                    # dual_platform_architecture.md
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

**Model registry** — `{workspace}/config/models_registry.json` (`schema_version: 2`; factory copy in `default_config/`): `engines`, `actions`, `parameters`, `versions`, bilingual `name` / `description`.

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

---

<a id="中文"></a>

## 中文

**丹青工作室 v4** — 基于 **MLX / CUDA** 双后端的插件化图像/视频生成引擎。前后端分离：FastAPI + Vue 3 SPA，REST 与 CLI 语义一致，SQLite 持久化任务与资产，界面与错误信息支持 **中英文**。

| | |
|---|---|
| **贡献者 / Agent 文档** | [AGENTS.md](AGENTS.md) |
| **桌面版（Tauri 2）** | [desktop/README.md](desktop/README.md) |
| **双平台引擎设计** | [docs/dual_platform_architecture.md](docs/dual_platform_architecture.md) |
| **图像基准测试** | [tests/benchmark/README.md](tests/benchmark/README.md) |

### 特性

- **双运行时**：Apple Silicon 上 MLX；安装 PyTorch CUDA 时可用 CUDA（以注册表 `backends` 为准）。
- **分层架构**：REST / CLI → 全局 `TaskScheduler` → 引擎 → Pipeline → `RuntimeContext` → SQLite。
- **模型即插件**：新模型主要改注册表、`model_configs`、`families/<family>/`、`_transformer_registry`；Pipeline 不写 `family ==` 分支。
- **契约化 API**：路由与 CLI 经 `contracts` 与 `IImageEngine` / `IVideoEngine` 进入引擎。
- **全局单队列**：图像/视频（及音频占位）串行执行；SSE 进度、优先级、队列位置、日志落库。
- **四大模块**：**创作**（按 `actions` 过滤模型）、**图库**（`assets` 表）、**模型**（安装/删除权重）、**设置**（预设、队列策略、系统状态）。
- **无限画布**（图像 / 视频 / 音频创作页）— 画廊 **网格** 与 **画布** 共用作品库；画布会话持久化排版、谱系连线与创作器状态（按媒介隔离）。

### 无限画布工作流

在 **创作页 → 画布视图**（顶部画廊条切换）：

1. **导入** — 右下「导入作品」（`I`）、画廊悬停「添加到画布」，或网格多选后切画布。
2. **迭代** — 选中节点，底部 **创作器** 灌参；浮动工具栏精修 / 分支 / 翻唱等。
3. **生成** — 新结果落入 **生成落点**（橙色框）；`S` 贴靠选中节点。
4. **谱系** — 父子连线（`E`）、会话图谱（`G`）、谱系侧栏（`Y`）；单击定位、双击关闭并跳转。
5. **会话** — 左上会话栏切换/新建/重命名（`/api/canvas/sessions` 同步）。

常用快捷键：`I` 导入 · `S` 落点贴靠 · `R` 区域引导 · `L/G/E` 图层/图谱/连线 · `Y` 谱系 · `F2` 重命名 · `Esc` 关面板/取消选择 · 空格拖移平移。

设置中的 **生成自动加入画布** 可在画廊视图下仍将结果落入当前会话落点区。

### ControlNet / 结构引导（FLUX.1）

Invoke 风格 **结构条件**（仅 FLUX.1 基底，如 `flux1-dev`）：

1. **模型页** — 安装 `flux1-dev` 与 ControlNet 包（`flux-canny-controlnet`、`flux-depth-controlnet`、`flux-redux` 等）。**Depth** 另需 `depth-pro` 工具模型；**Canny/Depth/Redux 预处理** 使用 OpenCV（Canny）或 **PyTorch**（Depth Pro + SigLIP/Redux，CPU）。
2. **创作器** — 高级参数选 ControlNet + 强度；绑定 **结构引导图**（须为画廊资产）。切换 ControlNet 会套用注册表推荐参数（如 Canny/Depth CFG ≈ 30）。
3. **画布** — 选中节点 →「结构引导分支」或「用作结构引导」；CTRL 叠加层与创作器同步。
4. **生成** — 请求携带 `structural_guide`：
   - **Canny / Depth** — 预处理 → VAE 编码 → 128 通道 patch 拼接 + 配套 LoRA（`flux1-canny-dev-lora` / `flux1-depth-dev-lora`）。
   - **Redux** — SigLIP + redux MLP 令牌并入 T5 上下文（无 patch embed）。
   - **Fill**（`flux-fill-controlnet`）— 仅局部重绘/扩图（retouch/extend），文生图不可用。

结构引导不能与参考图 img2img 同请求并用；绑定引导图时谱系为 `relation_type: controlnet`。

### 创作页 ↔ 模型 `actions`

与上文英文表格一致：文生图 `create`；改图 `rewrite`；局部修饰 `retouch`；扩展 `extend`；放大 `upscale`；文生视频 `create`；图生视频 `animate`。完整列表见工作区 `config/models_registry.json`（出厂默认在 `default_config/`）。

音频接口已入队，但**无推理后端**时会在任务日志中**显式失败**（不静默降级）。

### 环境要求

| 平台 | 说明 |
|------|------|
| **macOS（Apple Silicon）** | 主要目标；`make dev` 面向 macOS + Python 3.11 |
| **Linux / Windows + NVIDIA** | 需 `torch` + CUDA；部分 family 尚无 `*_cuda.py` 时会**明确报错** |
| **Python** | 3.11+，仓库根 `.venv/` |
| **内存** | 大模型建议 32 GB 及以上 |
| **Node.js** | 前端开发与桌面打包 |
| **ffmpeg** | 视频缩略图与时长（建议安装） |

### 快速开始

```bash
cd DanQing-Studio
make dev
# 或 make start / make stop
```

浏览器打开 **http://localhost:5800**（Vite，代理 `/api` → :7800），API 文档 **/docs**。

发布：`make pack-macos-desktop` / `make pack-linux-server`

```bash
# 前端热更新（API 另开终端跑在 7800）
make frontend-dev

# CLI 示例
bin/danqing-generate --model flux2-klein-9b --prompt "窗台上的猫"
```

权重目录：`./models/`；LoRA：`./models/Lora/`；输出：`./outputs/`；数据库：`./db/studio.db`。

### 项目结构

与英文「Project layout」一致：`backend/engine/families/` 存放各模型族；`out/frontend/dist/` 为 Vite 生产构建产物。

### 架构概要

```
REST / CLI → TaskScheduler → DanQing*Engine → Pipeline → RuntimeContext → SQLite
```

新模型五步：注册表 → `model_configs` → `transformer.py` → `remap_*` → `_transformer_registry`。详见 [AGENTS.md](AGENTS.md)。

### 桌面版

```bash
make pack-macos-desktop
```

说明见 [desktop/README.md](desktop/README.md)。macOS 仅 MLX sidecar；Linux/Windows 仅 CUDA sidecar（`DANQING_PYINSTALLER_PROFILE=cuda`）。

### 配置

- `default_config/workspace.pointer.json` — 自定义工作区路径（本地，不进 git）
- `{workspace}/config/.app_config.json` — 语言、主题、默认模型、`mlx_memory_limit`、`queue_image_first` 等
- `{workspace}/config/models_registry.json` — 模型能力、`actions`、`parameters`、双语名称
- `{workspace}/config/presets.json` — 提示词预设（须含 `applies_to`、`media_scope`）

### 常用 Make 目标

| 命令 | 用途 |
|------|------|
| `make start` / `stop` | 启停 API |
| `make frontend-dev` / `frontend-build` | 前端开发 / 构建 |
| `make frontend-canvas-unit` | 画布谱系/落点工具自检 |
| `make check-consistency` | 注册表与路由一致性（含画布单元测试） |
| `make check-engine-imports` | mlx/torch 导入边界检查 |
| `make bench-eval` / `bench-eval-smoke` | 图像模型 L1+L2 评测 |
| `make pack-macos-desktop` | 完整桌面安装包 |

### 许可证

MIT
