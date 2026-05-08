# AGENTS.md - DanQing Studio v4 (丹青工作室)

## What this is
丹青工作室 — MLX/CUDA 双后端插件化图像/视频生成引擎。前后端分离。

## Entry points
- `backend/main.py` — FastAPI 后端入口
- `frontend/index.html` — Vue 3 前端入口
- `bin/launch.sh` — 启动脚本
- `bin/stop.sh` — 停止脚本
- `requirements.txt` — Python 依赖（不含 mflux/mlx-video）
- `Makefile` — 基准测试 / lint

## Environment & dependencies
- **Python**: 3.11+
- **Venv**: `.venv/` in repo root
- **Key packages**: `fastapi`, `uvicorn`, `mlx`, `Pillow`, `transformers`, `safetensors`

> **mflux / mlx-video 不在项目 venv 中**。它们是旁路基准测试的依赖，安装在独立虚拟环境 `tests/benchmark/venv/`。
> 如需研究 mflux / mlx-video 源码实现，查 `tests/benchmark/venv/lib/python3.11/site-packages/mflux/` 和 `tests/benchmark/venv/lib/python3.11/site-packages/mlx_video/`。运行 `make bench-setup` 初始化该环境。

## Architecture (v4.0)

### 设计原则

#### 分层（Layering）
各层单向依赖，只通过接口契约通信：
```
REST API / CLI (入口层，平行)
    ↓ 仅依赖 contracts + interfaces
TaskScheduler (全局单队列，图像/视频共享，串行执行)
    ↓ 仅依赖 IImageEngine / IVideoEngine
DanQingImageEngine / DanQingVideoEngine (引擎层，路由到 Pipeline)
    ↓ 仅依赖 Pipeline + RuntimeContext
ImagePipeline / VideoPipeline (装配线：编码器 + 调度器 + 去噪 + VAE)
    ↓ 仅依赖 RuntimeContext 接口 + 配置数据
RuntimeContext (MLX / CUDA) + Transformer Models + Common Components
    ↓ 无业务依赖
V3TaskStore + SQLiteAssetStore (持久化，WAL 模式并发读写)
```

#### 模块化（Modularity）
- `backend/api/routes/` — REST 路由，按媒体类型拆分
- `backend/cli/` — CLI 命令，与 REST API 一一对应，共享 Engine 层
- `backend/scheduler/` — TaskScheduler，全局单 Worker 串行队列
- `backend/engine/` — 引擎 + Pipeline + 运行时 + 模型
- `backend/persistence/` — SQLite 持久化（WAL 模式）
- `backend/core/` — 接口定义 + 契约 + 容器
- `backend/services/` — 业务服务（设置、下载等）

#### 组件化（Componentization）
可替换组件通过接口注入，核心零修改：
- **RuntimeContext**：`MLXContext` / `CudaContext`，封装硬件操作
- **Scheduler**：`FlowMatchEulerScheduler` / `LinearScheduler`，去噪时间步策略
- **VAEDecoder**：参数化 `scaling_factor` / `shift_factor` / `pytorch_compatible`
- **TextEncoder**：`T5Encoder` / `Qwen3TextEncoder` / `CLIPEncoder`
- **ModelCache**：LRU 模型缓存，自动内存管理

#### 模型插件化（Model as Plugin）
新增模型 = 注册表 JSON + Config 数据类 + Transformer 类 + 权重 Remap：
- 注册表声明：`config/models_registry.json`（family / engine / actions / parameters / versions）
- 配置数据类：`backend/engine/config/model_configs.py`（in_channels / hidden_dim / encoder_type 等）
- Transformer 实现：`backend/engine/models/image/{family}.py`
- 权重映射：`backend/engine/common/weights.py`（diffusers → DanQing 键名转换）
- Pipeline 路由：`backend/engine/image_pipeline.py` 中 `family == "xxx"` 分支

#### Pipeline 组装化（Pipeline as Assembly Line）
`ImagePipeline.run_mlx()` 是装配线，由 `entry.family` 和 `config.*` 驱动：
1. 解析 model → 查注册表 → 获取 family / config
2. 按 `config.encoder_type` 选择文本编码器
3. 按 `family` 选择调度器（`_scheduler_name_for_family`）
4. 创建初始 latent（seed 确定性）
5. 跑通用去噪循环（`DenoisingPipeline` 或 inline loop）
6. VAE 解码（读取 `vae/config.json` 参数）
7. 资产落盘（`asset_store.create_from_file`）

**核心不变量**：新增模型不触碰 API / CLI / Engine / Scheduler / Persistence / 通用组件代码。

### 后端分层实现
```
API Layer (FastAPI)  ||  CLI Layer (bin/danqing-*)
    ↓                    ↓
TaskScheduler (全局串行队列，asyncio.PriorityQueue)
    ↓
DanQingImageEngine / DanQingVideoEngine (IImageEngine / IVideoEngine)
    ↓
ImagePipeline / VideoPipeline (后端无关，去噪循环 + VAE + 资产)
    ↓
RuntimeContext (MLX / CUDA) + Transformer models + Common components
    ↓
V3TaskStore + SQLiteAssetStore (SQLite WAL 模式持久化)
```

### 核心目录
- `backend/core/` — 接口定义（`media_interfaces.py` / `interfaces.py`）+ 契约（`contracts.py`）+ 容器
- `backend/engine/` — 引擎 + Pipeline + 运行时 + 模型 + 调度器 + VAE
- `backend/persistence/` — SQLite 持久化（WAL 模式，`V3TaskStore` + `SQLiteAssetStore`）
- `backend/services/` — 业务服务层（设置、下载等）
- `backend/api/routes/` — REST API 路由
- `backend/cli/` — CLI 命令（与 REST API 一一对应）
- `backend/scheduler/` — TaskScheduler（全局单 Worker）

### CLI 与 REST API 映射
| CLI 命令 | REST API 端点 | Engine 方法 |
|----------|---------------|-------------|
| `bin/danqing-generate` | `POST /api/images/generations` | `IImageEngine.generate()` |
| `bin/danqing-edit` | `POST /api/images/edits` | `IImageEngine.edit()` |
| `bin/danqing-upscale` | `POST /api/images/upscales` | `IImageEngine.upscale()` |
| `bin/danqing-video-generate` | `POST /api/videos/generations` | `IVideoEngine.generate()` |
| `bin/danqing-video-edit` | `POST /api/videos/edits` | `IVideoEngine.edit()` |

### 引擎与模型扩展
`ModelRegistry`（`backend/core/model_registry.py`）加载 `models_registry.json`；`EngineRegistry`（`backend/engine/engine_registry.py`）按 `engine` 字段绑定 `DanQingImageEngine` / `DanQingVideoEngine`。模型 Transformer 在 `backend/engine/models/image/`（图像）和 `backend/engine/models/video/`（视频）。

### 新模型接入流程（5 步）

以接入 `longcat-image` 为例：

**Step 1: 注册表声明** — `config/models_registry.json`
```json
"longcat-image": {
  "family": "longcat",
  "engine": "danqing-image",
  "media": "image",
  "actions": { "create": {}, "rewrite": {} },
  "parameters": { "steps": { "type": "int", "default": 4 }, ... },
  "versions": { "fp16": { "default": true, "local_path": "models/Base/longcat-image-fp16" } }
}
```

**Step 2: 配置数据类** — `backend/engine/config/model_configs.py`
```python
@dataclass
class LongCatConfig:
    in_channels: int = 64
    hidden_dim: int = 3072
    encoder_type: str = "qwen2.5_vl"  # 或 "t5"
    # ... 其他架构参数
```
`get_config_class("longcat")` 返回 `LongCatConfig`。

**Step 3: Transformer 实现** — `backend/engine/models/image/longcat.py`
```python
class LongCatTransformer(nn.Module):
    def __init__(self, config: LongCatConfig, ctx: RuntimeContext):
        # 按 config 构建网络结构
    def __call__(self, latents, timestep, **kwargs):
        # 前向推理
```

**Step 4: 权重映射** — `backend/engine/common/weights.py`
```python
def remap_longcat_weights(weights: dict) -> dict:
    # diffusers 键名 → DanQing 键名转换
    return remapped
```

**Step 5: Pipeline 路由** — `backend/engine/image_pipeline.py`
在 `run_mlx()` 中：
- 移除 `family == "longcat"` 的 `raise RuntimeError`
- 按 `config.encoder_type` 选择文本编码器
- 按 `family` 选择调度器（`_scheduler_name_for_family`）
- 处理模型特有的 conditioning 格式

**复用 vs 扩展：**
| 类别 | 内容 | 是否修改 |
|------|------|----------|
| 直接复用 | `DanQingImageEngine`、`ImagePipeline` 装配逻辑、`DenoisingPipeline` 去噪循环、`RuntimeContext`、`Scheduler` 基类+实现、`VAEDecoder`、CLI/REST API 接入层、`ModelCache`、进度回调/SSE | 零修改 |
| 配置扩展 | `models_registry.json` 条目、`LongCatConfig` 数据类 | 仅改声明 |
| 新增实现 | `LongCatTransformer`（注意力/MLP/RoPE/条件注入）、`remap_longcat_weights`、文本/视觉编码器适配、Pipeline 路由分支 | 模型特有逻辑 |

**验证路径：**
```
bin/danqing-generate --model longcat-image --prompt "..."
  → DanQingImageEngine.generate()
    → ImagePipeline.run_mlx()
      → LongCatTransformer + Scheduler + VAEDecoder
```

Benchmark 测试：
```
tests/benchmark/runner.py → subprocess(danqing-generate)
  → 与 mflux CLI 输出对比（PSNR ≥ 30 dB）
```

## Configuration

### 模型注册表（v2）
`config/models_registry.json` — `schema_version: 2`：
- 顶层 `engines`（danqing-image / danqing-video）
- 每个模型：`name` / `description` 为 `{ "zh", "en" }`；`media`：`image` | `video`
- **`actions`** 替代 `capabilities`：图像用 `create` / `rewrite` / `retouch` / `extend` / `upscale`；视频用 `create` / `animate`
- `parameters`：条目带 `type`（`int` | `float` | `enum` | `bool` | `object`）
- 格式转换脚本：`scripts/migrate_models_registry.py`（写回前在同目录生成 `.pre_v2.*.json` 备份；仓库不提交该备份）

### 设置文件
- `config/.app_config.json` — 应用设置
- `db/studio.db` — SQLite（`tasks` / `task_logs` + `assets` 表；WAL 模式）
- `config/presets.json` — 提示词预设（必填 **`applies_to`** 与 **`media_scope`**：`image` \| `video` 二选一）；创作页按动作 Tab + `media_scope` 过滤；格式异常时运行 `scripts/migrate_presets_mode_to_applies.py --write`（默认 dry-run；写回前生成 `*.pre_applies_migrate.*.json` 备份，勿提交备份）

## Hardcoded paths
- Models: `./models/`
- LoRAs: `./models/Lora/`
- Outputs: `./outputs/`
- Configs: `./config/.app_config.json`, `./config/presets.json`, `./config/models_registry.json`

## API Endpoints

### 图像
- `POST /api/images/generations` — 文生图
- `POST /api/images/edits` — 编辑
- `POST /api/images/upscales` — 放大

### 视频
- `POST /api/videos/generations`
- `POST /api/videos/edits`

### 任务与队列（全局单队列）
- `GET /api/tasks` — 任务列表（与 **`/api/tasks/list`** 等价；`limit`/`offset`；`kind`/`status`/`since`）
- `GET /api/tasks/{id}` — 任务详情（含 **`queue_position`** / **`estimated_*`** / **`model`** / **`error`** 等字段）
- `GET /api/tasks/{id}/logs` — 历史日志分页（`offset`/`limit`）
- `PATCH /api/tasks/{id}` — 仅 **`queued`** 可改 `{ "priority": "normal" | "high" }`（与提交时语义一致；调度器重建堆 + 进程重启后从 DB 恢复排队）
- `DELETE /api/tasks/{id}` — 取消
- `GET /api/tasks/{id}/stream` — SSE：`log`（含 **`ts`**）、**`progress`**（step/total/eta）、**`status`**、**`result`**（完成时）、**`done`**
- `GET /api/queue` — 运行中 / 排队快照（queued 项含 **`estimated_wait_seconds`**）；设置页「系统」**队列快照**亦经 **`api.gen.getQueue`**；**顶栏右侧**角标 + **`el-drawer`** 全局任务列表（`TasksStore` 由 **`app.js`** 统一轮询；`open-global-task-queue` 事件仍可由其他 UI 复用以打开同一抽屉）
- 任务 kind 常量：`backend/core/task_kinds.py`（勿手写 `image.generation` 等字符串）
- 设置 **`queue_image_first`**：图像任务先于视频出队（见 `AppSettings` / 设置页开关）

### 资产
- `POST /api/assets` — 上传资产
- `POST /api/assets/reconcile` — **磁盘对账**：比对 `assets.file_path` 与磁盘；默认 `{"dry_run": true}` 仅报告 `missing_asset_ids`；`dry_run: false` 时删除库中主文件已不存在的行（设置页 / CLI 调用后续可接）
- `GET /api/assets` — 列表（可选 **`kind`**、**`source_task_id`**、**`created_after`** ISO 下界）；`GET /api/assets/{id}/file` — 主文件
- `GET /api/assets/{id}/thumbnail` — 缩略图（图像 WebP 派生；视频依赖本机 **ffmpeg/ffprobe** 生成 poster）；任务产出的 `assets.metadata` 由管线写入 **steps / guidance / seed / mime_type** 及 **width/height**；视频 **`duration_seconds`**：ffprobe 优先，失败时用 **`num_frames/fps`** 写入列与 metadata

### 注册表与发现
- `GET /api/registry` — 完整 `models_registry.json` + `_index`（family/media/actions）
- `GET /api/models` — 轻量索引（可选 **`media`** / **`action`** / **`installed`**）；响应项含 **`installed`**；`GET /api/models/{id}` — 单模型；`POST /api/models/{id}/install` — 安装权重（进度 SSE 仍为 `GET /api/download/progress/{task_id}/stream`）；`POST /api/models/install-batch` — 批量启动安装（body: `{ "model_ids": [...] }`）；`DELETE /api/models/{id}/versions/{version_key}` — 删除该版本本地权重目录
- `GET /api/presets` — 预设只读列表（写入仍用 `/api/settings/presets`）
- `GET /api/adapters` — 适配器索引（当前为已安装 LoRA；可选 `for_model` 查询参数与兼容 LoRA 规则一致；`registry_slots` 预留给注册表扩展）
- `GET /api/system/health` — 存活 + 后端探测（**`mlx`** / **`cuda`**）+ **`gpu.memory_total`/`free`**；`GET /api/system/metrics` — CPU/内存轻量快照
- `GET /api/settings/system` — 系统信息（`memory_gb`、`mlx_memory_limit`、`env_ready` 等）；前端用 `mlx_memory_limit` 与注册表版本 **`size`** 做提交前 OOM 软提示（`DQMemoryHint`）

### 音频（占位）
- `GET|POST /api/audios/generations`、`POST /api/audios/edits`、`POST /api/audios/dubs` — 501；前端 **`api.audios.*`**（`api.js`）与上述路径对齐

### 其他
- `GET /api/gallery/images` — 后端图库列表（**仅** `list_assets`）；项含 **`duration_seconds`**（视频）；前端 **`api.gallery.listImages`** 走 **`api.gen.listAssets`** + **`assetRowToGalleryItem`** 映射为同一卡片字段；`POST /api/gallery/upload` 入库；媒体 **`GET /api/assets/{id}/file`**；缩略图 **`GET /api/assets/{id}/thumbnail`**；删除 **`DELETE /api/gallery/image?path=asset:{id}`**
- `GET /api/settings/registry` — 与设置页兼容的注册表视图
- 路由源码：`backend/api/routes/{images,videos,tasks,queue,assets,registry,...}.py`
- 完整文档: http://localhost:7860/docs

## Running
```bash
# 启动 (自动检查依赖)
./bin/launch.sh

# 手动启动
source .venv/bin/activate
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 7860
```

## Conventions
- 面向接口编程，各层只依赖 `backend/core/interfaces.py`
- 暗色主题，主色调 `#e94560`
- 深度国际化 (i18n)，支持 zh/en 完整切换
- 任务持久化: SQLite 存储任务状态和日志

## i18n Architecture
### Frontend
- `frontend/js/i18n.js` — All translation keys (zh + en)；顶层 `action.image.*` / `action.video.*`（与注册表 `actions` 键对齐）及 **`studio.*`**（模型下拉、队列、任务状态、提示词/步数、生成与日志文案、模型切换、高级参数/ControlNet/LoRA、蒙版工具栏、上传失败等跨页共用文案）；**`video.runtime*`** — 视频创作页成片时长 / 耗时与磁盘占用提示
- Vue I18n 9 (Composition API, `legacy: false`)
- `$t('section.key')` for template strings
- `$tt('section.key', {params})` for JS code strings
- `$mn(model)` / `$md(model)` for bilingual model names/descriptions
- `$pn(preset, chineseName)` for bilingual preset names
- Language / 导航 / 设置页等持久化键：`frontend/js/storage_keys.js` → `window.DQ_STORAGE`（`dq-studio.*.v3`，不读旧 `danqing-*` 键）
- 前端缓存：`frontend/js/api.js` 末尾挂载 **`window.api`**（与 `const api` 同源）；`frontend/js/stores/registry.js`（`RegistryStore`，优先 `api.registry.getFull`）、`frontend/js/stores/tasks_store.js`（`api.gen.getQueue` 轮询 + `api.tasks.logStreamUrl` SSE）；`frontend/js/components/AdapterPicker.js`（LoRA / 适配器选择，由 `RegistryParamsForm` 使用）；`frontend/js/components/AssetPicker.js`（参考图 / 编辑图 / 视频起始图 / 蒙版编辑器空态：上传 + 最近条 + `/api/assets` 资产库）；创作页产出预览 **`api.gallery.getImageUrl('asset:{id}')`**；编辑/参考/视频首帧字节 **`api.gen.urlToBlob`**（`blob:`/`data:` 走 `fetch`，同源走 `axios`）；`ImageEditor.js` 蒙版快捷键绑在组件根 `tabindex` + `@keydown`（非 `window`）；`frontend/js/composables/media_queue.js` 挂载 **`window.DQMediaQueue`**（`normalizeTaskRow`、`snapshotFullQueue` 顶栏全队列；`tasksForMedia` 按 `kind` 前缀过滤，供顶栏等消费）；`frontend/js/composables/memory_hint.js` 挂载 **`window.DQMemoryHint`**（`warnIfRisky` 提交前软警告）；**`task_status_ui.js`**（`DQTaskStatusUi`）、**`model_version_value.js`**（`DQModelVersionValue`）、**`studio_nav.js`**（`DQStudioNav.goSettings` / `goModels`，`navigate` 事件）
- Components in `frontend/js/components/` use `$t()` and `$tt()` universally

### Backend
- `backend/core/i18n.py` — Translation service
- `config/locales/zh.json` and `config/locales/en.json` — Error/status messages
- `t(key, locale, **params)` for backend translations
- `resolve_locale(accept_language)` for request-level language detection
- Backend reads `Accept-Language` header; falls back to `AppSettings.language`

### Config files with i18n
- `models_registry.json` — Has `name_en`/`description_en` per model
- `presets.json` — Has `name_en` per preset
- Shell scripts (`bin/launch.sh`, `bin/stop.sh`) — English console output

## Gotchas

### Fail loud：禁止静默降级 / 静默兼容回退（与引擎实现一致）
- **默认**：影响成片质量、耗时、行为语义的路径（图像 / 视频管线、模型加载、解析注册表等）**不得**在失败或缺能力时 **静默降级**、**静默兼容回退**（例如换一条更弱的 CLI、套用无关模型的默认配置、`except: pass` 吞错后继续当成功路径）。
- **必须显式失败**：应 **`RuntimeError` / 明确 HTTP 错误**，并尽量走 **`config/locales/zh.json` + `en.json`**（`t(...)`）与 **任务日志 / SSE**，让使用者从 UI 与日志能直接看到原因。
- **确有必要例外**（极少数：例如仅运维、仅诊断、或上游 API 短期不可用）：**必须先经用户同意**——在 **设置页显式开关** 或 **提交前确认对话框**（默认关闭 / 默认不降级），并在 **AGENTS.md 或 PR 说明** 写清：开关含义、降级行为、风险；任务开始时 **日志中记录** 已启用降级及具体路径。
- **禁止**：为「省事」在代码里写未文档化、无开关、无用户提示的回退分支。

- **数据库 schema**：`assets` / `tasks` 表以当前 `CREATE TABLE` 为准，**不做**运行时 `ALTER` 迁移；本地 schema 异常时删除 `db/studio.db` 与 `outputs/`（或至少 `outputs/assets/`）后重启即可重建。
- **macOS + Apple Silicon only** — MLX 加速依赖 Metal
- **First generation is slow** — 模型加载到内存
- **No build step** — 前端纯 CDN，无需打包
- **Add model = add config** — 新模型只需修改 JSON + 引擎类
- **EP container 子节点约束**：`<el-header>` 和 `<el-main>` 必须是 `<el-container>` 的**直接子节点**（Element Plus flex 布局据此检测），不可包进 Vue 组件。`TopNav` 组件只渲染导航**内容**（单根 `<div>`）；`<el-header>` 外壳留在 `index.html` 模板中。`TaskDrawer` 可独立封装，因 `el-drawer` 走 Teleport 到 body。
- **跨模块 ref 解包不可靠**：`app.js` 中 `activePage` 必须用**本地 `ref`**，通过 `watch` 与 `DQRouter.currentPage` 双向同步，不可直接赋值为路由模块的 ref。

## Reference
- `README.md` — 用户文档
- `config/models_registry.json` — 模型配置
- `backend/core/interfaces.py` — 接口定义
