# AGENTS.md - DanQing Studio v4 (丹青工作室)

## What this is
丹青工作室 — MLX/CUDA 双后端插件化图像/视频生成引擎。前后端分离。

## Entry points
- `backend/main.py` — FastAPI 后端入口
- `frontend/index.html` — Vue 3 前端入口
- `bin/launch.sh` — 启动脚本
- `bin/stop.sh` — 停止脚本
- `requirements.txt` — Python dependencies
- `Makefile` — 基准测试 / lint

## Environment & dependencies
- **Python**: 3.11+
- **Venv**: `.venv/` in repo root
- **Key packages**: `fastapi`, `uvicorn`, `mlx`, `Pillow`, `transformers`, `safetensors`

> Benchmark test dependencies are installed in an independent venv at `tests/benchmark/venv/`. Run `make bench-setup` to initialize that environment.

## Architecture (v4.0)

### 设计原则

#### 核心架构目标（治理锚点）

面向贡献者与自动生成代码，下列三条为**产品级约束**（实现与评审时对照自检；与 `.cursor/rules/no-silent-degrade.mdc`、`.cursor/rules/model-migration.mdc` 一致）。

1. **架构优化**：维持下文「分层 / 模块化 / 组件化」边界；**代码复用**优先抽公共能力到 `backend/engine/common/`（及与 Pipeline 正交的小模块），避免上游整树拷贝或平行 `runtime/` 分层。**新模型插件化**：`config/models_registry.json` + `backend/engine/config/model_configs.py` + `backend/engine/families/<family>/` + `backend/engine/_transformer_registry.py` 四者联动；Pipeline 骨架以注册表与多态驱动，**禁止**为省事增加 `family == …` 业务分支。

2. **契约化 REST API 与 CLI**：入口层只通过 [`backend/core/contracts.py`](backend/core/contracts.py) 的请求/响应模型与 [`backend/core/media_interfaces.py`](backend/core/media_interfaces.py) 的 `IImageEngine` / `IVideoEngine` 进入引擎，**与具体模型解耦**。模型差异由注册表（`actions`、`parameters`、`versions`）与 Transformer / Hook 多态吸收，**不在** `backend/api/routes/` 或 `backend/cli/` 内堆「按 model_id 分支」的重复业务逻辑。当前对外形态为**按媒体与动作的资源型 API**（`/api/images/*`、`/api/videos/*` 等）。若未来增加 **OpenAI 兼容** 的图像/视频/音频 HTTP 面，应为**单独路由前缀与适配 DTO**（例如 `/v1/...`），**内部仍转调**同一 contracts、Engine 与 Scheduler；**禁止**用单一 `/api/completions` 之类端点替代按媒体的资源型 API，也禁止复制第二套生成业务逻辑。

3. **RuntimeContext 之上双平台自适应**：[`backend/engine/runtime/_base.py`](backend/engine/runtime/_base.py) 为张量与模块工厂的**窄**后端抽象（见文件头说明）；[`MLXContext`](backend/engine/runtime/mlx.py) / [`CudaContext`](backend/engine/runtime/cuda.py) 为唯二实现。目标状态：`ImagePipeline` / `VideoPipeline`、各 `family` 的 Transformer 与共享组件在**热路径**上优先使用传入的 `RuntimeContext`（`ctx.Linear`、`ctx.matmul`、`ctx.eval` 等），**避免**在 `backend/engine/families/<family>/` 与 `backend/engine/common/` 中直接 `import mlx` / `import torch`。**平台差异**按 Go 式三文件组 `xxx.py` / `xxx_mlx.py` / `xxx_cuda.py`（形态 A：公共 ctx + 小钩子；形态 B：差异极大时 `xxx.py` 仅对外接口 + dispatch，整段实现放在 `*_mlx`/`*_cuda`）— 详见 [`docs/dual_platform_architecture.md`](docs/dual_platform_architecture.md) §8.5。**族内「文件数」预算**：`families/<family>/` 默认按 **折算逻辑单位 ≤8**；同一逻辑组件、同名 `stem` 的 `stem.py` + `stem_mlx.py` + `stem_cuda.py`（可只存在其中一部分文件）**合计计 1 个单位**，不按物理 `.py` 个数拆成 3。CI：`make check-engine-imports`。允许例外：仅 `backend/engine/runtime/` 内绑定后端；或经注册表声明**仅单后端**的隔离模块，且缺能力时须显式失败，禁止静默降级。引擎按注册表 `backends` 选择 runtime（见 `DanQingImageEngine` / `DanQingVideoEngine` 的 `_resolve_runtime`）。**存量**若仍存在后端泄漏，新改动**不得扩大**泄漏面；应逐步收敛。

**验收判据（摘要）**

| 维度 | 判据 |
|------|------|
| 插件化 | 新图像 family 原则上只改 JSON、`model_configs`、`_transformer_registry` 与 `backend/engine/families/<family>/`；Pipeline 不新增 `family ==` 式业务分支 |
| 族目录体量 | `families/<family>/` 默认 **折算逻辑单位 ≤8**（同名 `stem` 的 `stem.py`+`stem_mlx.py`+`stem_cuda.py` 一组计 **1**；可缺省某一后缀） |
| API/CLI | 新能力先扩展 contracts + 路由/CLI，再落 Engine；REST 与 CLI 语义对齐 |
| 双平台 | 注册表声明多 `backends` 的模型应以同一请求体在两 runtime 可执行；无 CUDA/MLX 实现时 **fail loud**，不静默换路径 |

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
- `backend/engine/` — 引擎 + [`pipelines/`](backend/engine/pipelines/)（装配线）+ [`families/`](backend/engine/families/)（各模型族）+ `runtime/` + `common/` + `config/`
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
新增模型 = 注册表 JSON + Config 数据类 + Transformer 类 + 权重 Remap + `_transformer_registry` 声明。**所有接入 `ImagePipeline` / `VideoPipeline` 的 DiT Transformer 必须继承 `TransformerBase`**（统一 `forward`/`__call__`、`load_weights`/`parameters`、生命周期 Hook、`refine_cfg_noise` 默认实现）；仅在确有特殊加载逻辑时覆盖 `load_weights`（并 `super()` 委托）。

- 注册表声明：`config/models_registry.json`（family / engine / actions / parameters / versions）
- 配置数据类：`backend/engine/config/model_configs.py`（含 `vae_scale`, `encoder_type`, `text_encoder_out_layers` 等）
- Transformer 实现：`backend/engine/families/<family>/transformer.py`（及各模型目录下组件；双后端按 `xxx.py` / `xxx_mlx.py` / `xxx_cuda.py` 拆分，见 `docs/dual_platform_architecture.md` §8.5）
- 权重映射：`backend/engine/families/<family>/weights.py` 等（diffusers → DanQing 键名转换）
- Pipeline 注册：[`backend/engine/_transformer_registry.py`](backend/engine/_transformer_registry.py) 中 `family` → 类 / remap / text encoder（**零 Pipeline 内 if-elif family 表**）
- **族目录体量**：`families/<family>/` 默认 **折算逻辑单位 ≤8**；同一 `stem` 的 Go 式 `stem.py` / `stem_mlx.py` / `stem_cuda.py` **合计计 1**（可缺省某一后缀），见上文「核心架构目标」与 [`docs/dual_platform_architecture.md`](docs/dual_platform_architecture.md) §8.5。
`ImagePipeline.run()` 是装配线，**由注册表 `parameters` + Config 数据类驱动**；去噪循环统一调用 `model(latents, t, ...)`，**避免在 Pipeline 内按 `family` 分支实现模型特有数学**（应下沉到 Transformer 多态方法）。
1. 解析 model → 查注册表 → 获取 family / config
2. 按 `config.encoder_type` 选择文本编码器（构造参数由 `config` 统一传入；各 Encoder 忽略无关 kwargs）
3. 按注册表 `parameters.scheduler.default` 选择调度器
4. 按 `config.vae_scale` 创建初始 latent
5. 跑通用去噪循环（全模型统一接口 `model(latents, t, txt_embeds=..., sigmas=...)`）；CFG 后处理见上 `refine_cfg_noise`
6. VAE 解码（通过权重键名 / VAE config 标志触发特殊预处理，非 `family` 硬编码）
7. 资产落盘（`asset_store.create_from_file`）

**核心不变量**：新增模型优先只改注册表、Config、Transformer、`_transformer_registry`；**尽量不修改** Pipeline 骨架。若必须改 Pipeline，应仅为新的 **注册表驱动开关** 或 **通用多态调用点**，不增加 `family ==` 分支。

#### 模型生命周期 Hook（拦截器模式）

`TransformerBase` 定义 Hook 接口（默认空实现），Pipeline 在关键节点调用。模型选择性覆盖以实现 LoRA/ControlNet 等扩展：

```
Pipeline.run()
│
├─ _load_model() → model.load_weights()
├─ Hook ①: after_load_weights(bundle_root)          ← LoRA/Adapter 权重合并
│
├─ 文本编码 (txt_embeds)
├─ Hook ②: prepare_conditioning(request, bundle)     ← ControlNet 编码控制图，返回 cond dict
│
├─ 调度器 (timesteps, sigmas)
├─ Hook ③: before_denoise(latents, timesteps, sigmas, **cond)  ← 注入 ControlNet 信号，修改 latents
│
├─ for step in timesteps:
│     noise_pred = model(latents, t, ...)
│     latents = scheduler.step(noise_pred, t, latents)
│     Hook ④: step_callback(step_idx, latents, noise_pred)  ← 动态条件/日志
│
└─ VAE 解码
```

| Hook | 调用时机 | LoRA 用法 | ControlNet 用法 |
|------|----------|-----------|-----------------|
| ① `after_load_weights(bundle_root)` | load_weights 后 | 加载并合并 LoRA 权重到模型 | — |
| ② `prepare_conditioning(request, bundle)` | 文本编码后 | — | 加载 ControlNet 模型，编码控制图 → cond dict |
| ③ `before_denoise(latents, timesteps, sigmas, **cond)` | 去噪循环前 | — | 注入 ControlNet 信号到 latents，返回修改后的 (latents, cond) |
| ④ `step_callback(step_idx, latents, noise_pred)` | 每步去噪后 | — | 动态调整注入强度 |

**CFG 后处理（多态，非 Hook）**：`refine_cfg_noise(noise_cond, noise_pred, *, cfg_renorm_min)` 定义于 `backend/engine/common/_base.py` → `TransformerBase`（默认恒等）。当注册表 `enable_cfg_renorm` 为 true 且 CFG 生效时，Pipeline 在标准 CFG 合并后调用；需要与参考实现一致时再在对应 Transformer 中覆盖。未继承 `TransformerBase` 的模型可提供同名方法或由 Pipeline `getattr` 回退为恒等。

**Hook 基类定义**：`backend/engine/common/_base.py` → `TransformerBase`。
新增模型覆盖需要的 Hook / `refine_cfg_noise` 即可；Pipeline 内避免新增 `family ==` 分支。

#### Pipeline 个性化规则（配置 / 多态 / Hook）

- **注册表 + Config**：标量/枚举/结构化默认（步数、调度器、`vae_scale`、`supports_guidance`、`enable_cfg_renorm` 等）只放在 `config/models_registry.json` 与 `model_configs.py`，由 Pipeline 注入；**不在 Pipeline 写 `if family == ...` 调参数**。
- **Transformer 多态**：与张量形状/算子强相关的差异（如 CFG renorm、timestep 语义、`forward` 内部约定）放在模型类的 **`forward` / `refine_cfg_noise` 等实例方法**，默认实现保持恒等或可忽略。
- **Hook**：仅用于 **横切、可选** 能力（LoRA 合并、ControlNet 条件、`before_denoise` 改 latent、每步日志）；不把核心扩散方程的主要分支塞进 Hook，以免调用顺序难推理。
- **能力声明**：模型未实现的入口 **不得** 在注册表 `actions` 中登记对应能力（例如未接线的文生图不要写 `create`），由 `IImageEngine.supports` + API/CLI/Engine 在入参阶段拒绝；Pipeline 保留 **防御性** 断言仅作最后防线，不依赖 `family` 字符串列表维护「谁能文生图」。
- **数据驱动检测**：同一 family 下权重变体（如 VAE flux2 风格预处理）优先用 **权重键 / config 标志** 触发，而非 `family` 硬编码。

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
- `backend/engine/` — 引擎 + [`pipelines/`](backend/engine/pipelines/)（装配线）+ [`families/`](backend/engine/families/)（各模型族）+ `runtime/` + `common/` + `config/` + 调度器 + VAE
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
`ModelRegistry`（`backend/core/model_registry.py`）加载 `models_registry.json`；`EngineRegistry`（`backend/engine/engine_registry.py`）按 `engine` 字段绑定 `DanQingImageEngine` / `DanQingVideoEngine`。图像 / 视频 Transformer 实现位于 `backend/engine/families/<family>/`（如 `families/flux1/`、`families/ltx/`），由 [`_transformer_registry.py`](backend/engine/_transformer_registry.py) 按 `family` 解析。

### 模型迁移与实现原则（强制）
新增或迁移接入 Pipeline 的模型时，遵守 Cursor 规则 **`.cursor/rules/model-migration.mdc`**：优先 **Flux1 / Flux2 / Z-Image** 式「少文件 + `remap_*` 注册表驱动」；禁止上游目录整棵拷贝；权重能短则短，**VAE 类长映射表合理可接受**；MLX 嵌套 `parameters()` 须显式展平 `_param_map` 与 remap 对齐。

### 新模型接入流程（5 步概要）

1. **注册表** — `config/models_registry.json`：`family`、`engine`、`actions`、`versions.local_path`
2. **Config** — `backend/engine/config/model_configs.py`：为该 `family` 增加 dataclass，并注册到 `FAMILY_CONFIG_MAP`
3. **Transformer** — `backend/engine/families/<family>/transformer.py`：继承 `TransformerBase`，构造注入 `RuntimeContext`
4. **权重映射（按需）** — `backend/engine/families/<family>/weights.py`：`remap_*` 将 checkpoint 键对齐 `_param_map`
5. **接线** — [`backend/engine/_transformer_registry.py`](backend/engine/_transformer_registry.py)：`/_TRANSFORMER`、`/_WEIGHT_REMAP`、（按需）`/_TEXT_ENCODER`

差异化逻辑放在 Transformer / remap / Text Encoder，**不在** `ImagePipeline` 写 `if family == ...` 硬分支。

**验证：** `bin/danqing-generate --model <id> --prompt "..."`；与参考 CLI 像素对比见 `tests/benchmark/`。

无参考 CLI 的模型：跑 **`make bench-sanity`** 或 `python -m tests.benchmark.run --sanity`，做成片健全性（拒纯白/纯黑/近乎单色平场）。

当前基准测试结果（2026-05）：
| 模型 | Action | PSNR | 状态 |
|------|--------|------|------|
| flux2-klein-9b | create | 31.9 dB | ✅ PASS |
| z-image | create | 28.6 dB | ⚠️ WARN |
| z-image-turbo | create | 16.7 dB | ❌ FAIL |

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

### 引擎 / 双平台维护
- **双平台 import 守门**：`make check-engine-imports`（[`scripts/check_engine_backend_imports.py`](scripts/check_engine_backend_imports.py)）— `mlx`/`torch` 仅允许 `backend/engine/runtime/` 与 `*_mlx.py` / `*_cuda.py`，或经 **`importlib.import_module`** 等动态加载（字面 `import mlx` / `import torch` 不得出现在其它路径）。[`scripts/engine_backend_import_allowlist.txt`](scripts/engine_backend_import_allowlist.txt) 默认 **空**（仅注释占位）；若临时豁免须评审且 **只缩不增**。设计说明见 [`docs/dual_platform_architecture.md`](docs/dual_platform_architecture.md) §8.5。

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
- `POST /api/audios/generations`、`POST /api/audios/edits` — 接受任务并入队；**无音频推理后端**，执行时在任务日志中 **显式失败**；前端 **`api.audios.*`**（`api.js`）与路径对齐（`dubs` 等未实现路由仍会 404）

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

- **核心架构目标（分层 / 契约化 API·CLI / RuntimeContext 双平台）**：完整条文与验收判据见上文 **设计原则 → 核心架构目标（治理锚点）**；评审与自动生成代码时以此为据，避免与 Pipeline、注册表、Runtime 边界不一致的实现。

- **数据库 schema**：`assets` / `tasks` 表以当前 `CREATE TABLE` 为准，**不做**运行时 `ALTER` 迁移；本地 schema 异常时删除 `db/studio.db` 与 `outputs/`（或至少 `outputs/assets/`）后重启即可重建。
- **macOS + Apple Silicon only** — MLX 加速依赖 Metal
- **First generation is slow** — 模型加载到内存
- **No build step** — 前端纯 CDN，无需打包
- **Add model** — 见「核心架构目标」插件化判据与下文「新模型接入流程」（注册表 + Config + Transformer + remap + `_transformer_registry`；非仅改 JSON 一处）
- **EP container 子节点约束**：`<el-header>` 和 `<el-main>` 必须是 `<el-container>` 的**直接子节点**（Element Plus flex 布局据此检测），不可包进 Vue 组件。`TopNav` 组件只渲染导航**内容**（单根 `<div>`）；`<el-header>` 外壳留在 `index.html` 模板中。`TaskDrawer` 可独立封装，因 `el-drawer` 走 Teleport 到 body。
- **跨模块 ref 解包不可靠**：`app.js` 中 `activePage` 必须用**本地 `ref`**，通过 `watch` 与 `DQRouter.currentPage` 双向同步，不可直接赋值为路由模块的 ref。

## Reference
- `README.md` — 用户文档
- `config/models_registry.json` — 模型配置
- `backend/core/interfaces.py` — 接口定义
