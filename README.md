# DanQing Studio v4.0 (丹青工作室)

现代化的 Apple MLX / CUDA 双后端插件化图像/视频生成引擎。前后端分离架构，支持 REST API、CLI、依赖注入、国际化和暗色主题。

## 特性

- **高性能后端**: Python + FastAPI + Apple MLX / CUDA 加速
- **现代化前端**: Vue 3 + Element Plus + 暗色主题
- **分层架构**: REST API / CLI → TaskScheduler → Engine → Pipeline → RuntimeContext → 持久化
- **面向接口编程**: 依赖注入容器，各层之间只依赖接口定义
- **模型即插件**: 新增模型仅需注册表配置 + Transformer 实现，零改动核心代码
- **国际化支持**: 中文/英文一键切换
- **双运行时**: MLX (Apple Silicon) / CUDA (NVIDIA) 自动适配
- **四大功能模块**:
  - **创作**: 文生图、图生图、视频生成、任务队列、提示词预设（按 **`applies_to`** 与当前动作过滤）
  - **图库**: 基于 SQLite **`assets`** 的浏览、预览、删除、再创作引用
  - **模型**: 注册表浏览、权重安装/删除、下载进度
  - **设置**: 系统配置、预设编辑、队列策略（如 **图像优先出队**）、环境状态

### 前端功能 ↔ 模型能力映射

创作页每个 Tab 需要模型注册表中对应的 `actions` 能力。模型在下拉列表中只显示与当前 Tab 兼容的项。

#### 图像创作页（image_create）

| 前端 Tab | 中文 | 所需模型 Action | 内部状态 | API 端点 |
|---|---|---|---|---|
| 文生图 | 文生图 | `create` | `editMode: image_generation` | `POST /api/images/generations` |
| 参考原图 | 参考原图改图 | `rewrite` | `editMode: image_editing` / `text_editing` / `rewriteDriveMode: reference` | `POST /api/images/edits` (operation: `rewrite`) |
| 按描述改图 | 按文字描述改图 | `rewrite` | `editMode: image_editing` / `text_editing` / `rewriteDriveMode: instruct` | `POST /api/images/edits` (operation: `rewrite`) |
| 局部修饰 | 局部重绘/修复 | `retouch` | `editMode: image_editing` / `inpainting` | `POST /api/images/edits` (operation: `retouch`) |
| 扩展画布 | 外扩/补全 | `extend` | `editMode: image_editing` / `outpainting` | `POST /api/images/edits` (operation: `extend`) |
| 精修放大 | 超分放大 | `upscale` | `editMode: image_upscale` | `POST /api/images/upscales` |

> 参考原图、按描述改图、局部修饰、扩展画布共用 `POST /api/images/edits`，通过 `operation` 字段（`rewrite` / `retouch` / `extend`）区分。

#### 视频创作页（video_create）

| 前端 Tab | 中文 | 所需模型 Action | 内部状态 | API 端点 |
|---|---|---|---|---|
| 文生视频 | 文生视频 | `create` | `videoWorkMode: create` | `POST /api/videos/generations` |
| 图生视频 | 图生视频 | `animate` | `videoWorkMode: animate` | `POST /api/videos/edits` |

完整注册表见 `config/models_registry.json`。

图像基准测试（**模型 × `actions` × mflux CLI** 对照表、PSNR 用例说明、跑法）见 [`tests/benchmark/README.md`](tests/benchmark/README.md)。

## 项目结构

```
DanQing-Studio/
├── backend/                    # 后端
│   ├── api/routes/            # REST API 路由（按媒体类型拆分）
│   ├── cli/                   # CLI 命令（与 REST API 一一对应）
│   ├── core/                  # 核心接口定义 + 契约 + 依赖注入容器
│   ├── engine/                # 引擎 + Pipeline + 运行时 + 模型
│   ├── persistence/           # SQLite 持久化层 (WAL 模式)
│   ├── scheduler/             # TaskScheduler（全局单 Worker 串行队列）
│   ├── services/              # 业务服务层（设置、下载等）
│   ├── utils/                 # 工具函数
│   └── main.py                # FastAPI 入口
├── frontend/                   # 前端 (Vue 3 + TypeScript + Vite + Element Plus + Pinia + Vue Router)
│   ├── src/                   # 应用源码（views、components、stores、utils、locales）
│   ├── index.html             # Vite HTML 入口
│   ├── vite.config.ts
│   ├── package.json
│   └── dist/                  # `npm run build` 产物（部署用）
├── bin/                        # 脚本目录
│   ├── launch.sh              # 启动脚本
│   └── stop.sh                # 停止脚本
├── scripts/                    # 桌面打包 + 代码门禁（check_consistency / check_engine_backend_imports）
├── config/                     # 配置文件目录
│   ├── models_registry.json   # 模型注册表（v2，schema_version: 2）
│   ├── .app_config.json       # 应用配置
│   ├── presets.json           # 提示词预设
│   └── locales/               # 后端翻译文件（zh.json / en.json）
├── db/                         # 数据库目录
│   └── studio.db              # SQLite（任务 + 资产，v3，WAL 模式）
├── models/                     # 模型目录
├── outputs/                    # 输出目录
├── tests/                      # 测试（含 benchmark 独立 venv；说明见 tests/benchmark/README.md）
└── requirements.txt            # Python 依赖
```

## 快速开始

### 环境要求

- macOS (Apple Silicon 推荐) 或 Linux + NVIDIA GPU
- Python 3.11+
- 至少 32GB 内存 (推荐)

### 安装

```bash
# 1. 克隆或下载项目
cd DanQing-Studio

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

> Benchmark tests use an independent virtual environment. To run benchmarks:
> ```bash
> make bench-setup  # Initialize tests/benchmark/venv/
> make bench-mflux / make bench-sanity
> python -m tests.benchmark mflux --all
> python -m tests.benchmark sanity --all
> ```

### 启动

```bash
# 使用启动脚本（自动检查依赖）
./bin/launch.sh

# 或手动启动
source .venv/bin/activate
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 7860
```

启动后访问: http://localhost:7860

### CLI 使用

```bash
# 图像生成
bin/danqing-generate --model flux1-schnell --prompt "a cat"

# 图像编辑
bin/danqing-edit --model flux-fill-controlnet --image input.png --mask mask.png --prompt "add a hat"

# 视频生成
bin/danqing-video-generate --model ltx-2.3-distilled --prompt "ocean waves"
```

完整 CLI ↔ REST API 映射见 `AGENTS.md`。

### 模型准备

将模型权重放入 `models/` 目录，LoRA 放入 `models/Lora/` 目录。

支持的模型格式:
- `.safetensors`
- `.bin`
- `.pt`
- `model_index.json`

## 后端架构

### 分层设计

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

### 依赖注入

所有组件通过 `Container` 注册和解析，例如资产存储:

```python
from backend.core.container import get_container
from backend.persistence.asset_store import SQLiteAssetStore

assets = get_container().resolve(SQLiteAssetStore)
```

### 接口定义

业务与持久化接口在 `backend/core/interfaces.py`；媒体生成能力在 `backend/core/media_interfaces.py`（`IImageEngine` / `IVideoEngine`）。调度器: `backend.scheduler.task_scheduler.TaskScheduler`。

### 模型插件化

新增模型按 **AGENTS.md**「新模型接入流程」与 **`.cursor/rules/model-migration.mdc`** 执行；概要如下（**Pipeline 装配骨架不增 `family` 分支**）：

1. **注册表声明** — `config/models_registry.json` 添加条目（含 `vae_scale`, `scheduler`, `text_encoder_out_layers` 等参数）
2. **配置数据类** — `backend/engine/config/model_configs.py` 添加 Config，并加入 `FAMILY_CONFIG_MAP`
3. **Transformer 实现** — `backend/engine/<family>/transformer.py`（继承 `TransformerBase`，注入 `RuntimeContext`）
4. **权重映射** — `backend/engine/<family>/weights.py` 等，提供 `remap_*_weights`
5. **注册表接线** — [`backend/engine/_transformer_registry.py`](backend/engine/_transformer_registry.py) 中 `_TRANSFORMER` / `_WEIGHT_REMAP` /（若需）`_TEXT_ENCODER` 条目

> **核心不变量**：新增模型优先只动注册表、Config、family 目录与 `_transformer_registry`；**不复制** API / CLI / Engine / Scheduler 业务路径。详见 [AGENTS.md](AGENTS.md)。

## API 文档

启动后访问自动生成的 API 文档:

- Swagger UI: http://localhost:7860/docs
- ReDoc: http://localhost:7860/redoc

## 开发

### 后端开发

```bash
# 热重载模式
python3 -m uvicorn backend.main:app --reload --port 7860
```

### 前端开发

前端使用 Vue 3 CDN 方式，无需构建步骤。直接编辑 `frontend/` 下的文件即可。

### 一致性检查

```bash
python3 scripts/check_consistency.py
```

预设文件格式转换：

```bash
make check-consistency
```

## 配置

配置文件保存在 `config/.app_config.json`:

```json
{
  "language": "zh",
  "theme": "dark",
  "default_model": "flux2-9b-distilled",
  "mlx_memory_limit": 120,
  "queue_image_first": true
}
```

## 环境变量

```bash
# .env 文件
HF_ENDPOINT=https://hf-mirror.com
HF_HUB_ENABLE_HF_TRANSFER=1
MLX_METAL_DEVICE_ONLY=1
MLX_METAL_MEMORY_LIMIT=120
```

## 许可证

MIT
