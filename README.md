# DanQing Studio v3.0 (丹青工作室)

现代化的 Apple MLX FLUX 图像生成工作站。前后端分离架构，支持 REST API、依赖注入、国际化和暗色主题。

## 特性

- **高性能后端**: Python + FastAPI + Apple MLX 加速
- **现代化前端**: Vue 3 + Element Plus + 暗色主题
- **分层架构**: REST API / Handler / 引擎 / 持久化 / 工具 / 依赖
- **面向接口编程**: 依赖注入容器，各层之间只依赖接口定义
- **国际化支持**: 中文/英文一键切换
- **四大功能模块**:
  - **创作**: 文生图、图生图、视频生成、任务队列、提示词预设（按 **`applies_to`** 与当前动作过滤）
  - **图库**: 基于 SQLite **`assets`** 的浏览、预览、删除、再创作引用
  - **模型**: 注册表浏览、权重安装/删除、下载进度（原「下载」能力并入此页）
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

#### 各 Action 支持的模型

| Action | 媒体 | 模型 |
|---|---|---|
| `create` | image | flux1-schnell, flux1-dev, flux1-krea-dev, flux2-klein-4b, flux2-klein-9b, flux2-klein-base-4b, flux2-klein-base-9b, z-image, z-image-turbo, fibo, fibo-lite, qwen-image, flux1-kontext |
| `rewrite` | image | 以上全部 + fibo-edit, fibo-edit-rmbg |
| `retouch` | image | flux-fill-controlnet（ControlNet，需蒙版） |
| `extend` | image | flux-fill-controlnet（ControlNet，需蒙版） |
| `upscale` | image | seedvr2-3b, seedvr2-7b |
| `create` | video | ltx-2.3-distilled, ltx-2.3-dev, wan-2.2-t2v-14b, wan-2.2-ti2v-5b |
| `animate` | video | ltx-2.3-distilled, ltx-2.3-dev, wan-2.2-i2v-14b, wan-2.2-ti2v-5b |

完整注册表见 `config/models_registry.json`。

## 项目结构

```
mflux-studio/
├── backend/                    # 后端
│   ├── api/routes/            # REST API 路由
│   ├── core/                  # 核心接口定义 + 依赖注入容器
│   ├── engine/                # MLX FLUX 图像生成引擎
│   ├── persistence/           # 持久化层 (JSON存储)
│   ├── services/              # 业务服务层
│   ├── utils/                 # 工具函数
│   └── main.py                # FastAPI 入口
├── frontend/                   # 前端 (Vue3 + CDN)
│   ├── css/theme.css          # 暗色主题样式
│   ├── js/                    # JavaScript 模块
│   │   ├── app.js             # Vue 应用入口
│   │   ├── i18n.js            # 国际化配置
│   │   ├── api.js             # API 客户端
│   │   ├── stores/            # RegistryStore、TasksStore（队列轮询 + 任务 SSE）
│   │   ├── composables/       # 如 media_queue（创作页队列过滤）
│   │   └── components/        # 页面组件
│   └── index.html             # HTML 入口
├── bin/                        # 脚本目录
│   ├── launch.sh              # 启动脚本
│   └── stop.sh                # 停止脚本
├── scripts/                    # 运维与 CI（如 check_consistency、注册表/预设迁移）
├── config/                     # 配置文件目录
│   ├── models_registry.json   # 模型注册表
│   ├── .app_config.json       # 应用配置
│   └── presets.json           # 提示词预设
├── db/                         # 数据库目录
│   └── studio.db              # SQLite（任务 + 资产，v3）
├── models/                     # 模型目录
├── loras/                      # LoRA 目录
├── outputs/                    # 输出目录
└── requirements.txt            # Python 依赖
```

## 快速开始

### 环境要求

- macOS (Apple Silicon 推荐)
- Python 3.11
- 至少 32GB 内存 (推荐)

### 安装

```bash
# 1. 克隆或下载项目
cd mflux-studio

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装 MFLUX 核心 (首次使用)
# 确保 .venv 已激活
pip install mflux mlx mlx-lm huggingface-hub safetensors tqdm requests
```

### 启动

```bash
# 使用启动脚本
./bin/launch.sh

# 或手动启动
source .venv/bin/activate
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 7860
```

启动后访问: http://localhost:7860

### 模型准备

将 FLUX 模型放入 `models/` 目录，LoRA 放入 `loras/` 目录。

支持的模型格式:
- `.safetensors`
- `.bin`
- `.pt`
- `model_index.json`

## 后端架构

### 分层设计

```
API Layer (media routes + gallery/settings/download)
    ↓ (依赖注入)
TaskScheduler + EngineRegistry + V3TaskStore + AssetStore
    ↓
`MFluxImageEngine` / `MlxVideoEngine`（`backend/engine/image/` · `backend/engine/video/`）
    ↓
`MFluxGenerationBackend` / `MLXVideoGenerationBackend` + `mlx_runtime.py`
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

落地路线与阶段验收见 **`docs/PLAN_COMPLETION_ROADMAP.md`**；Plan §7.5 与 v3 SQLite 列对照见 **`docs/PLAN_7_5_SCHEMA_REVIEW.md`**。

一致性检查（注册表 v2、路由、`presets.json` 的 **`applies_to`**、关键前端契约等）：

```bash
python3 scripts/check_consistency.py
```

若本地仍有带 legacy **`mode`** 的 `config/presets.json`，可先备份再执行：

```bash
python3 scripts/migrate_presets_mode_to_applies.py --write
```

## 配置

配置文件保存在 `config/.app_config.json`:

```json
{
  "language": "zh",
  "theme": "dark",
  "default_model": "flux2-9b-distilled",
  "mlx_memory_limit": 120
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

## 从 v1.x 迁移

v1.x 基于 tkinter 的单文件 GUI 已移除。
v2.0 是完全重构的版本，使用全新的前后端分离架构。

## 许可证

MIT
