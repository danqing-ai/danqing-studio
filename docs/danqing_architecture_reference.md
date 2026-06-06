# DanQing Studio 架构参考方案

> 目标：在保留 DanQing 产品注册表、下载中心和 fail-loud 治理的前提下，借鉴 ComfyUI 的低摩擦资产发现与组件复用思路，将新模型接入从“多处重复登记 + 大块搬运”收敛为“产品清单 + family contract + 少量工程注册”。

**相关文档**

- [engine_refactor_plan.md](engine_refactor_plan.md) — 分阶段重构计划与 PR 切片（fail-loud 门禁）
- [engine_new_model_checklist.md](engine_new_model_checklist.md) — 新模型接入即时 checklist
- [AGENTS.md](../AGENTS.md) — 治理锚点（权威实现规范）

**文档定位**：本文是**架构参考、认可条目全集与演进方案**（设计 + 治理 + 实施 + 验收均须能在本文找到对应条目）。  
即时落地规范仍以 `AGENTS.md` + checklist + `make verify-engine-stack` 为准。  
[engine_refactor_plan.md](engine_refactor_plan.md) 为本文实施章节的**执行摘要**（Phase / PR / Sprint 索引），不得包含本文未认可的额外方向。

---

## 目录

0. [深度复核结论](#0-深度复核结论)
1. [当前架构全景](#1-当前架构全景)
2. [核心问题诊断](#2-核心问题诊断)
3. [目标架构设计](#3-目标架构设计)
4. [复用层详细设计](#4-复用层详细设计)
5. [模型族接入规范](#5-模型族接入规范)
6. [权重加载与发现机制](#6-权重加载与发现机制)
7. [迁移路线图](#7-迁移路线图)
8. [CI 门禁与治理](#8-ci-门禁与治理)
9. [与 ComfyUI 的对比借鉴](#9-与-comfyui-的对比借鉴)
10. [附录：代码示例](#10-附录代码示例)
11. [治理硬约束与质量门禁](#11-治理硬约束与质量门禁)
12. [分阶段实施计划](#12-分阶段实施计划)
13. [PR 切片（7 个独立 review）](#13-pr-切片7-个独立-review)
14. [第一 Sprint 与 12 周成功指标](#14-第一-sprint-与-12-周成功指标)

---

## 0. 深度复核结论

本文档应定位为**治理参考与演进方案**，不是替代 `AGENTS.md`、`.cursor/rules/model-migration.mdc`、`docs/engine_new_model_checklist.md` 的即时实现规范。复核后的关键修正如下：

1. **模型注册文件必须保留**：`default_config/models_registry.json` / `{workspace}/config/models_registry.json` 是下载中心、模型商店、动作参数、许可证与安装状态的数据源，不能被目录扫描替代。
2. **可以减少注册信息，但不能减少事实来源**：重复字段应通过 `model_profiles` / family defaults / version defaults 派生；产品事实仍在 JSON，工程事实仍在 `_transformer_registry.py`、`model_configs.py` 与 codec registry。
3. **ComfyUI 的亮点是资产低摩擦，不是无约束动态图**：可借鉴目录扫描、组件识别、模型卡片、LoRA patch、工作流图验证；不可照搬“运行时猜结构并容忍失败”的宽松策略。
4. **新模型降本的核心不是自动推断所有东西**：而是提供模板、scaffold、family contract、权重 key parity 检查、组件目录约定和 CI 报告，让错误尽早显式暴露。
5. **MLX/CUDA 双端支持不是无条件自动获得**：只有纯 `ctx.*` 热路径可自然双端；涉及 VAE、特殊 attention、视频/音频后处理时必须提供 `*_mlx.py` / `*_cuda.py` 或在注册 `backends` 时 fail loud。
6. **Engine 重构净 LOC**：`backend/engine/` 治理 PR 应净删或持平；禁止新增平行 registry 树（`vae_codecs/`、`video_codecs/` 等）。
7. **双平台诚实**：registry `backends` 声明什么就必须能跑什么；缺能力在任务入口 fail loud，不静默换路径。
8. **不默认降级、不静默出烂图**：缺组件、remap 不完整、shape 不对、LoRA merge 失败 → 显式失败；禁止 90% key 匹配继续加载、禁止 `except: pass` 吞错后继续生成。

### 0.2 北极星（认可的目标形态）

```
产品清单 (models_registry.json)
  + family contract (model_configs + codec registry)
  + 少量工程注册 (_transformer_registry)
  + 约定式 bundle 校验 (manifest)
  + 权重 key parity
  + scaffold / CI 报告
```

### 0.3 治理硬约束（不可妥协）

| 约束 | 含义 |
|------|------|
| **Fail loud** | 缺 backend、缺组件、remap 不完整、shape 不对 → `RuntimeError` / 清晰 API 错误 + task log |
| **不静默降级** | 禁止换弱 CLI、unknown model shim、90% key 匹配继续加载、未文档化 fallback |
| **不偷偷出烂图** | 禁止缺 VAE 占位、LoRA merge 失败仍生成、CUDA 不可用却跑 MLX |
| **注册表保留** | `default_config/models_registry.json` 是下载中心唯一产品数据源；只能瘦身，不能删 |
| **Engine 净 LOC** | `backend/engine/` 重构净删或持平；不新增平行 registry 树 |
| **双平台诚实** | `backends` 与实现一致；否则注册时或任务启动时 fail loud |

### 0.4 明确不做（认可边界）

- 不用目录扫描替代 `models_registry.json`
- 不做 ComfyUI 式开放节点画布（首期）
- 不做「运行时从权重反推结构」
- 不做未文档化、默认开启的 fallback toggle
- 不从 `families/` 反推下载卡片；不扫描 `models/` 自动加入可生成列表
- 不把 Pipeline 变成任意图执行器（GenerationGraph 仅内部可观测性）

### 0.5 主要债务与优先级（认可的问题清单）

| 问题 | 风险 | 优先级 |
|------|------|--------|
| Qwen / seedvr2 等 ctx 孤岛 | CUDA 阻断、重复 attention/norm | P0 |
| 文本编码器分散在 family 内 | 双端不一致、维护翻倍 | P0 |
| 安装成功 ≠ 可加载 | 下载完才在 generate 时报错 | P0 |
| remap 无 parity 门禁 | 静默丢 key → 烂图 | P0 |
| registry 重复字段多 | 新模型登记成本高 | P1 |
| Pipeline 潜在 family 分支 | 插件化倒退 | P1 |
| common/ 边界模糊（如 qwen_image VAE 长表） | 复用假象、治理难 | P1 |

### 0.1 二轮复核：当前态 vs 目标态（2026-05-29）

| 主题 | 当前仓库（已存在） | 本文目标态 / 规划中 |
|------|-------------------|---------------------|
| 产品注册表 | `default_config/models_registry.json` + workspace 同步 | `profiles` / `parameter_templates` 解析层（**已落地** `registry_profiles.py`） |
| 工程注册 | `_transformer_registry.py` + `model_configs.py` + codec registry | 约定式 `register_image_family()` helper（**未落地**） |
| bundle 校验 | `bundle.manifest.json` + `FAMILY_BUNDLE_CONTRACTS` + pipeline assert | 下载中心组件状态（**已落地**） |
| weight parity | `check_engine_governance --rule parity` | 家族 remap vs `_param_map` 报告（**已落地**） |
| common/text_encoders | T5 / CLIP / Qwen3 MLX 部分存在；视频 T5 已接入 | 完整 Go-style 三文件组 + 族内 encoder 收敛（**部分**） |
| Qwen DiT | `transformer.py` 双端 dispatch；MLX 内层 `_T` singleton | 全路径 injected `ctx`（**部分**） |
| CogVideoX VAE | `families/cogvideox/vae*.py` | ctx 化与 parity（**未动**） |
| seedvr2 | `stem.py` + `stem_mlx.py` + `dit`/`vae`/`preprocess`/`weights`（**4 逻辑单位**） | 结构重组 **已落地**；`*_mlx` 热路径保留 `mx.*`（治理允许） |
| CI 门禁 | `make check-engine-imports` 等 → `scripts/check_engine_governance.py` | `measure_reuse` / family budget 报告（**部分未落地**） |
| registry `family` vs 目录 | 例：`qwen_image`（registry）↔ `families/qwen/`（代码） | 保持显式映射，禁止靠目录名猜测 |

**二轮结论**：§1–§2 以**当前树**描述债务；§3+ 为**目标架构**；§11–§14 为**认可的全部实施与验收条目**（与 engine_refactor_plan 对齐）。

---

## 1. 当前架构全景

> 下图反映**当前仓库**主要结构（非目标态）。音频（ace_step）、视频（hunyuan 等）族未在树中逐文件展开。

```
DanQing-Studio/
├── backend/
│   ├── api/routes/              # REST 端点
│   ├── cli/                     # CLI 工具
│   ├── core/                    # 契约、接口、DI、i18n、registry 解析
│   ├── engine/
│   │   ├── runtime/             # MLXContext / CudaContext
│   │   │   ├── _base.py         # RuntimeContext ABC
│   │   │   ├── mlx.py
│   │   │   └── cuda.py
│   │   ├── common/              # 共享算子层
│   │   │   ├── _base.py         # TransformerBase
│   │   │   ├── attention.py, norm.py, embeddings.py, schedulers.py
│   │   │   ├── vae/             # AutoencoderKL + qwen_image/ 子树（长表 VAE）
│   │   │   ├── text_encoders/   # T5 / CLIP / Qwen3 MLX（部分已落地）
│   │   │   └── bundle_weights/  # PathResolution、loader_mlx、definitions
│   │   ├── families/            # 模型族实现
│   │   │   ├── flux1/, flux2/, z_image/, fibo/     # 图像 DiT
│   │   │   ├── qwen/            # registry family=qwen_image；MLX 仍 nn.Module 热路径
│   │   │   ├── seedvr2/         # Shape B upscale job；7 逻辑单位
│   │   │   ├── ltx/, wan/, cogvideox/, hunyuan/    # 视频
│   │   │   └── ace_step/                           # Shape C 音频
│   │   ├── pipelines/           # image / video / upscale / music
│   │   ├── _transformer_registry.py
│   │   ├── vae_codec_registry.py
│   │   └── danqing_*_engine.py
│   └── main.py
├── frontend/
└── default_config/
    └── models_registry.json      # 产品清单（下载中心数据源）
```

---

## 2. 核心问题诊断

### 2.1 复用孤岛：Qwen 族（registry `family=qwen_image`）

入口层已双端 dispatch，但 MLX 热路径仍绕过 common：

```python
# families/qwen/transformer.py — 已 DelegatingDiTStem，按 ctx 选 MLX/CUDA
class QwenImageTransformer(DelegatingDiTStem):
    ...

# families/qwen/transformer_mlx.py — 仍直接 mlx.nn，未复用 common/attention
class QwenAttention(nn.Module):
    def __init__(self, dim, heads):
        self.q_proj = nn.Linear(dim, dim)  # mlx.nn
```

**影响**：
- MLX 与 CUDA 各维护一套 DiT 数学，parity 成本高
- 无法从 common attention/norm 受益
- 维护成本翻倍

### 2.2 文本编码器分散

| 位置 | 实现方式 | 状态 |
|------|---------|------|
| `common/text_encoders/t5_mlx.py`, `clip_mlx.py`, `qwen3_mlx.py` | 跨族 MLX；视频 T5 已用 | **部分落地** |
| `z_image/text_encoder*.py` | MLX 自研 + CUDA HF Qwen3 | 族内 + common 混用 |
| `flux2/text_encoder*.py` | MLX 为主 | 族内 |
| `qwen/text_encoder*.py` | MLX + CUDA 双文件 | 族内 |
| `flux1/flux1_t5_mlx.py`, `flux1_clip_mlx.py` | Flux1 专属 | 族内 |

**目标**：真正跨族（≥2 family）→ `common/text_encoders/` Go-style 三文件组；单族专属留 family。

### 2.3 common/ 边界（已改善项 + 剩余债务）

**已改善**：CogVideoX 3D VAE 已在 `families/cogvideox/vae*.py`，不再堆在 `common/vae/cogvideox_decoder*`。

**剩余债务**：

```
common/vae/qwen_image/     # Qwen-Image 长表 VAE；表长合理，但应视为 codec 子树而非新平行包
common/vae/qwen_image/*_mlx.py
```

**原则**：长映射表可留 `common/vae/<codec>/` 或 `families/<family>/`；禁止再增无 codec 注册的平行 wrapper 目录。

### 2.4 文件预算：seedvr2（Shape B → Go-style stem）

当前（**4 逻辑单位**，符合治理目标）：

```
seedvr2/
├── stem.py             (1)  # Shape B 公共入口 + re-export
├── stem_mlx.py         (1)  # 与 stem 同 stem 计 1
├── dit_mlx.py          (2)
├── preprocess_mlx.py   (3)
├── vae_mlx.py          (4)
└── weights.py          (并入 weights 表；与 stem 组分离)
```

**已完成**：删除 `upscale.py` / `job_mlx.py`；超分经 `stem` / `stem_mlx`；DiT/VAE/preprocess 仍用 `mx.*`（`*_mlx.py` 允许直连 MLX，避免 `RuntimeContext` 未覆盖 API 回归）。

---

## 3. 目标架构设计

### 3.0 三层架构 + 四份事实（认可的目标分层）

```
产品层     models_registry.json  →  DownloadService  →  bundle.manifest.json
契约层     model_configs.py + _transformer_registry.py + vae_codec_registry.py
运行层     ImagePipeline / VideoPipeline  →  GenerationGraph（日志 / 调试，首期不开放编辑）
```

| 事实 | 来源 | 用途 |
|------|------|------|
| 产品事实 | `default_config/models_registry.json` | 下载中心、参数面板、版本、许可证、i18n |
| 结构事实 | `model_configs.py` + Transformer 代码 | hidden size、latent scale、encoder_type、required components |
| 接线事实 | `_transformer_registry.py` + codec registry | 类、remap、text encoder、LoRA merge |
| 资产事实 | 本地 bundle + `bundle.manifest.json` | 已安装组件、文件路径、完整性 |

### 3.1 核心原则

1. **RuntimeContext 收窄**：不全能化，允许 DiT 大块放在 `*_mlx.py` / `*_cuda.py`
2. **common/ 仅放真跨族组件**：族专属逻辑回收到 `families/<id>/`
3. **Go-style 三文件组**：`stem.py` + `stem_mlx.py` + `stem_cuda.py`，同 stem 计 1 单位
4. **注册表保留，信息瘦身**：产品清单仍在 `models_registry.json`，但通过 family profile、version defaults 和参数模板减少重复字段
5. **约定驱动安装校验**：目录约定 + 显式 manifest + 本地扫描只用于验证和补全安装状态，不用于静默猜测模型结构
6. **CI 强制边界**：`mlx`/`torch` 仅出现在 `runtime/` 和 `*_mlx`/`_cuda`

### 3.1b 注册表分层：保留数据源，减少重复登记

当前“注册表”实际承担两类职责，应该拆清楚语义，而不是删除 JSON：

| 层级 | 文件/模块 | 事实类型 | 可否省略 |
|------|-----------|----------|----------|
| 产品清单 | `default_config/models_registry.json` | 下载中心卡片、来源、版本、安装路径、动作参数、许可证、推荐状态、i18n | **不可省略** |
| 运行时清单 | `{workspace}/config/models_registry.json` | 用户工作区可见模型、安装状态、恢复默认后的运行数据 | **不可省略** |
| 架构配置 | `backend/engine/config/model_configs.py` | hidden size、层数、latent scale、encoder_type、family contract | 不可省略，但可由 bundle config 覆盖显式白名单字段 |
| 工程注册 | `backend/engine/_transformer_registry.py` / codec registry | family 到类、remap、text encoder、LoRA merge、VAE codec 的代码入口 | 可通过约定减少条目，但不能靠动态导入吞错 |

建议引入 `profiles` / `parameter_templates` 概念，减少每个模型条目的重复信息：

```json
{
  "profiles": {
    "image_flux_like": {
      "engine": "danqing-image",
      "media": "image",
      "category": "base_models",
      "parameters": "$templates.image_txt2img_1024"
    }
  },
  "models": {
    "flux2-klein-9b": {
      "profile": "image_flux_like",
      "family": "flux2",
      "name": { "zh": "FLUX.2 Klein 9B", "en": "FLUX.2 Klein 9B" },
      "versions": {
        "default": {
          "source": "huggingface",
          "repo_id": "black-forest-labs/FLUX.2-Klein-dev",
          "local_path": "models/flux2-klein-9b"
        }
      }
    }
  }
}
```

约束：

- `family`、`engine/media`、`actions`、`backends` 必须在解析后存在；缺失直接报错。
- `profile` 只减少重复，不允许改变模型能力语义；解析后的 registry 仍应可由 `/api/registry` 完整返回。
- 下载中心只读取产品清单与版本信息，不从 `families/<family>/` 反推展示内容。
- 工程注册可以增加约定式 helper，例如 `register_image_family("flux2")` 自动定位 `families/flux2/transformer.py` 和 `weights.py`，但导入失败必须 fail loud。

### 3.2 目标目录结构（目标态，非当前树）

> 下列为演进目标；当前各 family 仍含 `transformer_mlx.py`、`lora_mlx.py` 等增量文件，以 `make check-engine-family-layout` 为准。

```
backend/engine/
│
├── runtime/                          # UNCHANGED
│   ├── _base.py                        # RuntimeContext ABC
│   ├── mlx.py                          # MLXContext
│   └── cuda.py                         # CudaContext
│
├── common/                             # 真跨族共享层
│   ├── _base.py                        # TransformerBase
│   ├── attention.py                    # SelfAttention, CrossAttention, TemporalAttention
│   ├── norm.py                         # RMSNorm, LayerNorm, GroupNorm, AdaLayerNorm
│   ├── embeddings.py                   # TimestepEmbedding, RoPE2D, RoPE3D, PatchEmbed2D/3D
│   ├── activations.py                  # silu, gelu
│   ├── schedulers.py                   # FlowMatchEuler, DDPM
│   ├── cache.py                        # ModelCache
│   ├── pipeline.py                     # Pipeline helper
│   ├── scale_factor.py                 # Scale factor helpers
│   ├── vae/                            # 仅跨族通用 VAE
│   │   ├── __init__.py
│   │   ├── autoencoder.py              # Standard SD VAE (ctx-based)
│   │   ├── common.py                   # Latent scale/shift, pixel tensor, PIL post-processing
│   │   ├── tiling.py                   # Interface + dispatch
│   │   ├── tiling_mlx.py               # MLX tiling
│   │   └── tiling_cuda.py              # CUDA tiling (no-op)
│   ├── text_encoders/                  # 统一文本编码器
│   │   ├── __init__.py
│   │   ├── factory.py                  # get_encoder(encoder_type, ctx, ...)
│   │   ├── qwen3.py                    # Body: Qwen3Encoder (~300行, ctx-based)
│   │   ├── qwen3_mlx.py                # Hook: load_weights (~30行)
│   │   ├── qwen3_cuda.py               # Hook: load_weights (~30行)
│   │   ├── clip.py                     # Body: CLIP encoder
│   │   ├── clip_mlx.py                 # Hook
│   │   ├── clip_cuda.py                # Hook
│   │   ├── t5.py                       # Body: T5 encoder
│   │   ├── t5_mlx.py                   # Hook
│   │   └── t5_cuda.py                  # Hook
│   └── bundle_weights/                 # 权重加载
│       ├── __init__.py
│       ├── base.py                     # Interfaces + definitions
│       ├── loader.py                   # Dispatch + shared logic
│       ├── loader_mlx.py               # mx.load / mx.save_safetensors
│       ├── loader_cuda.py              # safetensors.torch
│       ├── remap.py                    # Generic key remap utilities
│       └── resolution.py               # Path resolution
│
├── families/                           # 模型族实现
│   ├── flux1/
│   │   ├── transformer.py              # Flux1Transformer (ctx-only)
│   │   └── weights.py                  # remap_flux1_weights
│   ├── flux2/
│   │   ├── transformer.py              # Flux2Transformer (ctx-only)
│   │   ├── modules.py                  # Flux2Modulation (ctx-only)
│   │   └── weights.py                  # remap_flux2_weights
│   ├── z_image/
│   │   ├── transformer.py              # ZImageTransformer (ctx-only)
│   │   └── weights.py                  # remap_zimage_weights
│   ├── qwen/
│   │   ├── transformer.py              # QwenImageTransformer (thin wrapper)
│   │   ├── modules.py                  # QwenTransformer, Block, Attention, FF (all ctx)
│   │   ├── embeddings.py               # QwenTimesteps, QwenTimestepEmbedding (all ctx)
│   │   └── weights.py                  # remap_qwen_transformer_weights
│   ├── fibo/
│   │   ├── transformer.py              # FIBOTransformer (ctx-only)
│   │   └── weights.py
│   ├── ltx/
│   │   ├── transformer.py              # LTXTransformer (ctx-only)
│   │   └── weights.py                  # remap_ltx_weights
│   ├── wan/
│   │   ├── transformer.py              # WanTransformer (ctx-only)
│   │   └── weights.py                  # remap_wan_weights
│   ├── cogvideox/
│   │   ├── transformer.py              # 对外入口 (re-export)
│   │   ├── transformer_mlx.py          # 完整 MLX 实现
│   │   ├── weights.py                  # remap_cogvideox_weights
│   │   ├── vae.py                      # VAE decode entry (latents PIL)
│   │   ├── vae_mlx.py                  # Full MLX NHWC decoder
│   │   └── vae_cuda.py                 # Full CUDA NCHW decoder
│   └── seedvr2/
│       ├── pipeline.py                 # Main pipeline (ctx-ified)
│       ├── dit.py                      # DiT model (ctx-ified)
│       ├── vae.py                      # VAE interface + dispatch
│       ├── vae_mlx.py                  # MLX VAE
│       ├── vae_cuda.py                 # CUDA VAE
│       └── weights.py                  # Weight schema
│
├── pipelines/                          # 装配线
│   ├── image_pipeline.py               # ImagePipeline (registry-driven)
│   ├── video_pipeline.py               # VideoPipeline (registry-driven)
│   ├── image_upscale_pipeline.py
│   └── video_upscale_pipeline.py
│
├── _transformer_registry.py            # 模型注册
└── danqing_*_engine.py               # Engine 入口
```

---

## 4. 复用层详细设计

### 4.1 RuntimeContext 收窄原则

**原设计（过于全能）**：
```python
class RuntimeContext:
    # 40+ 个方法，试图覆盖所有算子
    def conv3d(self, ...): ...  # 深差异，不应在 ctx
```

**改进设计（分层）**：
```python
class RuntimeContext:
    # Tier 1: 通用算子，所有 DiT 热路径使用
    def Linear(self, in_features, out_features): ...
    def RMSNorm(self, dim): ...
    def matmul(self, a, b): ...
    def reshape(self, x, shape): ...
    def softmax(self, x, axis): ...
    def attention(self, q, k, v, mask=None): ...  # SDPA

    # Tier 2: 平台差异化大的算子，族内自行处理
    # conv3d 移除，由 families/<id>/vae_mlx.py 自行 import mlx.nn.Conv3d
```

### 4.2 Go-style 三文件组语义

**形态 A（默认）差异小**：
```python
# xxx.py = 公共 ctx 实现（主体，非空壳）
# xxx_mlx.py = 少量钩子
# xxx_cuda.py = 少量钩子

# common/text_encoders/qwen3.py (~300行)
class Qwen3Encoder:
    def __init__(self, ctx: RuntimeContext, model_path: str):
        self.ctx = ctx
        nn = ctx
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [Qwen3EncoderLayer(...) for _ in range(num_layers)]
        self.norm = nn.RMSNorm(hidden_size)

        # Backend-differentiated weight loading
        if ctx.backend == "mlx":
            from .qwen3_mlx import load_weights
        else:
            from .qwen3_cuda import load_weights
        weights = load_weights(model_path, ctx)
        self._assign_weights(weights)

# common/text_encoders/qwen3_mlx.py (~30行)
def load_weights(model_path: str, ctx) -> dict:
    import mlx.core as mx
    weights = {}
    for sf in sorted(Path(model_path).glob("*.safetensors")):
        weights.update(dict(mx.load(str(sf))))
    return _strip_prefix(weights, "model.")

# common/text_encoders/qwen3_cuda.py (~30行)
def load_weights(model_path: str, ctx) -> dict:
    import safetensors.torch
    weights = {}
    for sf in sorted(Path(model_path).glob("*.safetensors")):
        weights.update(safetensors.torch.load_file(str(sf), device=ctx.device))
    return _strip_prefix(weights, "model.")
```

**形态 B（兜底）差异大**：
```python
# xxx.py = 对外接口 + dispatch + 共享前后处理
# xxx_mlx.py = 平台完整实现
# xxx_cuda.py = 平台完整实现

# families/cogvideox/vae.py
class CogVideoXDecoder:
    def decode(self, latents_bcthw) -> Any:
        raise NotImplementedError

def create_cogvideox_decoder(ctx, bundle_root, vae_cfg) -> CogVideoXDecoder:
    if ctx.backend == "mlx":
        from .vae_mlx import CogVideoXDecoderMLX
        return CogVideoXDecoderMLX(ctx, bundle_root, vae_cfg)
    elif ctx.backend == "cuda":
        from .vae_cuda import CogVideoXDecoderCuda
        return CogVideoXDecoderCuda(ctx, bundle_root, vae_cfg)
    raise RuntimeError(f"CogVideoX decoder not available for backend: {ctx.backend}")

# families/cogvideox/vae_mlx.py (~700行)
class CogVideoXDecoderMLX(CogVideoXDecoder):
    def __init__(self, ctx, bundle_root, vae_cfg):
        import mlx.core as mx
        import mlx.nn as nn
        # Full NHWC implementation
    def decode(self, latents):
        # NHWC decode logic
```

### 4.3 common/ 边界定义

**留在 common/ 的条件**：
1. 被 >=2 个模型族使用
2. 与具体模型架构无关
3. 可用 ctx 完整表达

**应迁出 common/ 的**：
1. CogVideoX 3D VAE -> `families/cogvideox/vae_*.py`
2. Qwen-Image decoder -> `families/qwen/vae_*.py` 或 `common/vae/qwen_image_decoder_*.py`
3. 族特定的 text encoder -> 如果仅一个族使用，应放在族内

---

## 5. 模型族接入规范

### 5.1 新模型接入步骤（目标：分层短路径）

目标不是把所有模型压成同一种写法，而是把新模型接入拆成“产品登记”和“工程接线”两条短路径：

```
产品路径：models_registry.json  -> 下载中心 / 参数面板 / 安装路径
工程路径：model_configs.py      -> _transformer_registry.py -> families/<family>/
```

**Step 1: 选择形态与 scaffold**

对照 [engine_new_model_checklist.md §0](engine_new_model_checklist.md#0-plugin-shape-decision-a--b--c) 与 **五处注册触点**（registry、`model_configs`、`_transformer_registry`、codec registry、sync）：

```
Shape A: 标准 DiT + ImagePipeline / VideoPipeline  → 模板 family: flux2
Shape B: Job Pipeline（超分、非标准 denoise）       → 模板: seedvr2
Shape C: Generation Facade（音频 / 整栈）           → 模板: ace_step
```

优先运行：

```bash
python scripts/scaffold_image_family.py --family my_dit --class MyDiTTransformer
```

**Step 2: 写 transformer.py（~50-200行，视架构而定）**
```python
# families/new_model/transformer.py
from backend.engine.common._base import TransformerBase
from backend.engine.common.attention import SelfAttention
from backend.engine.common.norm import RMSNorm
from backend.engine.runtime._base import RuntimeContext

class NewModelTransformer(TransformerBase):
    def __init__(self, config, ctx: RuntimeContext):
        super().__init__(config, ctx)
        nn = ctx

        # 使用 common/ 算子定义 DiT 结构
        self.patch_embed = nn.Linear(config.in_channels, config.hidden_size)
        self.blocks = [
            DiTBlock(config.hidden_size, config.num_heads, ctx)
            for _ in range(config.depth)
        ]
        self.final_layer = nn.Linear(config.hidden_size, config.out_channels)

    def forward(self, latents, timestep, txt_embeds=None, sigmas=None):
        ctx = self.ctx
        x = self.patch_embed(latents)
        for block in self.blocks:
            x = block(x, timestep, txt_embeds)
        return self.final_layer(x)

class DiTBlock:
    def __init__(self, dim, heads, ctx: RuntimeContext):
        self.norm1 = ctx.RMSNorm(dim)
        self.attn = SelfAttention(dim, heads, ctx=ctx)
        self.norm2 = ctx.RMSNorm(dim)
        self.mlp = ctx.Linear(dim, dim * 4)

    def forward(self, x, t, txt=None):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

**Step 3: 写 weights.py（~30-200行）**
```python
# families/new_model/weights.py
def remap_newmodel_weights(weights: dict) -> dict:
    """Map HuggingFace/Diffusers weight keys to DanQing structure."""
    remapped = {}
    for k, v in weights.items():
        # Common prefix replacements
        k = k.replace("transformer.", "model.")
        k = k.replace("diffusion_model.", "model.")
        # Specific layer mappings
        k = k.replace("time_embed.", "timestep_embed.")
        remapped[k] = v
    return remapped
```

**Step 4: 配置产品 registry JSON（必需，但瘦身）**
```json
{
  "my-dit-7b": {
    "profile": "image_dit_txt2img",
    "family": "my_dit",
    "backends": ["mlx"],
    "actions": { "create": {} },
    "versions": {
      "default": {
        "source": "huggingface",
        "repo_id": "org/my-dit-7b",
        "local_path": "models/my-dit-7b"
      }
    }
  }
}
```

**Step 5: 配置工程注册（短而显式）**

五处注册触点（Shape A，与 checklist 一致）：

- [ ] `default_config/models_registry.json` + `make sync-models-registry`
- [ ] `backend/engine/config/model_configs.py`（`FAMILY_CONFIG_MAP`）
- [ ] `backend/engine/_transformer_registry.py`（`_TRANSFORMER`、`_WEIGHT_REMAP`、`_TEXT_ENCODER`、可选 `_IMAGE_LORA_MERGE`）
- [ ] `backend/engine/vae_codec_registry.py`（仅当 VAE `_class_name` ≠ 通用 `AutoencoderKL`）
- [ ] family 实现：`families/<family>/transformer.py`、`weights.py`（+ 可选 `text_encoder.py`）

```python
# backend/engine/config/model_configs.py
@dataclass
class MyDitConfig:
    hidden_dim: int = 3072
    num_layers: int = 32
    encoder_type: str = "t5"
    vae_scale: int = 16

FAMILY_CONFIG_MAP["my_dit"] = MyDitConfig

# backend/engine/_transformer_registry.py
_TRANSFORMER["my_dit"] = ("backend.engine.families.my_dit.transformer", "MyDiTTransformer")
_WEIGHT_REMAP["my_dit"] = ("backend.engine.families.my_dit.weights", "remap_newmodel_weights")
```

后续可用 `register_image_family("my_dit")` 之类的 helper 进一步减少样板，但 helper 只能生成显式映射，不能在运行时悄悄跳过失败导入。

### 5.2 族内文件预算约束

```
族内折算逻辑单位 <= 8

计算规则：
- stem.py + stem_mlx.py + stem_cuda.py -> 计 1 单位（同 stem）
- 独立文件（无 _mlx/_cuda 后缀）-> 计 1 单位
- __init__.py 不计

示例：
seedvr2/
├── __init__.py          (0)
├── pipeline.py          (1)  <- 包含 pipeline_mlx.py + pipeline_cuda.py
├── dit.py               (1)  <- 包含 dit_mlx.py + dit_cuda.py
├── vae.py               (1)  <- 包含 vae_mlx.py + vae_cuda.py
└── weights.py           (1)  <- 无后缀，独立
总计: 4 单位
```

### 5.3 降低新模型开发成本的具体机制

1. **Nearest-family 模板**：文档和 scaffold 以 Flux2 / Z-Image / Qwen / SeedVR2 / ACE-Step 为模板入口，而不是以上游目录为模板。
2. **参数模板**：registry 的 `steps`、`guidance`、`width/height`、`seed_support`、`preview_mode` 等重复字段通过 profile 继承，模型条目只写差异。
3. **安装 manifest 校验**：下载后生成或读取 bundle manifest，记录 `transformer`、`text_encoder`、`vae`、`tokenizer` 的实际文件；缺文件直接安装失败或模型不可用。
4. **权重 key parity 报告**：`remap_<family>_weights` 输出 key 与 `_param_map` 做集合比对，报告 missing / unexpected / shape mismatch，不允许静默丢 key。
5. **组件复用矩阵**：新增 family 前先检查 attention、norm、RoPE、scheduler、VAE、text encoder 是否已有 common 实现；CI 输出复用建议而不是只做事后拦截。
6. **LoRA merge contract**：把“是否支持 LoRA、merge 函数入口、patch 目标层”纳入 family contract，避免 Pipeline 分支扩散。

---

## 6. 权重加载与发现机制

### 6.1 ComfyUI 式资产发现，但 DanQing 必须显式校验

ComfyUI 的优势是“把文件放到约定目录后，系统能识别可用组件”。DanQing 可以借鉴这个体验，但不能让生成路径靠猜测继续运行。推荐三层机制：

1. **注册表声明可下载什么**：下载中心从 `models_registry.json` 的 `versions`、`source`、`repo_id`、`local_path`、`size`、`license` 读取数据。
2. **安装 manifest 记录实际有什么**：下载完成后扫描 bundle，生成 `bundle.manifest.json`（可缓存，用户可删除后重建）。
3. **Pipeline 加载时做强校验**：按 family contract 验证组件完整性、权重 key、shape、dtype；失败直接报错，不降级到其他路径。

### 6.2 约定驱动目录结构

```
models/
├── new-model-7b/
│   ├── config.json              # 模型配置（架构参数）
│   ├── diffusion_model.safetensors   # DiT 权重
│   ├── text_encoder/            # 文本编码器权重
│   │   ├── model-00001-of-00002.safetensors
│   │   └── model-00002-of-00002.safetensors
│   └── vae/                     # VAE 权重（可选，复用通用 VAE）
│       └── diffusion_pytorch_model.safetensors
```

约定目录只解决“少写路径”的问题，不解决“自动知道模型结构”的问题。模型结构仍由 `model_configs.py` 和 Transformer 代码定义。

### 6.3 安装 manifest

```json
{
  "schema_version": 1,
  "model_id": "new-model-7b",
  "family": "new_model",
  "components": {
    "transformer": ["diffusion_model.safetensors"],
    "text_encoder": ["text_encoder/model-00001-of-00002.safetensors"],
    "tokenizer": ["tokenizer/tokenizer.json"],
    "vae": ["vae/diffusion_pytorch_model.safetensors"]
  },
  "detected": {
    "parameter_count": 7200000000,
    "weight_format": "safetensors"
  }
}
```

manifest 的用途：

- 下载中心：显示已安装组件、缺失组件、占用空间。
- 运行时：快速定位组件文件，减少递归扫描成本。
- CI/benchmark：记录本次验证使用的 bundle 结构。
- 排错：安装成功但加载失败时，能区分“文件缺失”和“权重 key 不匹配”。

### 6.4 组件路径发现：现有 PathResolution + 目标 WeightResolver

**已落地**：`backend/engine/common/bundle_weights/resolution/path_resolution.py` 的 `PathResolution` — local / HF cache / HF download 规则链，用于权重路径解析。

**目标扩展**（规划中）：在 manifest + family contract 之上增加组件级 resolver：

```python
# 目标 API（尚未实现为独立 WeightResolver 类）
class ComponentResolver:
    def resolve(self, bundle_root: Path, contract: FamilyBundleContract) -> dict:
        """Discover component files and validate against explicit contract."""
        components = {}
        safetensors_files = sorted(bundle_root.rglob("*.safetensors"))
        for sf in safetensors_files:
            ...
        missing = [name for name in contract.required if not components.get(name)]
        if missing:
            raise RuntimeError(f"Bundle {bundle_root} missing required components: {missing}")
        return components
```

扫描只补全**组件路径**；**不推断** DiT 拓扑（仍由 `model_configs` + Transformer 定义）。

### 6.5 通用权重 key 规范化：候选规则 + 审计报告（目标模块）

> 仓库尚无 `common/bundle_weights/remap.py`；family remap 在 `families/<id>/weights.py` + Pipeline `_WEIGHT_REMAP`。下列为**目标**诊断 helper，仅供新 family 开发参考，**不得**在 parity 未达标时静默用于生产加载。

```python
# 目标: common/bundle_weights/remap.py（待建）
class WeightRemapper:
    """Try known key transforms and report why they do or do not match."""

    COMMON_PREFIXES = [
        ("transformer.", "model."),
        ("diffusion_model.", "model."),
        ("model.diffusion_model.", "model."),
        ("unet.", "model."),
    ]

    def normalize(self, weights: dict) -> dict:
        normalized = {}
        for k, v in weights.items():
            for old, new in self.COMMON_PREFIXES:
                if k.startswith(old):
                    k = new + k[len(old):]
                    break
            normalized[k] = v
        return normalized

    def adapt_to_structure(self, weights: dict, target_keys: set) -> dict:
        """Choose a declared mapping only when it reaches exact key parity."""
        candidates = [
            ("identity", lambda k: k),
            ("strip_transformer", lambda k: k.replace("transformer.", "model.", 1)),
            ("strip_diffusion_model", lambda k: k.replace("diffusion_model.", "model.", 1)),
        ]
        reports = []
        for name, fn in candidates:
            remapped = {fn(k): v for k, v in weights.items()}
            keys = set(remapped)
            missing = target_keys - keys
            unexpected = keys - target_keys
            reports.append((name, len(missing), len(unexpected)))
            if not missing and not unexpected:
                return remapped

        raise RuntimeError(f"No weight remap reached exact key parity: {reports}")
```

原则：

- 通用 remap 只能处理常见前缀、层号偏移、命名别名；复杂模型仍写 family-specific `remap_<family>_weights`。
- “90% 匹配即可继续”不适用于生成路径；最多用于诊断建议，最终加载必须 key parity 或显式 allowlist。
- VAE / text encoder 的长映射表可以存在，表长是信息量，不是失败。

---

## 7. 迁移路线图

> **完整 Phase、门禁、PR 切片、Sprint 与成功指标见 §11–§14**。本节保留与历史 checklist 兼容的 Phase 索引；实施时以 §12 为准。

| Phase | 主题 | 详见 |
|-------|------|------|
| 0 | 质量门禁先行（parity、manifest、registry profiles） | §12 Phase 0 |
| 1 | 注册表瘦身 + 下载可观测 | §12 Phase 1 |
| 2 | common 复用层收敛 | §12 Phase 2 |
| 3 | 消除 ctx 孤岛（flux / cogvideox / qwen / seedvr2） | §12 Phase 3 |
| 4 | Pipeline 可观测性（GenerationGraph、LoRA contract） | §12 Phase 4 |
| 5 | 新模型接入 DX（scaffold、family contract） | §12 Phase 5 |
| 6 | 治理扩展（family budget、复用率报告） | §12 Phase 6 |

**注意**：CogVideoX VAE **已在** `families/cogvideox/vae*.py`；Phase 3 对 cogvideox 的任务是 **ctx 化 + parity**，不是再次迁移 VAE。

---

## 8. CI 门禁与治理

> **已落地**：`make check-engine-imports`、`check-engine-family-layout`、`check-engine-governance`、`verify-engine-stack` → [`scripts/check_engine_governance.py`](../scripts/check_engine_governance.py)。  
> **规划中**：下文 `check_family_budget.sh`、`measure_reuse.py` 为示例，尚未独立入库。

### 8.1 零泄漏检查（已通过 governance 实现）

```bash
make check-engine-imports
# 等价: python scripts/check_engine_governance.py --rule imports
```

允许列表见 `scripts/check_engine_governance.allowlist`（imports / layout / primitives / attention 等节）。

### 8.2 族内文件预算检查（规划示例）

```bash
#!/bin/bash
# 目标脚本: scripts/check_family_budget.sh（待建）

MAX_UNITS=8

for family_dir in backend/engine/families/*/; do
    units=0
    stems=()

    for file in "$family_dir"*.py; do
        basename=$(basename "$file" .py)
        # Strip _mlx/_cuda suffix to get stem
        stem=$(echo "$basename" | sed 's/_mlx$//;s/_cuda$//')

        if [[ ! " ${stems[@]} " =~ " ${stem} " ]]; then
            stems+=("$stem")
            ((units++))
        fi
    done

    if [ $units -gt $MAX_UNITS ]; then
        echo "$family_dir exceeds budget: $units units (max $MAX_UNITS)"
        exit 1
    fi

    echo "$family_dir: $units units"
done
```

### 8.3 复用率度量（规划示例）

```python
# 目标脚本: scripts/measure_reuse.py（待建）
import ast
from pathlib import Path

def measure_family_reuse(family_dir: Path) -> dict:
    """Measure reuse degree of a model family."""
    total_lines = 0
    reused_lines = 0

    for py_file in family_dir.rglob("*.py"):
        if py_file.name.startswith("__"):
            continue

        with open(py_file) as f:
            content = f.read()
            tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if "backend.engine.common" in module:
                    reused_lines += len(node.names)

            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if getattr(node.func.value, 'id', None) in ("ctx", "nn"):
                        reused_lines += 1

        total_lines += len(content.splitlines())

    return {
        "family": family_dir.name,
        "total_lines": total_lines,
        "reused_lines": reused_lines,
        "reuse_rate": reused_lines / max(total_lines, 1),
    }
```

---

## 9. 与 ComfyUI 的对比借鉴

| 维度 | ComfyUI | DanQing Studio (目标) | 可借鉴度 |
|------|---------|----------------------|---------|
| **权重发现** | 自动扫描目录 | 下载 registry + 安装 manifest + 强校验 | 高 |
| **格式识别** | 自动推断较多 | 只识别组件路径；结构仍由 family contract 定义 | 中 |
| **结构解析** | 运行时从权重/节点推断 | 代码定义结构，bundle config 只能覆盖白名单字段 | 低 |
| **组件拆分** | MODEL / CLIP / VAE / LoRA 组件化 | Transformer / Text Encoder / VAE / Tokenizer / LoRA manifest | 高 |
| **内存管理** | 自动 VRAM 分页 / unload | RuntimeContext + ModelCache + 显式 release 策略 | 中 |
| **LoRA 注入** | patch model weights | family-level merge contract + key/shape 校验 | 高 |
| **节点连接** | 用户可编辑动态图 | 内部 GenerationGraph，先服务日志/调试/可观测性 | 中 |
| **新模型接入** | 拖文件进目录，社区节点兜底 | scaffold + registry profile + family contract + parity test | 高 |

**核心差异**：
- ComfyUI 是**节点图 + 资产目录 + 社区扩展**生态，容忍运行时动态装配和局部失败。
- DanQing 是**产品化 API/CLI + 下载中心 + 双平台 RuntimeContext**，必须保证动作、参数、下载、任务日志和桌面打包一致。
- DanQing 可以学习 ComfyUI 的组件化体验，但生成路径仍要 fail loud，不能靠“猜到 90%”继续生成。

**可借鉴的具体点**：
1. **组件目录和安装状态**：下载中心按组件展示缺失项，类似 ComfyUI 的模型目录直觉。
2. **模型卡片 + 版本资产**：registry 保留为模型商店数据源，但通过 profile/template 减少重复字段。
3. **内部节点图**：把一次生成拆成 load / encode / denoise / decode / save 节点，先用于日志和调试，后续再考虑可视化工作流。
4. **LoRA/ControlNet 插件契约**：每类插件声明适用 family、patch target、参数 schema 和 merge 函数，不在 Pipeline 写分支。
5. **资产扫描器作为诊断工具**：扫描本地 `models/`，提示“可登记 / 缺 manifest / family 不支持”，但不自动把未知模型加入可生成列表。

**不应照搬的点**：

1. 不用目录扫描替代 `models_registry.json`，否则下载中心、i18n、许可证、推荐模型和安装路径会失去可信数据源。
2. 不在运行时动态加载未知社区节点；DanQing 的桌面打包和双平台能力需要显式 import reachability。
3. 不允许宽松 remap、缺组件跳过、失败后换弱路径；这些会违反 fail-loud 和基准可复现性。
4. 不把 Pipeline 变成任意图执行器；业务入口仍是契约 API/CLI，图只是内部装配和可观测性表示。

### 9.1 建议的内部 GenerationGraph

首期不做 ComfyUI 式开放画布，只把现有 Pipeline 装配显式化：

```python
graph = GenerationGraph(
    nodes=[
        LoadComponent("text_encoder", required=True),
        LoadComponent("transformer", required=True),
        LoadComponent("vae", required=True),
        EncodePrompt("text_encoder"),
        Denoise("transformer"),
        DecodeLatents("vae"),
        SaveAsset(),
    ],
    edges=[
        ("text_encoder", "EncodePrompt"),
        ("transformer", "Denoise"),
        ("vae", "DecodeLatents"),
    ],
)
```

收益：

- 任务日志可以显示节点级进度，用户知道卡在下载、加载、编码还是 VAE 解码。
- 新模型接入时只要声明缺省图形态，Pipeline 仍保持 registry-driven。
- 后续增加 ControlNet、IP-Adapter、LoRA 时可以作为插件节点插入，而不是扩散成 `if family == ...`。

---

## 11. 治理硬约束与质量门禁

> 前提：**不默认降级、不静默出烂图、不偷偷换路径**。质量回归靠 benchmark gate，不靠「能跑就行」。

### 11.1 三层验证（L1 / L2 / L3）

```
L1 静态：verify-engine-stack + consistency + import/layout gates
L2 加载：manifest + key parity + shape check（加载阶段 fail）
L3 生成：bench-sanity / bench-mflux / family-specific parity case
```

**规则**

- 改动 family → 至少跑 L1 + L2 + 该 family sanity
- 有 mflux reference → 必须 PSNR gate
- bench FAIL 不允许用 fallback 路径「先合 PR」
- DiT remap parity < 100% → 加载失败，日志列出 top missing keys
- manifest 缺 required component → 模型「已下载但不可用」，generate 入口直接拒

### 11.2 基准命令

| 类型 | 命令 | 用途 |
|------|------|------|
| 治理栈 | `make verify-engine-stack` | governance + unit tests |
| 一致性 | `make check-consistency` | registry / routes / i18n |
| 无 reference sanity | `make bench-sanity-case ID=` | 输出非空、非 NaN、尺寸正确 |
| mflux PSNR | `make bench-mflux-case ID=` | 数值对齐 reference |
| 音频 RMS | `make bench-audio-sanity` | 音频非静音 |
| 引擎单测 | `make test-engine-unit` | contract / registry 回归 |

### 11.3 用户可见错误（fail loud 落地）

所有加载/生成失败须落到：

- HTTP / CLI 明确错误
- task log / SSE 节点级原因（与 GenerationGraph 锚点一致）
- i18n：`default_config/locales/zh.json` + `en.json`

---

## 12. 分阶段实施计划

### Phase 0：质量门禁先行（1 周）

> 先立「不能烂」的网，再动结构。

**交付**（2026-05 落地状态）

- [x] Weight key parity：`make check-engine-governance --rule parity`
- [x] Bundle completeness：`bundle_manifest.py` + `assert_media_bundle_ready`（`bundle_layout.py`）
- [x] Registry 解析层：`registry_profiles.py`（`profiles` / `parameter_templates`）
- [x] CI：`check_consistency` + registry profile 单测

**门禁**：§11.1 L1；改动 family 时加 L3 sanity / mflux。

### Phase 1：注册表瘦身 + 下载可观测（1–2 周）

**交付**

- [x] `profiles` / `parameter_templates`（`image_dit_standard` / `video_dit_standard`）
- [x] 下载完成后写 `bundle.manifest.json`（`DownloadService`）
- [x] 下载中心 UI：组件完整性（`ModelsView` + `bundle_component_status`）
- [ ] registry 解析报告：继承链、重复参数、无效 profile（CLI 报告，非阻塞）

**不做**：§0.4 所列边界。

### Phase 2：common 复用层收敛（2 周）

**交付**

- [x] `common/bundle_layout.py`：T5 路径 + `assert_media_bundle_ready`
- [ ] `common/text_encoders/`：≥2 族共用 T5 / CLIP / Qwen3（部分已有，未全收敛）
- [x] manifest + family contract 合并在 `bundle_manifest.py`（无 `bundle_weights/` 平行树）
- [ ] CogVideoX：VAE 已在 `families/cogvideox/`；ctx 化与 parity 待做
- [x] `scaffold_image_family.py` 已接 governance / registry 提示

**门禁**

```bash
make check-engine-imports
make check-engine-family-layout
make check-engine-family-primitives
```

### Phase 3：消除 ctx 孤岛（3–4 周，按 family 切片 PR）

| 顺序 | Family | 动作 | 验收 |
|------|--------|------|------|
| 1 | flux1 / flux2 / fibo | 热路径 `ctx.*`（Flux2 RoPE 保留 `_apply_rope_bhsd`） | `flux1-dev-create` / `flux2-klein-9b-create` PASS |
| 2 | cogvideox | ctx 化；VAE parity | video sanity |
| 3 | qwen | MLX `_T` singleton；CUDA parity | bench / smoke |
| 4 | seedvr2 | Go-style stem；Shape B 不变 | `seedvr2-7b-upscale-sanity` PASS |

**每族 PR 检查清单**

- [x] 无新增 Pipeline `if family == ...`（本轮回合）
- [ ] remap parity 100%（parity 门禁 + 按族 bench）
- [ ] `backends` 与实现一致
- [x] bench 不退化（flux1/flux2/seedvr2/z-image-turbo 已跑）

### Phase 4：Pipeline 可观测性（2 周）

**交付**

- [x] 节点日志：`pipeline_progress.pipeline_graph_step`（非开放编辑画布）
- [x] Task 日志 / SSE 与 graph 锚点对齐（image / video / upscale）
- [x] LoRA：不支持则入口拒（`lora_support`）
- [x] 下载中心组件完整性

### Phase 5：新模型接入 DX（持续）

**目标**：标准 Shape A 新 family **0.5–1 天**可 merge（不含数学 parity 对齐）

```bash
python scripts/scaffold_image_family.py --family my_dit --class MyDiTTransformer
make sync-models-registry
make verify-engine-stack
bin/danqing-generate --model my-model --prompt "test" --steps 4
```

**Family contract 最小集**：`encoder_type` / `vae_scale` / required components；`remap_*` + parity；LoRA 支持声明；诚实 `backends`。

### Phase 6：治理与文档（1 周）

- [x] 更新 AGENTS.md、checklist、`engine_refactor_plan.md`（执行索引）
- [ ] 扩展 governance：family budget 报告、复用率度量（§8.2–8.3）
- [ ] 复用率报告纳入 PR review

---

## 13. PR 切片（7 个独立 review）

### PR-1：Registry profiles 解析

- **范围**：`registry_format.py` 或等价层；`/api/registry` 行为不变；`check-consistency` 扩展
- **验收**：现有模型行为不变；JSON 可解析
- **不做**：删 registry；目录扫描生成条目

### PR-2：Bundle manifest + 下载中心组件状态

- **范围**：`DownloadService._finalize_version_install`；family contract 校验；前端组件完整性
- **验收**：缺组件 → 不可 generate，错误明确；manifest 可诊断重建，不自动登记模型

### PR-3：Weight key parity 工具

- **范围**：`check_weight_parity.py` 或挂入 `check_engine_governance.py`
- **验收**：flux2 / z-image 等 parity 通过或可行动报告；失败 CI 红

### PR-4：flux1 / flux2 / fibo ctx 化

- **验收**：`make bench-mflux-case ID=flux2-klein-9b`；sanity；`verify-engine-stack`

### PR-5：qwen ctx 化 + CUDA parity

- **范围**：`transformer_mlx.py` 收敛至 common；CUDA layout 对齐
- **验收**：`bin/danqing-generate --model qwen-image --runtime cuda ...`
- **风险**：高；独立 PR

### PR-6：seedvr2 Go-style 重构

- **范围**：7 逻辑单位 → ≤4；Shape B 不变
- **验收**：upscale sanity；`check-engine-family-layout`

### PR-7：GenerationGraph + 任务日志节点

- **范围**：Pipeline 内部 DAG；SSE 节点 progress；首期不开放用户编辑图
- **验收**：失败可定位到节点；无新增 family 硬分支

每个 PR 正文须含：**范围 / 不做 / 验收命令 / fail-loud 行为说明**。

---

## 14. 第一 Sprint 与 12 周成功指标

### 14.1 第一 Sprint（Week 1–2，优先开工）

1. **Weight key parity CLI** — 加载前拦截 silent 丢 key
2. **Registry profile 解析** — 减重复登记，不改现有运行时行为
3. **Download 后写 manifest** — 安装阶段暴露缺文件
4. **flux2 bench gate 写入 PR 模板** — 改动 engine family 必附 bench 结果

### 14.2 12 周成功指标

| 指标 | 现状 | 目标 |
|------|------|------|
| 新 Shape A family 接入 | 2–3 天 | 0.5–1 天 |
| registry 单模型重复字段 | 高 | profile 后减少 50%+ |
| 安装后首次 generate 失败率 | 高（缺文件/key） | 前置到下载/安装阶段 |
| ctx 孤岛 family 数 | qwen / seedvr2 等 | 0（或显式 Shape B/C 文档化） |
| bench FAIL 静默合并 | 偶发 | 0（CI 拦截） |
| Pipeline family 硬分支 | 存量 | 只增 registry-driven 开关 |

### 14.3 GitHub Issue 标题模板

1. `[engine] Phase 0: weight key parity gate + bundle completeness check`
2. `[engine] Phase 1: registry profiles/templates + manifest on download`
3. `[engine] Phase 2: common text_encoders + bundle_weights convergence`
4. `[engine] Phase 3a: flux1/flux2/fibo ctx migration`
5. `[engine] Phase 3b: qwen ctx + CUDA parity`
6. `[engine] Phase 3c: seedvr2 Go-style refactor`
7. `[engine] Phase 4: GenerationGraph + node-level task logs`

---

## 10. 附录：代码示例

### A. 完整的 Qwen ctx 化示例

```python
# families/qwen/transformer.py (改进后)
from backend.engine.common._base import TransformerBase
from backend.engine.common.attention import SelfAttention
from backend.engine.common.norm import RMSNorm
from backend.engine.common.embeddings import TimestepEmbedding
from backend.engine.runtime._base import RuntimeContext

class QwenImageTransformer(TransformerBase):
    def __init__(self, config, ctx: RuntimeContext):
        super().__init__(config, ctx)
        nn = ctx

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = [
            QwenTransformerBlock(config.hidden_size, config.num_heads, ctx)
            for _ in range(config.num_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size)

    def forward(self, latents, timestep, txt_embeds=None, sigmas=None):
        ctx = self.ctx
        x = self.embed_tokens(latents)
        for block in self.blocks:
            x = block(x, timestep, txt_embeds)
        return self.norm(x)

class QwenTransformerBlock:
    def __init__(self, dim, heads, ctx: RuntimeContext):
        self.input_norm = ctx.RMSNorm(dim)
        self.attn = SelfAttention(dim, heads, ctx=ctx)
        self.post_norm = ctx.RMSNorm(dim)
        self.mlp = ctx.Linear(dim, dim * 4)

    def forward(self, x, t, txt=None):
        residual = x
        x = self.attn(self.input_norm(x))
        x = residual + x
        residual = x
        x = self.mlp(self.post_norm(x))
        return residual + x
```

### B. 通用权重 remap 示例

```python
# common/bundle_weights/remap.py
class AutoRemapper:
    """Try common key mapping rules and require exact parity."""

    def remap(self, weights: dict, target_keys: set[str]) -> dict:
        """
        weights: original weight dict (key -> tensor)
        target_keys: flattened TransformerBase parameter key set

        Try common mapping rules, but do not continue unless the chosen rule
        exactly matches the target parameter key space.
        """
        rules = [
            ("identity", lambda k: k),
            ("transformer_prefix", lambda k: k.replace("transformer.", "model.", 1)),
            ("diffusion_prefix", lambda k: k.replace("diffusion_model.", "model.", 1)),
            ("unet_prefix", lambda k: k.replace("unet.", "model.", 1)),
            ("time_alias", lambda k: k.replace("time_embed.", "timestep_embed.", 1)),
        ]

        reports = []
        for name, rule in rules:
            remapped = {rule(k): v for k, v in weights.items()}
            keys = set(remapped)
            missing = sorted(target_keys - keys)
            unexpected = sorted(keys - target_keys)
            reports.append({"rule": name, "missing": len(missing), "unexpected": len(unexpected)})
            if not missing and not unexpected:
                return remapped

        raise RuntimeError(f"No remap rule reached exact key parity: {reports}")
```

### C. 新模型接入完整示例

```python
# families/my_dit/transformer.py
from backend.engine.common._base import TransformerBase
from backend.engine.common.attention import SelfAttention
from backend.engine.common.norm import RMSNorm
from backend.engine.runtime._base import RuntimeContext

class MyDiTTransformer(TransformerBase):
    def __init__(self, config, ctx: RuntimeContext):
        super().__init__(config, ctx)
        nn = ctx

        self.patch_embed = nn.Linear(config.in_channels, config.hidden_size)
        self.time_embed = nn.Embedding(config.max_period, config.hidden_size)
        self.blocks = [
            DiTBlock(config.hidden_size, config.num_heads, ctx)
            for _ in range(config.depth)
        ]
        self.final = nn.Linear(config.hidden_size, config.out_channels)

    def forward(self, latents, timestep, txt_embeds=None, sigmas=None):
        ctx = self.ctx
        x = self.patch_embed(latents)
        t = self.time_embed(timestep)
        for block in self.blocks:
            x = block(x, t, txt_embeds)
        return self.final(x)

class DiTBlock:
    def __init__(self, dim, heads, ctx: RuntimeContext):
        self.norm1 = ctx.RMSNorm(dim)
        self.attn = SelfAttention(dim, heads, ctx=ctx)
        self.norm2 = ctx.RMSNorm(dim)
        self.mlp = ctx.Sequential(
            ctx.Linear(dim, dim * 4),
            ctx.SiLU(),
            ctx.Linear(dim * 4, dim),
        )

    def forward(self, x, t, txt=None):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# families/my_dit/weights.py
def remap_mydit_weights(weights: dict) -> dict:
    remapped = {}
    for k, v in weights.items():
        k = k.replace("transformer.", "model.")
        k = k.replace("time_emb.", "time_embed.")
        remapped[k] = v
    return remapped

# default_config/models_registry.json
{
  "my-dit-7b": {
    "profile": "image_dit_txt2img",
    "family": "my_dit",
    "backends": ["mlx"],
    "actions": { "create": {} },
    "versions": {
      "default": {
        "source": "huggingface",
        "repo_id": "org/my-dit-7b",
        "local_path": "models/my-dit-7b"
      }
    }
  }
}
```

---

## 总结

| 指标 | 改进前（现状） | 改进后（目标） | 说明 |
|------|----------------|----------------|------|
| 单模型样板代码量 | 多处重复 registry + family 样板 | scaffold + profile + snippets | 目标态 |
| 新模型接入时间 | 2–3 天 | Shape A 目标 0.5–1 天 | 不含数学 parity 对齐 |
| CUDA 支持 | Qwen 等 MLX/CUDA 双实现分叉 | ctx-only 受益；差异大时显式 `*_cuda.py` | 边界清楚 |
| 注意力 / norm 维护 | 多 family 各一套 | common 统一 | **估算**，以 governance 报告为准 |
| 文本编码器 | family 内分散 + common 部分 | 跨族进 common | 进行中 |
| seedvr2 逻辑单位 | 7 | 目标 4 | layout gate |
| 下载中心数据源 | 与工程注册职责易混 | 产品 registry 保留 | **已明确** |
| 安装排错 | 多在 load/generate 才失败 | manifest + component completeness | **规划中** |

**核心原则**：
1. RuntimeContext 收窄，不全能化
2. common/ 仅放真跨族组件
3. Go-style 三文件组，同 stem 计 1 单位
4. `models_registry.json` 保留为下载中心和产品事实来源
5. profile/template 减少重复登记，解析后必须得到完整 registry
6. manifest 负责资产发现，family contract 负责结构校验
7. CI 强制边界，防止复用孤岛再生
8. fail loud 与三层验证（§11）；Phase / PR / Sprint 见 §12–§14

**认可条目索引**（本文自洽，无需另读 plan 即可评审）：

| 类别 | 章节 |
|------|------|
| 硬约束、不做、债务优先级 | §0.2–§0.5 |
| 现状与诊断 | §1–§2 |
| 目标分层、registry、复用层 | §3–§4 |
| 接入规范、manifest、parity | §5–§6 |
| CI / ComfyUI 边界 | §8–§9 |
| 质量门禁、Phase、PR、Sprint、指标 | §11–§14 |

---

## 变更记录

| 日期 | 说明 |
|------|------|
| 2026-05-29 | 二轮复核：对齐当前仓库树；区分当前态/目标态 |
| 2026-05-29 | 三轮：合并 engine_refactor_plan 全部认可条目入 §0.2–§0.5、§3.0、§11–§14；本文为认可条目全集 |
