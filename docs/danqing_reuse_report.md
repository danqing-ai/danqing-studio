# DanQing Studio 数学算子复用报告

**目标**：通过分层抽象数学算子，减少新模型接入的代码量，使新族尽量只写「结构定义 + 权重映射 + registry」。

**复核**：2026-05-28（分支 `docs/danqing-reuse-report-review`），对照当前 `main` 上 `backend/engine/` 与 `make check-engine-governance` 门禁。

---

## 0. 复核结论（相对初稿）

| 初稿论断 | 复核结果 |
|----------|----------|
| flux1/flux2/fibo/cogvideox 在 `forward()` 里直接 `mx.*` | **已过时**：`transformer.py` 仅为对外 stem；`mx.*` 在 `*_mlx.py`，符合导入边界 |
| qwen 族「零 RuntimeContext、完全重复 attention/norm」 | **部分错误**：`transformer_mlx.py` 已复用 `common/attention`、`embeddings`、`norm` 函数；仍为 **MLX-only `nn.Module` 单文件**（~690 行），无 CUDA DiT |
| `common/text_encoders/`「规划中」 | **已实现**：`t5` / `clip` / `qwen25vl` / `qwen3`（含部分 `*_cuda.py`） |
| CogVideoX VAE 在 `common/vae/cogvideox_decoder.py` | **不存在**；已在 `families/cogvideox/vae.py` + `vae_mlx.py` |
| P2「CI 强制 mlx/torch 边界」未做 | **已做**：`make check-engine-imports` + 缩小的 allowlist（当前 mainly `qwen/weights.py`） |
| P2「族内文件预算」未做 | **部分已做**：`make check-engine-family-layout` + `check-engine-family-primitives` 等；SeedVR2 仍超理想文件数 |
| 新模型 ~100 行即可接入 | **仍为目标态**；当前干净族（如 z_image DiT stem）仍约 **数百行** 且多数仅 MLX DiT |

---

## 1. 当前架构分层

### Layer 0: RuntimeContext (`backend/engine/runtime/_base.py`)

- **窄契约**（见文件头注释）：模块工厂 + 张量/内存 API + 权重 I/O；**新算法默认不进 ABC**，平台差异用 `xxx_mlx.py` / `xxx_cuda.py`（见 [`docs/dual_platform_architecture.md`](dual_platform_architecture.md) §8.5）。
- `mlx` / `torch` 顶层导入仅允许：`runtime/*.py`、`*_mlx.py`、`*_cuda.py`（`make check-engine-imports`）。

### Layer 1: `common/` 共享组件

| 区域 | 内容 | 复用方式 |
|------|------|----------|
| `_base.py` | `TransformerBase` | 各 image/video DiT 入口 |
| `attention.py` | `scaled_dot_product_attention_*`、mask 构建、`SelfAttention(ctx)` 等 | flux/z_image/qwen/cogvideox/wan/seedvr2 等多族 |
| `norm.py` | RMS/Layer/Ada、modulation unpack | 同上 |
| `embeddings.py` | timestep、RoPE、patch embed 等 | 同上 |
| `schedulers.py` | FlowMatch、DDPM 等 | Pipeline |
| `vae/` | 通用 decoder/encoder、tiling、`qwen_image/` 子树 | 跨族 + Qwen 图像 VAE |
| `text_encoders/` | T5、CLIP、Qwen2.5-VL、Qwen3（懒加载） | 部分族仍保留族内 encoder 文件 |
| `bundle_weights/` | 权重加载与 remap 辅助 | Pipeline / 各 `weights.py` |

### Layer 2: `families/<id>/`（Go 式 stem）

典型形态（与 [`.cursor/rules/model-migration.mdc`](../.cursor/rules/model-migration.mdc) 一致）：

- `transformer.py` — 对外接口 + dispatch（常 re-export `transformer_mlx`）
- `transformer_mlx.py` / `transformer_cuda.py` — 平台实现（**同 stem 计 1 逻辑单位**）
- `weights.py` — `remap_*` + Pipeline 注册
- 可选：`text_encoder.py` + `*_mlx.py` / `*_cuda.py`，`vae.py` + `*_mlx.py`

**已无** 全仓库统一的 `families/*/modules.py` 平行小树。

### Layer 3: `pipelines/`

- `image_pipeline.py` / `video_pipeline.py` — registry 驱动，族无关业务分支。

---

## 2. 各模型族复用现状（2026-05）

| 模型族 | DiT / 主路径 | `common/` 算子 | 双端（MLX+CUDA） | 主要缺口 |
|--------|--------------|----------------|-----------------|----------|
| **z_image** | `transformer_mlx.py` ~764 行 | attention、embeddings、norm | 文本 encoder 有 `text_encoder_cuda.py`；**DiT 仅 MLX** | DiT CUDA stem |
| **ltx** | 单文件 `transformer.py` ~291 行 | 同上（ctx 贯穿） | **仅 MLX**（无 `transformer_cuda`） | 拆 stem + CUDA |
| **wan** | `transformer_mlx.py` | 部分 common + 族内 `WanSelfAttention` | VAE 有 mlx；**DiT 仅 MLX** | 族内 attention 与 common 收敛 |
| **flux1** | `transformer_mlx.py` ~554 行 | SDPA、patch、norm | **仅 MLX**；T5/CLIP 在族内 `*_mlx.py` | 文本 encoder 迁入/复用 `common/text_encoders` |
| **flux2** | `transformer_mlx.py` ~564 行 | 同上 | **仅 MLX** | 同上 + VAE 在 `vae_mlx.py` |
| **fibo** | `transformer_mlx.py` ~636 行 | embeddings、SDPA | **仅 MLX** | 文本 encoder 仍族内 |
| **cogvideox** | `transformer_mlx.py` ~486 行 | attention、embeddings、norm | **仅 MLX**；VAE 已在族内 | RoPE 在 `rotary_mlx.py`（合规） |
| **qwen** | `transformer_mlx.py` ~690 行 | **已接** common 函数层 | **仅 MLX**；`weights.py` 在 import allowlist | DiT 改 ctx/`SelfAttention` 或补 `transformer_cuda`；去掉 allowlist |
| **seedvr2** | 7× `*_mlx.py` + `upscale.py` / `weights.py` stem | `dit_mlx`/`vae_mlx` 用 SDPA、RMS 等 | **仅 MLX** upscale 孤岛 | `job_mlx` 含 schedule+result；`video_restore_mlx` 可再收 |
| **hunyuan** | `transformer_mlx.py` + 族内 VAE/SR | 部分 common | **仅 MLX** | 与视频族治理对齐 |
| **ace_step** | `transformer.py` + **mlx/cuda** | common 全套 | **音频族范例：双端** | 非图像 DiT，作参考 |

**说明**：在 `*_mlx.py` 内使用 `mx.*` **不是**「漏网」，而是当前治理允许的 Tier-2 实现；初稿将「`transformer.forward` 里 `mx.reshape`」列为 P1，**已不再适用**。

---

## 3. 理想 vs 现实（量化，近似）

### 目标态（治理锚点，见 `AGENTS.md`）

```
families/new_model/transformer.py   # stem + dispatch
families/new_model/transformer_mlx.py
families/new_model/transformer_cuda.py   # 若 registry 声明双 backends
families/new_model/weights.py            # remap_* 注册
default_config/models_registry.json
```

族内 **折算逻辑单位 ≤ 8**；`stem.py` + `stem_mlx.py` + `stem_cuda.py` = **1** 单位。

### 现实（DiT 单文件行数，不含 weights/VAE/encoder）

| 族 | 主 DiT 文件行数 | 对 common 依赖 | 备注 |
|----|-----------------|----------------|------|
| ltx | ~291 | 高（单文件 ctx） | 最接近「结构-only」形态，但无 CUDA |
| cogvideox | ~486 | 高 | VAE 已族内 |
| flux1 / flux2 | ~554 / ~564 | 高 | 文本侧仍分散 |
| fibo | ~636 | 中高 | |
| qwen | ~690 | 中高（函数级复用） | 非初稿 ~1600 行；仍 nn.Module 堆叠 |
| z_image | ~764 | 高 | |
| seedvr2 | ~3.4k（多 mlx 文件合计） | 中（dit/vae 有 common） | 文件数仍偏多 |

**维护面**：attention/SDPA/RoPE/modulation 已有专项门禁（`check-engine-attention-paths` 等），但 **各族仍保留自定义 Block 类**，未统一到单一 `SelfAttention(ctx)` 调用风格。

---

## 4. 根因（更新后）

### 4.1 双端缺口集中在 DiT，而非「完全未复用 common」

- Qwen 已使用 `scaled_dot_product_attention_bhsd_mx`、`apply_complex_rope_bshd`、`unpack_modulation_3way` 等，但 **未** 提供 `transformer_cuda.py`，且 `weights.py` 仍依赖 MLX remap（allowlist 明示）。
- 多数图像族 DiT 仅有 `transformer_mlx.py`；**ace_step** 是少数具备 `transformer_cuda.py` 的族。

### 4.2 文本编码器：common 已有，族内重复未清完

```
common/text_encoders/     → T5, CLIP, Qwen25VL, Qwen3（部分 CUDA）
families/flux1/         → text_encoder.py（实现 common/flux1_dual）
families/flux2/         → text_encoder.py（实现 common/qwen3 Flux2TextEncoder）
families/qwen/          → text_encoder.py（实现 common/qwen_image_mlx）
families/wan/           → text_encoder.py（实现 common/wan_umt5_mlx）
families/z_image/       → text_encoder_{mlx,cuda}.py（双端较好）
```

统一方向不变：按 registry `encoder_type` 收敛到 `common/text_encoders/`，族内只保留薄 dispatch。

### 4.3 `common/vae/` 边界

- **合理**：通用 `decoder.py` / `encoder.py`、tiling、`qwen_image/*`（Qwen 图像 VAE 专用子树）。
- **已纠正**：CogVideoX 3D VAE **不在** `common/vae/` 杂项；无需再迁。
- **警惕**：避免新增「族专属解码器」进 `common/vae/`（违反 model-migration 规则时用 registry + 族内 stem）。

### 4.4 SeedVR2 文件预算

当前族根目录 **9 个** `*_mlx.py`（`dit`、`vae`、`embed`、`job`、`preprocess`、`schedule`、`result`、`video_restore`、`weights`），**无** 对外 `transformer.py` / `pipeline.py` stem。目标仍是合并为少量 stem（各配 `_mlx`/`_cuda`，同 stem 计 1 单位）。

---

## 5. 改进收益（仍为方向性估算）

| 指标 | 当前 | 目标 | 说明 |
|------|------|------|------|
| 新图像族 DiT 行数 | 约 300–800（单 mlx 文件）+ weights | stem 薄封装 + 双端 hook | 取决于是否强制 CUDA |
| 接入时间 | 数天级（含权重对齐） | 数小时级 | 需 common 与 registry 模板成熟 |
| CUDA 支持 | 声明 `backends` 的模型需真实 `*_cuda` 或 ctx 路径 | 缺则 **fail loud** | 不可静默降级 |
| SDPA/RoPE 维护 | common 函数 + 各族 Block | 优先扩 common，减族内副本 | 已有 CI 辅助 |

---

## 6. 动作清单（按优先级，对齐仓库现状）

### 已完成或部分完成

- [x] **mlx/torch 导入边界**：`make check-engine-imports`（allowlist 已清空；`qwen/weights.py` 仅为懒加载 facade，`weights_mlx.py` 承载 MLX 映射）。
- [x] **Flux2 文本编码器**：`Flux2TextEncoder` 迁入 `common/text_encoders/qwen3_mlx.py`，族内 `text_encoder.py` 薄封装。
- [x] **FIBO / SeedVR2 对外 stem**：`fibo/text_encoder.py`、`seedvr2/upscale.py` + registry / Pipeline 经 stem 引用。
- [x] **文本编码器迁入 common**：`flux1_dual` + `flux1_t5/clip_mlx`、`qwen_image_mlx`、`wan_umt5_mlx`；族内保留 deprecated 重导出路径。
- [x] **SeedVR2 文件合并**：`embed_mlx` 并入 `preprocess_mlx`（少 1 个 mlx 文件）。
- [x] **Qwen DiT stem**：`transformer.py` 双端 dispatch 占位（CUDA 显式 fail loud）。
- [x] **族目录平行树禁令**：`make check-engine-family-layout`。
- [x] **CogVideoX VAE 族内化**：`families/cogvideox/vae.py`。
- [x] **公共 text_encoders 目录**：`common/text_encoders/`。
- [x] **多族 DiT 使用 common SDPA/embed/norm 函数**（含 qwen、seedvr2 dit）。

### P0 — 最大复用/双端缺口

1. **Qwen 图像 DiT CUDA**：初版 `transformer_cuda.py`（diffusers `QwenImageTransformer2DModel` + bundle 加载）与 `text_encoder_cuda.py`；registry `backends: [mlx, cuda]`。需在真机 bundle 上跑通生成 parity。
2. **文本编码器收敛**：flux1/flux2/qwen/wan 已迁入 `common/text_encoders/`；flux1 仍保留 mflux 对齐 T5/CLIP（未改用 generic `T5Encoder`）。

### P1 — 结构与一致性

3. **图像 DiT 双端模板**：以 ace_step / z_image text encoder 为参考，为 flux/ltx/wan 等补 `transformer_cuda.py` 或等价 ctx 路径（与 registry `backends` 一致）。
4. **SeedVR2 合并 stem**：`embed`/`schedule`/`result` 已并入 `preprocess_mlx` / `job_mlx`；`weights.py` + `upscale.py` 对外 stem；目标仍收拢 `video_restore_mlx` 与 CUDA。

### P2 — 持续治理（已存在，保持收紧）

5. **缩小** `engine_backend_import_allowlist.txt`（禁止无审查扩容）。
6. **维持** `check-engine-attention-paths` / `sdpa` / `rope` / `modulation` 与 `verify-engine-stack` 在 PR 前通过。

---

## 7. 架构图（与现网一致）

```
REST API / CLI
      ↓
TaskScheduler → DanQing*Engine
      ↓
ImagePipeline / VideoPipeline (registry)
      ↓
RuntimeContext (mlx.py | cuda.py)     Registry + model_configs
      ↓
common/  attention | norm | embeddings | schedulers | vae | text_encoders | bundle_weights
      ↓
families/<family>/
  transformer.py          ← dispatch / 对外契约
  transformer_mlx.py      ← Tier-2 实现（允许 mx.*）
  transformer_cuda.py     ← 可选（ace_step 等）
  weights.py              ← remap_* → _WEIGHT_REMAP
      ↓
新族目标：薄 stem + common 原语 + registry；禁止上游平行目录树
```

---

## 8. 结论

> **当前复用问题的主因，已从「common 粒度不对」转为「双端与 stem 未收敛、族内仍有重复 encoder/Block」。**
>
> `common/` 已提供 SDPA、norm、embedding 原语及 `SelfAttention(ctx)`；多数族的 **导入边界** 已合规（`mx.*` 在 `*_mlx.py`）。剩余工作集中在：**(1) Qwen/SeedVR2 等 MLX-only 族的 CUDA 或 ctx 补齐；(2) 文本 encoder 从族目录迁入 `common/text_encoders/`；(3) SeedVR2 等超大 mlx 面合并为少量 stem。**
>
> 新模型 **~100 行** 仍是 registry + 治理下的目标态，不是当前默认现实；以 **ltx ~291 行单文件 ctx DiT** 为近期可参考下限。
