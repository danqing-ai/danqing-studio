# 量化推理显存节省 — 实现方案与进度

**状态**：Phase 0–4 已实现（DiT 量化推理 + TE/VAE 短载与可选量化加载）  
**目标**：本地量化（`source_type: derived`）与预下载量化（`source_type: prequantized`）的 int4/int8 bundle，在 **MLX 推理时以 `QuantizedLinear` 常驻显存**，显著降低 DiT 峰值/稳态 VRAM。  
**非目标**：本阶段不解决 CUDA 侧量化推理；不默认量化 VAE / Text Encoder（可单列后续阶段）。

实现入口：

- `backend/engine/common/bundle/quant_inference.py` — mode 解析、缓存估算、LoRA 冲突
- `backend/engine/common/model/quantized_load.py` — 量化加载（无 dequantize）
- `backend/engine/pipelines/image_model_load.py` / `video_model_load.py` — 标准 image/video DiT
- `backend/engine/families/ltx/transformer_mlx.py` — LTX 2.3 收敛
- `backend/engine/families/ace_step/transformer.py` + `audio_model_load.py` — ACE-Step 收敛
- `backend/engine/families/seedvr2/stem_mlx.py` — SeedVR2 预量化 vs 运行时量化

---

## 1. 问题陈述

| 现状（改前） | 用户预期 |
|------|----------|
| int4/int8 主要省磁盘与下载 | 装 int4 后能在更小显存机器上跑通 |
| `load_weights` 统一反量化 | 权重以 4/8 bit 参与 matmul |
| `quantize_runtime()` 已实现但未接入主路径 | 选量化版本即启用 |
| LTX / SeedVR2 各自为政 | 框架统一、registry 驱动 |

**产品承诺（Phase 1 后）**：registry 声明了 `quantization.bits ∈ {4,8}` 且 `scheme: mlx_affine` 的版本，在 MLX 后端上推理时 DiT **默认**以量化形态运行；若无法做到须 **fail loud**（任务日志 + 明确错误），不得静默退回稠密推理。

---

## 2. 设计原则

1. **Registry 驱动**：行为由 `distribution.versions.<key>.quantization` 决定，禁止 Pipeline `if family == …` 分支。
2. **Fail loud**：MLX 缺能力、层无法量化、LoRA 与量化冲突、CUDA 选量化版本 → 显式 `RuntimeError` + i18n。
3. **单一加载契约**：扩展 `TransformerBase` / `load_*_transformer`，LTX 等特例逐步收敛到同一钩子，避免第三条加载路径。
4. **LoRA 边界清晰**：首版声明「量化 DiT + LoRA 同时启用 → 报错」；后续 Phase 3 再支持 merge 后重量化。
5. **Engine 净 LOC**：复用 `quantize_runtime`、`_mlx_affine_infer_bits_and_group_size`、`read_bundle_affine_bits_if_quantized`；LTX 收敛留 Phase 2。

---

## 3. 目标架构

### 3.1 推理形态（两种 mode）

| Mode | 触发条件 | 显存 | 用途 |
|------|----------|------|------|
| `dense` | fp16/bf16 全精度版本；或 `quantization.inference: dense` | 高 | 默认全精度、CUDA、对标 |
| `quantized` | 版本含 `quantization.bits` 且 backend=mlx；bundle 含 affine 张量 | 低（DiT） | derived / prequantized |

**判定**：`resolve_inference_weight_mode(entry, version_key, ctx, weight_keys, bundle_affine_bits)`。

### 3.2 加载流水线（量化推理）

```
实例化 model (nn.Linear 骨架)
  → sanitize 键名
  → 扫描 checkpoint：收集含 {weight, scales, biases} 的层前缀集合 Q
  → apply_quantized_skeleton(model, Q, bits, group_size)
  → 直接灌 packed 张量（禁止 dequantize）
  → ctx.eval(parameters)
  → after_load_weights → [LoRA：dequantize→merge→requantize touched 层]
```

### 3.3 `TransformerBase.load_weights`

- `inference_mode=None` 或 `kind=dense`：原有 dequantize 路径（兼容）
- `kind=quantized`：委托 `load_weights_quantized_inference()`

---

## 4. 实现阶段与进度

### Phase 0 — 契约与观测 ✅

- [x] `WeightInferenceMode` + `resolve_inference_weight_mode()`
- [x] 任务日志：`load_transformer` 输出 `inference=quantized bits=N cache_est_gb=…`
- [x] 单元测试：mode 解析、registry bits、LoRA 冲突、缓存估算
- [x] 文档：`docs/engine_architecture.md` §12.6

### Phase 1 — DiT 量化推理（核心）✅

- [x] `load_weights_quantized_inference()`：predicate 量化 + 直接加载 affine 张量
- [x] 接入 `load_image_transformer` / `load_video_transformer`
- [x] 覆盖走 `TransformerBase.load_weights` 的 image/video DiT（经 stem 委托）
- [x] **LoRA（Phase 3）**：量化 DiT + adapters → 重量化合并；`quantized_lora: false` 时 `RuntimeError`
- [x] CUDA 选用量化版本 → fail loud（`error.quantized_inference_mlx_only`）
- [x] `ModelCache` key 追加 `:q4` / `:q8` / `:dense`；`put` 使用推理态 size 估算
- [ ] **手动验收**：`mlx.active_memory_gb()` fp16 vs int4 对比（需本机有模型）

### Phase 2 — 家族收敛与边角 ✅

- [x] LTX：`load_ltx23_x0_model` → `LTX23Transformer.load_weights`（`module_root=self.model`）；移除 `_apply_ltx23_quantization`
- [x] SeedVR2：`stored_q` 预量化 → `dit.load_weights(quantized)`；`fp16 + -q` 仍走 `quantize_runtime`
- [x] ACE-Step DiT：`AceStepTransformer.load_weights` + `audio_model_load` 传 registry entry/version
- [x] `resolve_dit_inference_weight_mode`：registry 优先；无 entry 时从 bundle 元数据推断
- [x] 无法 `in_features % 64 == 0` 的 Linear：加载时 fail loud（`apply_quantized_skeleton`）

### Phase 3 — LoRA + 量化 ✅

- [x] 策略 A：`merge_lora_adapters_common` 对 uint32 权重 dequantize → 加 delta → `to_quantized`（`quantized_lora.py`）
- [x] 策略 B：registry `quantization.quantized_lora`（默认 **true**；显式 `false` 时 fail loud）
- [x] `allow_cache`：有 adapters 仍跳过缓存（行为不变）；LoRA 日志含 `requantized_layers=yes`

### Phase 4 — TE / VAE 显存（可选）✅

- [x] `resolve_component_inference_weight_mode()`（`text_encoder` / `vae`；默认 dense，bundle 含 `*.scales` 时继承 DiT bits）
- [x] VAE：`create_loaded_vae_decoder` 支持量化 `Linear`（`mid_attn`）；decode 后 `release_vae_decoder_memory`（`vae_release_after_decode` 默认 true）
- [x] T5：`T5Encoder` 检测 affine 权重时走 `load_weights_quantized_inference`（无 scales 时行为不变）
- [x] TE/VAE derived 转换：`quantization.text_encoder.bits` / `quantization.vae.bits` → `convert_model` 额外量化子目录（`derived_quant_mlx.py`）
- [x] 参考模型：`flux2-klein-4b` 的 `int4`/`int8` derived 声明 `text_encoder.bits`；Qwen3 TE 加载接 registry + affine 推理

### Phase 5 — CUDA（占位 / 远期）🔒

**非本里程碑交付**；单独 PR 再开。当前已实现占位行为：

- [x] CUDA 后端 + int4/int8 版本 → `error.quantized_inference_mlx_only`（fail loud，不静默降级）
- [ ] `quantize_runtime` CUDA 或 torchao 路径（远期）

---

## 5. LoRA 与缓存（Phase 1 生效）

| 场景 | 行为 |
|------|------|
| 量化版本 + 无 LoRA | 正常推理；可 `ModelCache` |
| 量化版本 + LoRA | 合并后 touched 层重量化；`quantized_lora: false` 时拒绝 |
| fp16 + LoRA | 现状不变 |
| 缓存 key | `image:{id}:{version}:q4` 等 |

---

## 6. 本地量化 / 预下载量化

二者产出格式相同，**加载路径完全共用**。`convert_model` 默认只量化 DiT；可选在 derived 版本声明：

```json
"quantization": {
  "bits": 4,
  "scheme": "mlx_affine",
  "text_encoder": { "bits": 4 },
  "vae": { "bits": 4 }
}
```

则 `text_encoder/`、`vae/`（及 `text_encoder_2/`）下 safetensors 一并走同一 MLX affine 算法；`config.json` 等 sidecar 原样复制。

---

## 7. 验收标准

### 7.1 功能

- [ ] 2 个 image + 1 个 video int4 模型 MLX 生成 smoke（需本机权重）
- [x] derived / prequantized 同一 `resolve_inference_weight_mode` + `load_weights` 路径
- [x] CI：`QuantizedInferenceModeTests` in `tests/engine_unit.py`

### 7.2 显存（MLX）

- [ ] int4 DiT 稳态显存较 fp16 **≥40% 下降**（手动记录 `active_memory_gb`）

### 7.3 失败路径

- [x] CUDA + int4 版本 → 明确错误
- [x] 声明量化但 bundle 无 `*.scales` → 明确错误
- [x] registry bits 与 metadata 冲突 → 明确错误
- [x] 量化 + LoRA → 默认允许（重量化）；`quantized_lora: false` 时明确错误

### 7.4 CI

```bash
python -m py_compile <touched>
make test-engine-unit
make check-engine-governance
```

---

## 8. 关键文件

| 文件 | 状态 |
|------|------|
| `backend/engine/common/bundle/quant_inference.py` | ✅ 已建 |
| `backend/engine/common/model/quantized_load.py` | ✅ 已建 |
| `backend/engine/common/model/base.py` | ✅ 分派 |
| `backend/engine/pipelines/image_model_load.py` | ✅ |
| `backend/engine/pipelines/video_model_load.py` | ✅ |
| `backend/engine/pipelines/audio_model_load.py` | ✅ Phase 2 |
| `backend/engine/families/ltx/transformer_mlx.py` | ✅ Phase 2 |
| `backend/engine/families/ace_step/transformer.py` | ✅ Phase 2 |
| `backend/engine/families/seedvr2/stem_mlx.py` | ✅ Phase 2 |
| `backend/engine/common/model/quantized_lora.py` | ✅ Phase 3 |
| `backend/engine/common/bundle/lora_mlx.py` | ✅ Phase 3 |
| `backend/core/derived_quant_mlx.py` | ✅ Phase 4 转换 |
| `backend/core/derived_quant_layout.py` | ✅ Phase 4 `component_targets` |
| `default_config/locales/zh.json` / `en.json` | ✅ 错误文案 |
| `tests/engine_unit.py` | ✅ `QuantizedInferenceModeTests` |
| `docs/engine_architecture.md` | ✅ §12.6 |

---

## 9. 开放问题

1. 量化 + LoRA **默认允许**（dequantize→merge→requantize）；仅 `quantized_lora: false` fail loud；不静默回退 fp16。
2. fp16 现场 `dit_runtime_quantize_bits` 设置项：未做（可复用 `quantize_runtime`）。
3. registry `size` 仍表示磁盘体积；缓存预算用 `estimate_dit_cache_size_gb()` 推算。

---

## 10. 一句话摘要

**量化 bundle 已是推理格式**：DiT 以 `QuantizedLinear` 常驻；`flux2-klein-4b` 的 derived int4/int8 已声明 `text_encoder` 一并量化；CUDA 仅占位 fail loud。待办：本机显存 / 生成 smoke。
