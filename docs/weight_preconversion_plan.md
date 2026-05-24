# 权重预转换方案 — DanQing 专用格式（MLX 平台）

**Status**: 设计稿，待复核后实施  
**Author**: Claude Code (基于 mlx-examples 权重转换模式调研)  
**Date**: 2026-05-24

---

## 1. 问题

当前每次推理都执行重复的权重转换工作：

```
从磁盘加载 safetensors
  → 改 key name（字符串替换）
  → 改 tensor shape（reshape/transpose）
  → 加载进模型
```

这些工作在 `ImagePipeline._load_model()`、`VideoPipeline._load_model()`、
VAE decode、Flux1 T5/CLIP encoder 等 6+ 个位置重复执行，分散在：

- `_transformer_registry.py` — `_WEIGHT_REMAP` / `_VIDEO_WEIGHT_REMAP` / `_AUDIO_WEIGHT_REMAP`
- `image_pipeline.py:_load_model()` — `remap_fn(w)` 调用
- `video_pipeline.py:_load_model()` — 同上 + mlx-forge 检测 + restore
- `video_bundle_layout.py` — `looks_like_mlx_forge_ltx()` + `restore_diffusers_names_from_mlx_forge_ltx()`
- `flux1/t5_encoder_mlx.py` — `remap_flux1_t5_weights()`
- `flux1/clip_encoder_mlx.py` — `remap_flux1_clip_weights()` + `nest_flux1_clip_weights()`
- image+video VAE decode — `remap_vae_weights()`（2 处）

**总运行时逻辑约 115 行**，每次加载都执行。

---

## 2. 目标

**安装时一次性完成权重转换**，推理时零 remap、零 transpose：

```
安装/首次加载: 原始 diffusers 权重 → WeightConverter → DanQing 格式 → 写入磁盘
推理:         从磁盘加载 → 直接 model.load_weights()（不做任何转换）
```

---

## 3. 约束

1. **单平台运行**：DanQing 安装要么在 MLX Mac 上，要么在 CUDA Linux 上，**不会跨平台共享模型目录**。
2. **MLX 专用格式**：预转换后的权重使用 MLX NHWC layout（Conv2d: `[O,H,W,I]`，Conv3d: `[O,T,H,W,I]`），key name 对齐 DanQing 模块命名。
3. **不删除 remap 函数定义**：`families/*/weights.py` 中的 `remap_*_weights()` 函数保留，供 `WeightConverter` 复用。
4. **向后兼容**：旧模型首次加载时自动懒转换，用户无感知。
5. **LoRA 保持现状**：本次不涉及 LoRA 预转换。
6. **单一路径**：不引入"运行时 remap"和"预转换"并存的双路径，全部走预转换。

---

## 4. 预转换内容

### 4.1 Key Name Remap（所有组件）

将原始 diffusers/PyTorch 命名替换为 DanQing 模块命名：

| 原始 key | DanQing key |
|---------|------------|
| `transformer_blocks.0.attn.to_q.weight` | `double_blocks.0.img_attn.qkv.weight` |
| `x_embedder.proj.weight` | `patch_embed.proj.weight` |
| `decoder.mid_block.resnets.0.norm1.weight` | `mid_resnet1.norm1.weight` |

**实现**：复用现有的 `remap_*_weights()` 函数（字符串替换 + regex）。

### 4.2 Tensor Layout 转换（Conv weight 专属）

| 类型 | 原始 layout | 转换后 layout |
|------|-----------|-------------|
| Conv2d weight | `[O, I, H, W]` (PyTorch NCHW) | `[O, H, W, I]` (MLX NHWC) |
| Conv3d weight | `[O, I, T, H, W]` (PyTorch NCDHW) | `[O, T, H, W, I]` (MLX NHWDC) |
| Linear weight | `[O, I]` | 不变 |
| Bias/Norm | 1D | 不变 |

**实现**：在 WeightConverter 中统一处理（检测 `key.endswith(".weight")` + `tensor.ndim in (4, 5)`）。

### 4.3 Tensor Shape 转换（特殊层）

| 层 | 原始 shape | 转换后 shape |
|---|-----------|-------------|
| Flux1 patch_embed | Linear `[out, in]` | Conv2d `[out, 1, 1, in]` |
| LTX proj_in | Linear `[out, in]` | Conv3d `[out, 1, 1, 1, in]` |

**实现**：复用现有的 `_reshape_patch_embed_weight()`、`_reshape_proj_in_weight()`。

---

## 5. 转换范围

### 5.1 必须转换的组件

| 组件 | family | 文件 | 复杂度 | 备注 |
|------|--------|------|--------|------|
| **Image DiT** | flux1 | `families/flux1/weights.py` | 中 | key 替换多，含 patch_embed reshape |
| | flux2 | `families/flux2/weights.py` | 中 | |
| | z_image | `families/z_image/weights.py` | 低 | |
| | qwen_image | `families/qwen/weights_mlx.py` | 高 | WeightMapper 1100+ 行，但 remap 函数可复用 |
| | fibo | 待查 | | |
| **Video DiT** | wan | `families/wan/weights.py` | 低 | 含 diffusers/ori 双格式检测 |
| | ltx | `families/ltx/weights.py` | 低 | 含 mlx-forge restore |
| | cogvideox | `families/cogvideox/weights.py` | 低 | |
| | hunyuan | `families/hunyuan/weights.py` | 低 | Conv3d transpose |
| **Audio DiT** | ace_step | `families/ace_step/weights.py` | 待查 | |
| **VAE** | 通用 | `common/vae/weight_remap.py` | 低 | decoder + encoder |
| **Flux1 Text Encoder** | flux1 | `families/flux1/weights.py` | 低 | T5 + CLIP 自定义 encoder |

### 5.2 不转换的组件

- **标准 T5/CLIP**（`mlx_lm` / `transformers` 自带 `load_weights()`，无需 remap）
- **LoRA**（保持现状，动态加载时处理）
- **已量化的 int4/int8 权重**（`convert_model()` 已输出 DanQing 格式）

---

## 6. 核心组件设计

### 6.1 WeightConverter

```python
# backend/engine/weight_converter.py
from pathlib import Path
from typing import Any, Callable
import mlx.core as mx


class WeightConverter:
    """安装时权重预转换 — key remap + NHWC layout + 标记写入"""

    # family → (模块路径, remap函数名)
    _TRANSFORMER_REMAP = {
        "flux1":      ("backend.engine.families.flux1.weights",     "remap_flux1_weights"),
        "flux2":      ("backend.engine.families.flux2.weights",     "remap_flux2_weights"),
        "z_image":    ("backend.engine.families.z_image.weights",   "remap_zimage_weights"),
        "qwen_image": ("backend.engine.families.qwen.weights_mlx",  "remap_qwen_transformer_weights"),
        # ... video/audio families
    }

    def convert_bundle(self, bundle_root: Path, family: str) -> None:
        """转换 bundle 内所有组件，写入 .danqing_converted 标记"""
        if self._is_converted(bundle_root):
            return

        # 1. Transformer DiT
        self._convert_transformer(bundle_root, family)

        # 2. VAE
        vae_dir = bundle_root / "vae"
        if vae_dir.exists():
            self._convert_vae(vae_dir)

        # 3. Flux1 自定义 encoder
        if family == "flux1":
            self._convert_flux1_encoders(bundle_root)

        self._write_marker(bundle_root, family)

    def _convert_transformer(self, bundle_root: Path, family: str) -> None:
        """DiT 权重：读取 → remap → NHWC transpose → 覆盖保存"""
        src_dir = bundle_root / "transformer"
        if not src_dir.exists():
            return

        remap_fn = self._get_remap_fn(family)
        if remap_fn is None:
            return

        # 读取所有 shard
        weights: dict[str, Any] = {}
        for sf in sorted(src_dir.glob("*.safetensors")):
            weights.update(dict(mx.load(str(sf))))

        if not weights:
            return

        # Remap key name
        weights = remap_fn(weights)

        # 处理 mlx-forge LTX（转换前先 restore）
        if family == "ltx":
            from backend.engine.families.ltx.weights import (
                restore_diffusers_names_from_mlx_forge_ltx,
                looks_like_mlx_forge_ltx_transformer_keys,
            )
            if looks_like_mlx_forge_ltx_transformer_keys(weights):
                weights = restore_diffusers_names_from_mlx_forge_ltx(weights)
                weights = remap_fn(weights)

        # NHWC transpose（Conv weight）
        weights = self._apply_nhwc_transpose(weights)

        # 分片保存（max 5GB/shard，复用 mlx-examples 模式）
        self._save_sharded(weights, src_dir, max_gb=5)

    def _convert_vae(self, vae_dir: Path) -> None:
        """VAE 权重：remap + transpose → 覆盖保存"""
        from backend.engine.common.vae import remap_vae_weights

        weights: dict[str, Any] = {}
        for sf in sorted(vae_dir.glob("*.safetensors")):
            weights.update(dict(mx.load(str(sf))))

        if not weights:
            return

        weights = remap_vae_weights(weights)
        weights = self._apply_nhwc_transpose(weights)
        self._save_sharded(weights, vae_dir, max_gb=5)

    def _apply_nhwc_transpose(self, weights: dict[str, Any]) -> dict[str, Any]:
        """Conv weight NCHW → NHWC"""
        for k, v in weights.items():
            if not k.endswith(".weight"):
                continue
            if v.ndim == 4:
                weights[k] = v.transpose(0, 2, 3, 1)
            elif v.ndim == 5:
                weights[k] = v.transpose(0, 2, 3, 4, 1)
        return weights

    def _save_sharded(
        self, weights: dict[str, Any], target_dir: Path, max_gb: int = 5
    ) -> None:
        """分片保存，覆盖原始文件"""
        max_bytes = max_gb << 30
        shards: list[dict[str, Any]] = []
        current: dict[str, Any] = {}
        current_size = 0

        for k, v in weights.items():
            if current_size + v.nbytes > max_bytes and current:
                shards.append(current)
                current = {}
                current_size = 0
            current[k] = v
            current_size += v.nbytes

        if current:
            shards.append(current)

        # 删除旧文件
        for old in target_dir.glob("*.safetensors"):
            old.unlink()

        # 写入新文件
        if len(shards) == 1:
            mx.save_safetensors(str(target_dir / "model.safetensors"), shards[0])
        else:
            weight_map: dict[str, str] = {}
            for i, shard in enumerate(shards):
                name = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
                mx.save_safetensors(str(target_dir / name), shard)
                for k in shard:
                    weight_map[k] = name

            # 写 index.json
            import json
            index = {"metadata": {}, "weight_map": weight_map}
            with open(target_dir / "model.safetensors.index.json", "w") as f:
                json.dump(index, f, indent=2)

    def _get_remap_fn(self, family: str) -> Callable | None:
        """动态导入 remap 函数"""
        entry = self._TRANSFORMER_REMAP.get(family)
        if entry is None:
            return None
        import importlib
        mod = importlib.import_module(entry[0])
        return getattr(mod, entry[1], None)

    def _is_converted(self, bundle_root: Path) -> bool:
        marker = bundle_root / ".danqing_converted"
        if not marker.exists():
            return False
        # 可扩展：读取 marker 内容，检查 family/version 是否匹配
        return True

    def _write_marker(self, bundle_root: Path, family: str) -> None:
        import json
        marker = bundle_root / ".danqing_converted"
        data = {
            "family": family,
            "version": "1.0",
            "converted_at": str(datetime.now()),
        }
        with open(marker, "w") as f:
            json.dump(data, f, indent=2)
```

### 6.2 向后兼容：懒转换

```python
# backend/engine/pipelines/_base.py 或混入 ImagePipeline/VideoPipeline

class ConversionMixin:
    def _ensure_converted(self, bundle_root: Path | None, family: str) -> None:
        """确保权重已预转换，未转换则自动执行"""
        if bundle_root is None:
            return
        marker = bundle_root / ".danqing_converted"
        if marker.exists():
            return

        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Auto-converting {family} weights to DanQing format...")

        from backend.engine.weight_converter import WeightConverter
        converter = WeightConverter()
        converter.convert_bundle(bundle_root, family)
```

### 6.3 Pipeline 改造

**ImagePipeline._load_model()**

```python
def _load_model(self, family: str, config, entry, version_key: str | None):
    trans_cls = _get_transformer_class(family)
    model = trans_cls(config, self.ctx)

    bundle_root = self._local_bundle_root(entry, version_key)
    tp = (bundle_root / "transformer") if bundle_root else None
    if tp is None or not tp.exists():
        return None

    # 确保已转换（向后兼容）
    self._ensure_converted(bundle_root, family)

    w = {}
    for sf in sorted(tp.glob("*.safetensors")):
        w.update(self.ctx.load_weights(str(sf)))
    # ❌ 删除: if remap_fn: w = remap_fn(w)

    from backend.engine.common.safetensors_affine_quant import read_bundle_affine_bits_if_quantized
    bundle_affine_bits = read_bundle_affine_bits_if_quantized(w, tp)

    model.load_weights(
        list(w.items()),
        strict=False,
        ctx=self.ctx,
        bundle_affine_bits=bundle_affine_bits,
    )
    self.ctx.eval(*[p for _, p in model.parameters()])
    return model
```

**VAE decode**

```python
# 移除: decoder_w = remap_vae_weights(vae_weights)
# 直接: vae.load_weights(list(vae_weights.items()), strict=False)
```

**Flux1 T5Encoder**

```python
# 移除: remapped = remap_flux1_t5_weights(weights)
# 改为: for key, param in self._param_map.items(): param[:] = weights[key]
```

---

## 7. 删除清单（运行时 remap 逻辑）

| 文件 | 删除内容 |
|------|---------|
| `_transformer_registry.py` | `_WEIGHT_REMAP` / `_VIDEO_WEIGHT_REMAP` / `_AUDIO_WEIGHT_REMAP` 注册表；`get_weight_remap()` / `get_video_weight_remap()` / `get_audio_weight_remap()` 函数 |
| `image_pipeline.py` | `remap_fn = _get_weight_remap(family)`；`if remap_fn: w = remap_fn(w)` |
| `video_pipeline.py` | 同上；`looks_like_mlx_forge_ltx(w)` 检测；`restore_diffusers_names_from_mlx_forge_ltx(w)` 调用 |
| `video_bundle_layout.py` | `looks_like_mlx_forge_ltx_transformer_keys()`；`restore_diffusers_names_from_mlx_forge_ltx()`（可选保留供 converter 复用，但运行时不再调用） |
| `flux1/t5_encoder_mlx.py` | `remap_flux1_t5_weights()` 调用 |
| `flux1/clip_encoder_mlx.py` | `remap_flux1_clip_weights()` + `nest_flux1_clip_weights()` 调用 |
| image+video VAE decode | `remap_vae_weights()` 调用（2 处） |

**保留的文件**（供 converter 复用）：
- `families/*/weights.py` — `remap_*_weights()` 函数定义
- `common/vae/weight_remap.py` — `remap_vae_weights()` 函数定义

---

## 8. 新增清单

| 文件 | 内容 | 行数估计 |
|------|------|---------|
| `backend/engine/weight_converter.py` | `WeightConverter` 类 | ~200 |
| `bin/danqing-convert` | CLI 入口 | ~50 |
| `backend/cli/convert_cmd.py` | CLI 逻辑（如有 backend/cli 模式） | ~80 |

---

## 9. CLI 设计

```bash
# 转换已安装模型
bin/danqing-convert --model flux1-dev

# 强制重新转换（修复损坏）
bin/danqing-convert --model flux1-dev --force

# 转换所有已安装模型
bin/danqing-convert --all

# 查看转换状态
bin/danqing-convert --status --model flux1-dev
# 输出: flux1-dev:original ✅ (converted_at: 2026-05-24)
```

---

## 10. 实施顺序

### Phase 1: PoC — flux1（1-2 天）

1. 新建 `backend/engine/weight_converter.py`
2. 实现 `convert_bundle()` → `convert_transformer()`，复用 `remap_flux1_weights()`
3. 改造 `ImagePipeline._load_model()`，移除 flux1 的 remap 调用
4. 添加 `_ensure_converted()` 懒转换
5. 运行 `make bench-sanity-case ID=flux1` 验证输出一致性

### Phase 2: 推广 Image family（1-2 天）

z_image、flux2、qwen_image、fibo 逐个接入 WeightConverter。

### Phase 3: Video family（2-3 天）

wan、ltx（注意 mlx-forge restore 链式调用）、cogvideox、hunyuan。

### Phase 4: VAE + Flux1 Encoder（1-2 天）

通用 VAE `convert_vae()` + Flux1 T5/CLIP encoder 改造。

### Phase 5: 下载集成 + CLI（1 天）

- `DownloadService._finalize_version_install()` 安装后自动调用 `WeightConverter`
- 新增 `bin/danqing-convert`

### Phase 6: 清理 + 验证（2-3 天）

1. 删除 `_transformer_registry.py` 中的 `_WEIGHT_REMAP` 等注册表
2. 删除 Pipeline 中的运行时 remap 调用
3. `make bench-sanity` 全量验证
4. `make bench-sanity-case` video 验证
5. `make verify-engine-stack`

---

## 11. 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| **qwen_image WeightMapper 复杂度** | 1100+ 行映射逻辑，转换时性能或内存问题 | PoC 阶段单独测试，必要时增量转换 |
| **mlx-forge LTX 格式** | 转换前需先 restore 再 remap，链式调用易错 | 在 converter 中封装为原子操作，保持与运行时相同的顺序 |
| **大模型分片** | hunyuan ~23GB，单文件内存压力大 | 复用 `make_shards()` 模式，max 5GB/shard |
| **量化权重兼容性** | int4/int8 已经是 DanQing 格式 | 检测 `.scales` 键，跳过转换 |
| **用户手动替换权重** | 用户可能覆盖已转换的权重文件 | marker 文件检测 + `--force` 重新转换 |
| **转换中断** | 转换过程 crash 导致部分文件写入 | 原子写入：先写到临时目录，完成后再 rename |

---

## 12. 关键决策记录

1. **方案选择**：只做 key remap + NHWC layout（做法 2），不改 dtype 或量化。量化通过现有 `convert_model()` 单独处理。
2. **MLX 专用**：权重格式为 MLX NHWC，不兼容 CUDA。基于"软件要么 MLX 要么 CUDA"的约束。
3. **不保留原始文件**：转换后覆盖原始 `.safetensors`，写入 `.danqing_converted` 标记。不浪费磁盘。
4. **LoRA 保持现状**：本次不涉及，未来可扩展。
5. **运行时零 remap**：转换完成后，Pipeline 中彻底删除 `remap_fn` 调用，不保留双路径。

---

## 13. 收益总结

| 维度 | 变化 |
|------|------|
| **推理代码** | `_load_model()` 从 ~25 行 → ~8 行，零 family 分支 |
| **新增 family** | 不需要改 Pipeline，只需在 converter 注册 remap 函数 |
| **运行时逻辑** | 删除 ~115 行分散的 remap 调用 |
| **新增代码** | `WeightConverter` ~200 行（集中一处） |
| **净代码量** | 减少 ~80 行运行时逻辑 |
| **排查便利** | 磁盘上的权重文件 = 模型直接可用的格式，key 即参数名 |
| **推理性能** | 微提升（省去每次加载的 O(n) remap，对大模型约省 1-3 秒） |
