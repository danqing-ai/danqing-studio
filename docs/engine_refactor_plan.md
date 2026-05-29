# DanQing Engine 重构 Plan（执行索引）

> **唯一详细来源**：[danqing_architecture_reference.md](danqing_architecture_reference.md) §0–§14。本文只保留 Phase 清单与验收命令，不重复 design 正文。

## 硬约束

Fail loud · 保留 `models_registry.json` · `backend/engine/` 重构净删或持平 · 双平台诚实

## Phase 状态

| Phase | 内容 | 状态 |
|-------|------|------|
| 0 | parity / manifest / profiles 解析 / CI | ✅ |
| 1 | profiles 瘦身、下载组件状态、manifest 写入 | ✅（15 图像 + 10 视频 profile） |
| 2 | common 收敛、scaffold | ✅ `bundle_layout` |
| 3 | ctx 孤岛（flux/qwen/seedvr2 按族 PR） | ✅ seedvr2 stem 重组；flux/qwen/fibo ctx（seedvr2 DiT 仍 `mx.*` 于 `*_mlx`） |
| 4 | Pipeline 节点日志 | ✅ image/video/upscale generate + edit 对齐 |
| 5–6 | DX / 治理文档 | ✅ AGENTS 锚点 + checklist 同步 |

## 验收

```bash
make verify-engine-stack
make sync-models-registry   # 同步 registry 到 workspace
make bench-mflux-case ID=flux2-klein-9b-create
make bench-mflux-case ID=flux1-dev-create
make bench-mflux-case ID=z-image-turbo-create
make bench-sanity-case ID=seedvr2-7b-upscale-sanity
```

## Registry 瘦身

`apply_standard_profile()` 在 `backend/core/registry_profiles.py`；迁移后 expanded 文档不变。

## PR 切片

见 architecture §13：profiles → manifest → parity → flux ctx → qwen → seedvr2 → graph/LoRA → DX
