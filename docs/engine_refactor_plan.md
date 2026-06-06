# DanQing Engine 重构 Plan（执行索引）

> **唯一详细来源**：[danqing_architecture_reference.md](danqing_architecture_reference.md) §0–§14。本文只保留 Phase 清单与验收命令，不重复 design 正文。

## 硬约束

Fail loud · 保留 `models_registry.json` · `backend/engine/` 重构净删或持平 · 双平台诚实

## Phase 状态

| Phase | 内容 | 状态 |
|-------|------|------|
| 0 | parity / manifest / profiles 解析 / CI | ✅ |
| 1 | profiles 瘦身、下载组件状态、manifest 写入 | ✅（15 图像 + 10 视频 profile）；audit：`make report-registry-audit` |
| 2 | common 收敛、scaffold | ✅ VAE bootstrap + `build_standard_vae_preview_session` / `create_loaded_vae_decoder` / `vae_forward_to_pil` |
| 3 | ctx 孤岛（flux/qwen/seedvr2 按族 PR） | ✅ seedvr2 stem；flux/qwen/fibo ctx |
| 4 | Pipeline 节点日志 | ✅ image/video/upscale + `validate_bundle_graph_step` |
| 5–6 | DX / 治理文档 | ✅ scaffold 预算提示；report：`make report-family-budget` / `make report-family-reuse` |

## 验收

```bash
make verify-engine-stack
make sync-models-registry   # 同步 registry 到 workspace
make bench-mflux-case ID=flux2-klein-9b-create
make bench-mflux-case ID=flux1-dev-create
make bench-mflux-case ID=z-image-turbo-create
make bench-sanity-case ID=seedvr2-7b-upscale-sanity
make report-registry-audit      # Phase 1 shrink hints (non-blocking)
make report-family-budget       # Phase 6 logical-unit report
make report-family-reuse
```

## Registry 瘦身

`apply_standard_profile()` 在 `backend/core/registry_profiles.py`；迁移后 expanded 文档不变。

## PR 切片

见 architecture §13：profiles → manifest → parity → flux ctx → qwen → seedvr2 → graph/LoRA → DX
