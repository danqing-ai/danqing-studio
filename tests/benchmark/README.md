# Image Eval Benchmark

统一图像模型评测：**L1 完整性** + **L2 PickScore**。

## 快速开始（推荐魔塔下载 judge）

```bash
make bench-setup
make bench-download-judge    # 魔塔 AI-ModelScope/PickScore_v1 → workspace/models/Benchmark/pickscore-v1
make bench-eval-smoke
```

## Judge 来源（优先级）

1. `DANQING_BENCH_JUDGE_MODEL=/path/to/pickscore` — 显式本地目录
2. `{workspace}/models/Benchmark/pickscore-v1/` — `make bench-download-judge` 写入
3. `DANQING_BENCH_JUDGE_SOURCE=modelscope`（**默认**）— 评测时自动 `snapshot_download`
4. `DANQING_BENCH_JUDGE_SOURCE=huggingface` — 走 HF（国内通常较慢）

魔塔模型页：https://modelscope.cn/models/AI-ModelScope/PickScore_v1

## 命令

```bash
make bench-eval-smoke          # P1 create + E2 edit（CI 友好）
make bench-eval                # full prompt matrix
make bench-eval-case ID=flux2-klein-9b:P1:create
make bench-eval-case ID=z-image-turbo:P1:create
make bench-eval-case ID=flux2-klein-9b:E2:rewrite
make bench-eval-calibrate      # 写入 golden PickScore 基线
python -m tests.benchmark download-judge --force
```

## 设计

| 层 | 内容 |
|----|------|
| **L0** | CLI 生成（无 bundle / 下载未完成 → **SKIP**） |
| **L1** | 可读、期望分辨率、非退化（方差） |
| **L2** | PickScore；`max(JUDGE_FLOOR, golden×0.85)` |

- **Prompt Pack**：`prompts.json`（P1–P5 create，E1–E2 edit，U1 upscale）
- **Fixtures**：384² 语义图 `fixtures/edit_source.png`（edit）、`fixtures/upscale_source.png`（upscale）
- **Edit judge**：`场景描述 + 编辑指令`（见 `prompts.json` → `fixtures.edit_scene`）
- **Bundle 就绪**：`FamilyBundleContract` + 无 `.incomplete` / 空 safetensors

## 依赖

- 生成：项目 `.venv`（MLX）
- 评分：`tests/benchmark/venv`（torch + transformers<5 + modelscope）
