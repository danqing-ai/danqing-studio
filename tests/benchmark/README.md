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
python -m tests.benchmark --keep-judge-loaded   # 全矩阵时保留 PickScore（默认每 case 后卸载）
```

## 设计

| 层 | 内容 |
|----|------|
| **L0** | CLI 生成（无 bundle / 下载未完成 → **SKIP**） |
| **L1** | 可读、期望分辨率、非退化（方差） |
| **L2** | PickScore；`max(judge_floor, golden×0.85)` |

- **Prompt Pack**：`prompts.json`（P1–P5 create，E1–E2 edit，U1 upscale）
- **Fixtures**：384² 语义图 `fixtures/edit_source.png`（edit / extend）、`fixtures/upscale_source.png`（upscale）、`fixtures/edit_mask.png`（retouch）
- **Extend L1**：源图 384×384，`extend` 方向 `right`、外扩 256px → 期望 **640×384**
- **Edit judge**：仅 **编辑指令**（fixture 已承载场景；PickScore 对「场景+指令」会系统性偏低）
- **`judge_floor`**：`prompts.json` 条目可选下限（如 E1 rewrite、upscale 为 `0.17`），覆盖 golden×0.85 过严的情况
- **Bundle 就绪**：`FamilyBundleContract` + 无 `.incomplete` / 空 safetensors

## 内存与进程模型

- **生成**：每个 case 独立子进程调用 `bin/danqing-*`；子进程退出时 `engine_session` 卸载 ModelCache + runtime（与单次 CLI 一致）
- **MLX 上限**：子进程默认 `DANQING_MLX_MEMORY_LIMIT_GB=64`（可用 `DANQING_BENCH_MLX_MEMORY_GB` 覆盖，16–120）
- **Judge**：默认 **每 case 后** `reset_judge_cache()` 释放 torch PickScore；长矩阵可加 `--keep-judge-loaded` 省加载时间（父进程峰值更高）
- **图像 TE**：encode 完成后按 `release_text_encoder_after_encode`（默认开）释放文本编码器权重再加载 DiT，对齐视频 T5 `release_t5_after_encode`

## 依赖

- 生成：项目 `.venv`（MLX）
- 评分：`tests/benchmark/venv`（torch + transformers<5 + modelscope）
