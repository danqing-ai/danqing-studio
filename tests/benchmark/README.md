# Benchmark

三套套件，统一入口 `python -m tests.benchmark`：

| Make | CLI | 说明 |
|------|-----|------|
| `make bench-mflux` | `mflux --all` | 全模型 mflux PSNR/SSIM 对照 |
| `make bench-mflux-case ID=…` | `mflux --case ID` | 单个 mflux 用例 |
| `make bench-diffusers` | `diffusers --all` | 图像 diffusers 对照（仅无 mflux 对位时注册；当前可能 0 条） |
| `make bench-diffusers-case ID=…` | `diffusers --case ID` | 单个 diffusers 对比用例 |
| `make bench-sanity` | `sanity --all` | 全模型成片健全性（反噪声/反平场 + 质量评分） |
| `make bench-sanity-case ID=…` | `sanity --case ID` | 单个健全性用例 |
| `make bench-sanity-case ID=ace-step-xl-sft-sanity` | 同上 | ACE-Step MLX 10s 音频 RMS 健全性（需 `models/Audio/acestep-v15-xl-sft`） |
| `make bench-sanity-case ID=ace-step-xl-sft-inspiration-lm` | 同上 | 短描述 + 空歌词 + planner/codes（llm_dit） |
| `make bench-sanity-case ID=ace-step-xl-sft-cover-sanity` | 同上 | cover 编辑（`danqing-audio-edit` + fixture WAV） |
| `make bench-audio-sanity-ace-step` | 快捷目标 | ACE-Step 四条 sanity 串联 |
| `make bench-audio-sanity` | 快捷目标 | ACE-Step（含 inspiration + cover）各跑一条 10s 健全性 |
| `make bench-audio-sanity-lm` | 快捷目标 | ACE-Step 含 5Hz LM 扩写 |
| `make bench-wan-sanity` | `sanity --case wan-2.2-ti2v-5b-sanity` | Wan 5B 快速视频健全性（4 步、17 帧） |
| `make bench-wan-baseline` | `sanity --case wan-2.2-ti2v-5b-baseline` | Wan 5B 耗时基线（8 步、81 帧，打印 `[BASELINE] total_sec`） |
| `make bench-sanity-case ID=ace-step-xl-sft-cuda-sanity` | 同上 | CUDA 路径（无 GPU 时 SKIP） |

列出用例：`python -m tests.benchmark mflux --list` / `diffusers --list` / `sanity --list`

首次跑 mflux 对照：`make bench-setup`（独立 venv）。rewrite/upscale 需 `make bench-src`。

用例定义：`cases.py`。输出：`tests/benchmark/outputs/`。

参考对比说明（对齐 mflux 模式）：
- 图像 PSNR 主对照：`make bench-mflux`（`ALL_CASES`）。`diffusers` 子命令仅收录 **没有** 对应 mflux 用例的模型；已有 mflux 的（如 z-image-turbo）不在此套件重复。
- `--list-runnable` 会按本地 bundle 过滤可跑用例：`python -m tests.benchmark diffusers --list-runnable`

质量门禁可在 `SanityCase` 里按模型覆写：`image_quality_thresholds` / `audio_quality_thresholds` /
`video_quality_thresholds`。推荐先复用 `cases.py` 顶部模板（如 `IMAGE_THRESHOLDS_REWRITE`、`AUDIO_THRESHOLDS_ACE_STEP`、
`VIDEO_THRESHOLDS_WAN`）再微调单 case。可选语义门禁：`semantic_gate_enabled=True`（image/video 默认 CLIP，audio 默认 CLAP）。

推荐语义门禁 profile（环境变量）：
- `DANQING_BENCH_SEMANTIC_PROFILE=off`（默认）：关闭语义门禁
- `DANQING_BENCH_SEMANTIC_PROFILE=core`：开启 image/video 语义门禁（推荐先用）
- `DANQING_BENCH_SEMANTIC_PROFILE=all`：再额外开启 audio(CLAP) 语义门禁
