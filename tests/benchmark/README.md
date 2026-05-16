# Benchmark

两套套件，统一入口 `python -m tests.benchmark`：

| Make | CLI | 说明 |
|------|-----|------|
| `make bench-mflux` | `mflux --all` | 全模型 mflux PSNR/SSIM 对照 |
| `make bench-mflux-case ID=…` | `mflux --case ID` | 单个 mflux 用例 |
| `make bench-sanity` | `sanity --all` | 全模型成片健全性（拒黑/白/平场） |
| `make bench-sanity-case ID=…` | `sanity --case ID` | 单个健全性用例 |

列出用例：`python -m tests.benchmark mflux --list` / `sanity --list`

首次跑 mflux 对照：`make bench-setup`（独立 venv）。rewrite/upscale 需 `make bench-src`。

用例定义：`cases.py`。输出：`tests/benchmark/outputs/`。
