# DanQing Studio — Tauri 2 桌面壳

本目录提供 **Tauri 2** 原生窗口；业务仍由 **FastAPI REST**（PyInstaller 打包的 `danqing-api` sidecar）提供。与「仅通过 HTTP 访问引擎」一致。

## 前置

- **Rust 工具链**（`cargo`、`rustc` 须在 PATH 中）。若未安装：
  - 安装：<https://rustup.rs/>（推荐 `rustup` 默认 profile）
  - 装完后新开终端，或执行：`source "$HOME/.cargo/env"`
  - 验证：`command -v cargo && cargo --version`
- **应用图标**：`src-tauri/icons/icon.png` 须为 **RGBA** PNG（512×512 等）；若打包报 “not RGBA”，可用项目 venv：`python3 -c "from PIL import Image; p='src-tauri/icons/icon.png'; i=Image.open(p).convert('RGBA'); i.save(p)"`（在 `desktop/` 下执行），或 `npm exec tauri icon <源图.png>`。
- 仓库根目录 **Python 3.11 + `.venv`**，已安装依赖与 **PyInstaller**
- **MLX 侧车**：当前 PyInstaller 元数据以 **macOS Apple Silicon + mlx `.dylib`** 为主；在 Windows/Linux 上需自行扩展 `scripts/pyinstaller_common.py` 的二进制列表后再打 sidecar

## 开发（不打包 sidecar）

1. 终端 A：仓库根目录 `./bin/launch.sh`（或 `uvicorn backend.main:app --host 127.0.0.1 --port 7860`）
2. 终端 B：

```bash
cd desktop
npm install
npm run dev
```

`tauri dev` 会打开 WebView 指向 `http://127.0.0.1:7860`（见 `src-tauri/tauri.conf.json` 的 `devUrl`）。

## 发布构建（sidecar + Tauri）

在**仓库根目录**执行（顺序：先 PyInstaller，再 Tauri，以便打入 `dist/danqing-api`）：

```bash
./scripts/build_desktop_bundle.sh
# 或: python scripts/build_sidecar.py && cd desktop && npm install && npm run build
```

或使用 Makefile：`make desktop-bundle`（会先跑 `make desktop-prereqs` 检查 `npm` 与 `cargo`）。

产物（随平台变化）在 `desktop/src-tauri/target/release/bundle/`。

## 运行时环境变量（sidecar）

| 变量 | 说明 |
|------|------|
| `DANQING_HTTP_HOST` | 默认 `0.0.0.0`；Tauri 设为 `127.0.0.1` |
| `DANQING_HTTP_PORT` | 默认 `7860`；Tauri 选空闲端口并注入 |
| `DANQING_USER_DATA_DIR` | 可写数据根（models / outputs / db / config）；Tauri 设为 `app_data_dir()/server-data` |

## Homebrew Cask（建议）

官方仓库不一定收录；自建 tap 中 Cask 的 `url` 指向 CI 产出的 **`.dmg` / `.zip`** 即可，与 PyInstaller+Tauri 产物一致。参见仓库根 `AGENTS.md` 分发说明（若已补充）。
