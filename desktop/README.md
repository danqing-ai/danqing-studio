# DanQing Studio — Tauri 2 桌面壳

本目录提供 **Tauri 2** 原生窗口；业务仍由 **FastAPI REST**（PyInstaller 打包的 `danqing-api` sidecar）提供。

## 构建产物目录

所有打包产物在仓库根目录 **`out/`** 下（见 `scripts/out_paths.py`）：

| 路径 | 内容 |
|------|------|
| `out/frontend/dist/` | Vite 生产构建 |
| `out/sidecar/danqing-api/` | PyInstaller sidecar |
| `out/desktop/bundle/` | `.app` / `.dmg`（发布产物） |
| `out/desktop/cargo/` | Cargo 中间产物（可清理） |

清理：`make clean` 或 `python scripts/clean_build.py`

## 前置

- **Rust**（`cargo` 在 PATH）
- 仓库根 **Python 3.11 + `.venv`**，已安装 **PyInstaller**
- **Node.js**（`npm`）

## 开发（不打包 sidecar）

1. 终端 A：仓库根 `./bin/launch.sh`
2. 终端 B：`cd desktop && npm install && npm run dev`

## 发布构建

```bash
make desktop-bundle
# 或: ./scripts/build_desktop.sh
```

顺序：`out/frontend/dist` → `out/sidecar/danqing-api` → Tauri bundle。

macOS 默认 **MLX 精简 sidecar**（无 torch / `*_cuda`）。完整 CUDA 包：

```bash
make desktop-bundle DANQING_PYINSTALLER_PROFILE=full
```

## 运行时环境变量（sidecar）

| 变量 | 说明 |
|------|------|
| `DANQING_HTTP_HOST` | 默认 `0.0.0.0`；Tauri 设为 `127.0.0.1` |
| `DANQING_HTTP_PORT` | Tauri 选空闲端口并注入 |
| `DANQING_USER_DATA_DIR` | 可写数据根（models / outputs / db / config） |
