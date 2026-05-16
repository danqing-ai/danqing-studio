# Scripts

## 桌面打包

| 脚本 | 说明 |
|------|------|
| `build_desktop.sh` | 全流程：前端 → MLX sidecar → Tauri |
| `build_sidecar.py` | PyInstaller → `out/sidecar/danqing-api/` |
| `clean_build.py` | 清理 `out/` 与遗留 `dist/`、`build/` |
| `out_paths.py` | 统一构建路径 |
| `pyinstaller_common.py` | PyInstaller 元数据（MLX / full profile） |
| `pyinstaller_runtime_hook.py` | 冻结运行时钩子（由 common 生成） |
| `pyinstaller_hooks/` | PyInstaller 排除钩子 |

## 代码门禁

| 脚本 | Make 目标 |
|------|-----------|
| `check_consistency.py` | `make check-consistency` |
| `check_engine_backend_imports.py` | `make check-engine-imports` |
| `test_engine_unit.py` | `make test-engine-unit` |
| `make_lint.py` | `make lint` |
