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
| `check_engine_family_primitives.py` | `make check-engine-family-primitives` |
| `check_engine_attention_paths.py` | `make check-engine-attention-paths` |
| `check_engine_sdpa_paths.py` | `make check-engine-sdpa-paths`（禁止 families 直呼 MLX/Torch SDPA） |
| `check_engine_rope_paths.py` | `make check-engine-rope-paths` |
| `check_engine_modulation_paths.py` | `make check-engine-modulation-paths` |
| `check_models_registry_contracts.py` | `make check-models-registry-contracts` |
| `check-engine-governance`（Make 聚合目标） | 一次执行全部 engine 架构门禁 + `check-consistency` |
| `verify-engine-stack`（Make 聚合目标） | 一次执行 `check-engine-governance` + `test-engine-unit` |
| `test_engine_unit.py` | `make test-engine-unit` |
| `make_lint.py` | `make lint` |
