#!/usr/bin/env bash
# Full desktop package: Vite UI -> PyInstaller sidecar -> Tauri bundle.
# Run from repository root. Outputs under ``out/`` (see scripts/out_paths.py).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v npm >/dev/null 2>&1; then
  echo "npm not found; install Node.js" >&2
  exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo not found. Install Rust from https://rustup.rs/" >&2
  exit 1
fi

PYTHON="${PYTHON:-python3}"
if [[ -x "$ROOT/.venv/bin/python3" ]]; then
  PYTHON="$ROOT/.venv/bin/python3"
fi

echo "==> Frontend -> out/frontend/dist"
cd "$ROOT/frontend"
npm install
npm run build

echo "==> pack-macos-desktop (MLX sidecar + Tauri)"
cd "$ROOT"
export DANQING_PYINSTALLER_PROFILE="${DANQING_PYINSTALLER_PROFILE:-mlx}"
make pack-macos-desktop

echo "Done."
echo "  Sidecar:  $ROOT/out/sidecar/danqing-api"
echo "  Desktop:  $ROOT/out/desktop/bundle/"
