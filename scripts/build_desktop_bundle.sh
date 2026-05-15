#!/usr/bin/env bash
# Build PyInstaller sidecar then Tauri desktop bundle (run from repo root).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> PyInstaller sidecar (dist/danqing-api)"
python3 scripts/build_sidecar.py

if ! command -v npm >/dev/null 2>&1; then
  echo "npm not found; install Node.js or run from desktop/: npm install && npm run build" >&2
  exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo not found. Install Rust from https://rustup.rs/ and ensure ~/.cargo/bin is on PATH (e.g. source \"\$HOME/.cargo/env\")." >&2
  exit 1
fi

echo "==> Tauri bundle"
cd "$ROOT/desktop"
npm install
npm run build

echo "Done. See desktop/src-tauri/target/release/bundle/"
