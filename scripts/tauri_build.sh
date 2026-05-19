#!/usr/bin/env bash
# Back-compat entry: macOS desktop Tauri build.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$ROOT/scripts/tauri_build_macos.sh" "$@"
