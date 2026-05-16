#!/usr/bin/env bash
# Stage PyInstaller sidecar into src-tauri/resources/ (Tauri cannot use ../../ paths cleanly).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/out/sidecar/danqing-api"
DST="$ROOT/desktop/src-tauri/danqing-api"

if [[ ! -f "$SRC/danqing-api" ]]; then
  echo "Missing sidecar at $SRC" >&2
  echo "Run: make desktop-sidecar" >&2
  exit 1
fi

echo "==> Stage sidecar for Tauri: src-tauri/danqing-api"
rm -rf "$DST"
mkdir -p "$(dirname "$DST")"
rsync -a --delete "$SRC/" "$DST/"
