#!/usr/bin/env bash
# macOS Apple Silicon (aarch64) Tauri release build. Cargo artifacts -> out/desktop/cargo/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/desktop"

if [[ "$(uname -s)" != Darwin ]]; then
  echo "DanQing desktop bundle is macOS-only (MLX sidecar)." >&2
  exit 1
fi
if [[ "$(uname -m)" != arm64 ]]; then
  echo "DanQing desktop requires Apple Silicon (arm64). MLX is not supported on Intel Mac." >&2
  exit 1
fi

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/out/desktop/cargo}"
mkdir -p "$CARGO_TARGET_DIR"

PYTHON="${PYTHON:-python3}"
if [[ -x "$ROOT/.venv/bin/python3" ]]; then
  PYTHON="$ROOT/.venv/bin/python3"
fi

if [[ -n "${DANQING_DESKTOP_VERSION:-}" ]]; then
  "$PYTHON" "$ROOT/scripts/set_desktop_version.py" "$DANQING_DESKTOP_VERSION"
elif git -C "$ROOT" describe --exact-match --tags HEAD >/dev/null 2>&1; then
  echo "==> Desktop version from git tag"
  "$PYTHON" "$ROOT/scripts/set_desktop_version.py"
fi

"$ROOT/scripts/prepare_tauri_resources.sh"

if ! rustup target list --installed | grep -q '^aarch64-apple-darwin$'; then
  echo "==> rustup target add aarch64-apple-darwin"
  rustup target add aarch64-apple-darwin
fi

echo "==> Tauri build (aarch64-apple-darwin)"
echo "    CARGO_TARGET_DIR=$CARGO_TARGET_DIR"
npm install
npm exec tauri build -- --target aarch64-apple-darwin

"$PYTHON" "$ROOT/scripts/stage_desktop_bundle.py"

# Ad-hoc sign embedded sidecar so Gatekeeper allows spawn from the .app
if APP=$(find "$ROOT/out/desktop/bundle" -name "*.app" -print -quit); then
  SC_DIR="$APP/Contents/Resources/danqing-api"
  if [[ -f "$SC_DIR/danqing-api" ]]; then
    echo "==> codesign sidecar in .app"
    for f in danqing-api libmlx.dylib libjaccl.dylib; do
      [[ -f "$SC_DIR/$f" ]] && codesign -s - --force "$SC_DIR/$f" || true
    done
  else
    echo "Warning: sidecar not at $SC_DIR/danqing-api (check prepare_tauri_resources.sh)" >&2
  fi
fi
