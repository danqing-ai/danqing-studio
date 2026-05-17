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

# Stale DMG mounts / rw.* temps break create-dmg; AppleScript layout fails on macOS 15+.
cleanup_dmg_artifacts() {
  local bundle_root="$CARGO_TARGET_DIR/aarch64-apple-darwin/release/bundle"
  [[ -d "$bundle_root" ]] || return 0
  find "$bundle_root" -maxdepth 3 -name 'rw.*.dmg' -delete 2>/dev/null || true
  while IFS= read -r dmg; do
    [[ -n "$dmg" ]] && hdiutil detach "$dmg" -force 2>/dev/null || true
  done < <(
    hdiutil info 2>/dev/null | awk -v root="$bundle_root" '
      /^image-path/ && index($0, root) {
        sub(/^image-path[[:space:]]*:[[:space:]]*/, "")
        print
      }
    '
  )
}

echo "==> Tauri build (aarch64-apple-darwin)"
echo "    CARGO_TARGET_DIR=$CARGO_TARGET_DIR"
cleanup_dmg_artifacts
npm install
# Tauri passes --skip-jenkins to bundle_dmg.sh when CI=true (no Finder AppleScript).
export CI=true
npm exec tauri build -- --target aarch64-apple-darwin
cleanup_dmg_artifacts

"$PYTHON" "$ROOT/scripts/stage_desktop_bundle.py"

# Ad-hoc sign .app (CI/GitHub DMG is unsigned; without this macOS shows「已损坏，无法打开」).
if APP=$(find "$ROOT/out/desktop/bundle" -name "*.app" -print -quit); then
  echo "==> codesign .app (ad-hoc)"
  SC_DIR="$APP/Contents/Resources/danqing-api"
  if [[ -d "$SC_DIR" ]]; then
    while IFS= read -r -d '' bin; do
      codesign -s - --force "$bin" 2>/dev/null || true
    done < <(find "$SC_DIR" -type f \( -perm -111 -o -name '*.dylib' -o -name '*.so' \) -print0 2>/dev/null)
  else
    echo "Warning: sidecar not at $SC_DIR (check prepare_tauri_resources.sh)" >&2
  fi
  if [[ -d "$APP/Contents/MacOS" ]]; then
    for bin in "$APP/Contents/MacOS"/*; do
      [[ -f "$bin" ]] && codesign -s - --force "$bin" || true
    done
  fi
  codesign -s - --force --deep "$APP"
  codesign --verify --deep --strict "$APP" 2>/dev/null || \
    echo "Warning: codesign verify reported issues (app may still run after clearing quarantine)" >&2
fi
