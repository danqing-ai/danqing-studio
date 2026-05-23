.PHONY: help clean lint dev start stop test test-integration \
	frontend-install frontend-dev frontend-build frontend-typecheck \
	bench-setup bench-src bench-mflux bench-mflux-case bench-sanity bench-sanity-case \
	bench-audio-sanity bench-audio-sanity-lm bench-audio-sanity-heartmula \
	bench-wan-sanity bench-wan-baseline \
	check-consistency check-ep-boundary check-theme-legacy check-ui-compat check-engine-imports \
	sync-models-registry \
	strip-el-tokens test-engine-unit \
	pack-prereqs \
	pack-macos-desktop-sidecar pack-macos-desktop-shell pack-macos-desktop \
	pack-linux-server-venv pack-linux-server-sidecar pack-linux-server-archive pack-linux-server \
	pack-windows-venv pack-windows-sidecar \
	pack-windows-server-archive pack-windows-server \
	pack-windows-desktop-shell pack-windows-desktop pack-windows-desktop-release \
	desktop-prereqs desktop-sidecar desktop-tauri desktop-bundle \
	linux-cuda-venv linux-cuda-sidecar release-linux-cuda-tar release-linux-cuda \
	windows-cuda-venv windows-cuda-sidecar windows-cuda-desktop-sidecar \
	windows-desktop-tauri windows-desktop-bundle release-windows-desktop \
	release-windows-cuda-zip release-windows-cuda

PYTHON := .venv/bin/python3
BENCH_PY := tests/benchmark/venv/bin/python3
BENCH_PIP := tests/benchmark/venv/bin/pip
BENCH_BIN := tests/benchmark/venv/bin
BENCH_OUT := tests/benchmark/outputs
SRC_IMG := $(BENCH_OUT)/rewrite_src.png
OUT_DIR := $(CURDIR)/out

# Release packaging (see scripts/out_paths.py)
# Naming: pack-<platform>-<product>-<step>
#   platform: macos | linux | windows
#   product:  desktop (Tauri) | server (API only, zip/tar)
#   step:     venv | sidecar | shell | archive | bundle | release
RELEASE_VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
TORCH_INDEX_URL ?= https://download.pytorch.org/whl/cu124

# ============================================================================
# Benchmark (independent venv: tests/benchmark/venv)
# ============================================================================

BENCH_PYTHON ?= $(shell command -v python3.11 >/dev/null 2>&1 && echo python3.11 || echo python3)

bench-setup:
	$(BENCH_PYTHON) -m venv tests/benchmark/venv
	$(BENCH_PIP) install 'mflux>=0.15.0' 'transformers>=5.0,<6' 'huggingface-hub>=1.1.6,<2.0'
	@echo "Optional video deps: pip install -r tests/benchmark/requirements.txt (LTX git packages)"

$(SRC_IMG):
	@mkdir -p $(BENCH_OUT)
	$(PYTHON) $(CURDIR)/bin/danqing-generate \
		--model z-image-turbo \
		--prompt "a simple landscape" --seed 1 --steps 4 \
		--size 256x256 --output $(SRC_IMG)

bench-src: $(SRC_IMG)

bench-mflux: $(SRC_IMG)
	$(PYTHON) -m tests.benchmark mflux --all

bench-mflux-case: $(SRC_IMG)
	@if [ -z "$(ID)" ]; then \
		echo "Usage: make bench-mflux-case ID=<case-id>"; \
		exit 2; \
	fi
	$(PYTHON) -m tests.benchmark mflux --case $(ID)

bench-sanity:
	$(PYTHON) -m tests.benchmark sanity --all

bench-sanity-case:
	@if [ -z "$(ID)" ]; then \
		echo "Usage: make bench-sanity-case ID=<case-id>"; \
		exit 2; \
	fi
	$(PYTHON) -m tests.benchmark sanity --case $(ID)

bench-audio-sanity:
	$(PYTHON) -m tests.benchmark sanity --case ace-step-xl-sft-sanity
	$(PYTHON) -m tests.benchmark sanity --case heartmula-oss-3b-happy-new-year-sanity

bench-audio-sanity-lm:
	$(PYTHON) -m tests.benchmark sanity --case ace-step-xl-sft-sanity-lm

bench-audio-sanity-heartmula:
	$(PYTHON) -m tests.benchmark sanity --case heartmula-oss-3b-happy-new-year-sanity

bench-wan-sanity:
	$(PYTHON) -m tests.benchmark sanity --case wan-2.2-ti2v-5b-sanity

bench-wan-baseline:
	$(PYTHON) -m tests.benchmark sanity --case wan-2.2-ti2v-5b-baseline

# ============================================================================
# Frontend
# ============================================================================

FRONTEND_DIR := $(CURDIR)/frontend

frontend-install:
	cd $(FRONTEND_DIR) && npm install

frontend-dev: frontend-install
	cd $(FRONTEND_DIR) && npm run dev

frontend-build: frontend-install
	cd $(FRONTEND_DIR) && npm run build

frontend-typecheck: frontend-install
	cd $(FRONTEND_DIR) && npm run typecheck

# ============================================================================
# Dev server
# ============================================================================

dev start:
	@chmod +x scripts/*.sh
	@./scripts/start.sh

stop:
	@chmod +x scripts/*.sh
	@./scripts/stop.sh

test: test-engine-unit

test-integration:
	@echo "No integration tests yet (placeholder)"

# ============================================================================
# Quality gates
# ============================================================================

check-consistency:
	$(PYTHON) scripts/check_consistency.py

sync-models-registry:
	$(PYTHON) scripts/sync_workspace_registry.py

check-ep-boundary:
	$(PYTHON) scripts/check_ep_boundary.py

check-theme-legacy:
	$(PYTHON) scripts/check_theme_legacy.py

check-ui-compat:
	$(PYTHON) scripts/check_ui_compat.py

strip-el-tokens:
	$(PYTHON) scripts/strip_el_tokens.py

check-engine-imports:
	$(PYTHON) scripts/check_engine_backend_imports.py
	@echo "Engine backend import gate OK"

test-engine-unit:
	PYTHONPATH=. $(PYTHON) scripts/test_engine_unit.py

lint:
	$(PYTHON) scripts/make_lint.py
	@echo "Lint OK"

clean:
	$(PYTHON) scripts/clean_build.py

# ============================================================================
# Release packaging — pack-<platform>-<product>-<step>
# ============================================================================

pack-prereqs:
	@command -v npm >/dev/null 2>&1 || (printf '%s\n' 'npm not found. Install Node.js: https://nodejs.org/' >&2; exit 1)
	@command -v cargo >/dev/null 2>&1 || (printf '%s\n' 'cargo not found. Install Rust: https://rustup.rs/' >&2; exit 1)
	@echo "pack-prereqs OK (npm + cargo)"

# --- macOS desktop (MLX sidecar) — build on Darwin arm64 ---

export DANQING_PYINSTALLER_PROFILE ?= mlx

pack-macos-desktop-sidecar: frontend-build
	DANQING_PYINSTALLER_PROFILE=$(DANQING_PYINSTALLER_PROFILE) $(PYTHON) scripts/build_sidecar.py

pack-macos-desktop-shell: pack-prereqs
	@./scripts/tauri_build_macos.sh

pack-macos-desktop: frontend-build pack-macos-desktop-sidecar pack-macos-desktop-shell
	@echo "pack-macos-desktop -> $(OUT_DIR)/desktop/bundle/"

# --- Linux server (CUDA tar.gz) — build on Linux x86_64 ---

pack-linux-server-venv:
	@test -d .venv || python3.11 -m venv .venv || python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install torch torchvision --index-url $(TORCH_INDEX_URL)
	$(PYTHON) -m pip install -r requirements-linux-cuda.txt pyinstaller

pack-linux-server-sidecar: frontend-build
	DANQING_PYINSTALLER_PROFILE=full $(PYTHON) scripts/build_sidecar.py

pack-linux-server-archive: pack-linux-server-sidecar
	RELEASE_VERSION=$(RELEASE_VERSION) $(PYTHON) scripts/package_linux_cuda_release.py --version $(RELEASE_VERSION)
	@echo "pack-linux-server-archive -> $(OUT_DIR)/dist/danqing-studio-linux-cuda-x86_64-$(RELEASE_VERSION).tar.gz"

pack-linux-server: pack-linux-server-venv pack-linux-server-archive

# --- Windows (CUDA) — build on Windows x86_64 ---

pack-windows-venv:
	@test -d .venv || py -3.11 -m venv .venv || python -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install torch torchvision --index-url $(TORCH_INDEX_URL)
	$(PYTHON) -m pip install -r requirements-linux-cuda.txt pyinstaller

pack-windows-sidecar: frontend-build
	DANQING_PYINSTALLER_PROFILE=full $(PYTHON) scripts/build_sidecar.py

pack-windows-server-archive: pack-windows-sidecar
	RELEASE_VERSION=$(RELEASE_VERSION) $(PYTHON) scripts/package_windows_cuda_release.py --version $(RELEASE_VERSION)
	@echo "pack-windows-server-archive -> $(OUT_DIR)/dist/danqing-studio-windows-cuda-x86_64-$(RELEASE_VERSION).zip"

pack-windows-server: pack-windows-venv pack-windows-server-archive

pack-windows-desktop-shell: pack-prereqs
	$(PYTHON) scripts/tauri_build.py --platform windows

pack-windows-desktop: pack-windows-sidecar pack-windows-desktop-shell
	@echo "pack-windows-desktop -> $(OUT_DIR)/desktop/bundle/"

pack-windows-desktop-release: pack-windows-venv pack-windows-desktop

# ============================================================================
# Deprecated aliases (old names → pack-*)
# ============================================================================

desktop-prereqs: pack-prereqs
desktop-sidecar: pack-macos-desktop-sidecar
desktop-tauri: pack-macos-desktop-shell
desktop-bundle: pack-macos-desktop

linux-cuda-venv: pack-linux-server-venv
linux-cuda-sidecar: pack-linux-server-sidecar
release-linux-cuda-tar: pack-linux-server-archive
release-linux-cuda: pack-linux-server

windows-cuda-venv: pack-windows-venv
windows-cuda-sidecar: pack-windows-sidecar
windows-cuda-desktop-sidecar: pack-windows-sidecar
windows-desktop-tauri: pack-windows-desktop-shell
windows-desktop-bundle: pack-windows-desktop
release-windows-desktop: pack-windows-desktop-release
release-windows-cuda-zip: pack-windows-server-archive
release-windows-cuda: pack-windows-server

# ============================================================================
# Help
# ============================================================================

help:
	@echo "DanQing Studio v4 — Makefile"
	@echo ""
	@echo "Benchmark:"
	@echo "  bench-setup / bench-src / bench-mflux / bench-mflux-case"
	@echo "  bench-sanity / bench-sanity-case / bench-audio-sanity / bench-audio-sanity-heartmula"
	@echo ""
	@echo "Frontend:  frontend-install | frontend-dev | frontend-build | frontend-typecheck"
	@echo "Dev:       dev | start | stop"
	@echo "Test:      test | test-integration"
	@echo "Quality:   lint | check-*"
	@echo "Clean:     clean"
	@echo ""
	@echo "Release (pack-<platform>-<product>):"
	@echo "  macOS desktop (MLX):     pack-macos-desktop"
	@echo "  Linux server (CUDA):     pack-linux-server"
	@echo "  Windows desktop (CUDA):  pack-windows-desktop-release   (on Windows)"
	@echo "  Windows server zip:      pack-windows-server              (optional)"
	@echo ""
	@echo "Steps (when not using all-in-one targets above):"
	@echo "  pack-prereqs                  npm + cargo for Tauri"
	@echo "  pack-macos-desktop-sidecar    PyInstaller MLX sidecar"
	@echo "  pack-macos-desktop-shell      Tauri .app / .dmg"
	@echo "  pack-linux-server-venv        CUDA venv (Linux)"
	@echo "  pack-linux-server-sidecar     PyInstaller CUDA sidecar"
	@echo "  pack-linux-server-archive     .tar.gz"
	@echo "  pack-windows-venv             CUDA venv (Windows)"
	@echo "  pack-windows-sidecar          PyInstaller CUDA sidecar"
	@echo "  pack-windows-desktop-shell    Tauri NSIS installer"
	@echo "  pack-windows-server-archive   .zip (headless)"
