.PHONY: bench-setup bench-src bench-mflux bench-mflux-case bench-sanity bench-sanity-case test-engine-unit check-consistency check-engine-imports lint start stop help clean frontend-install frontend-dev frontend-build frontend-typecheck desktop-prereqs desktop-sidecar desktop-tauri desktop-bundle

PYTHON := .venv/bin/python3
BENCH_PY := tests/benchmark/venv/bin/python3
BENCH_PIP := tests/benchmark/venv/bin/pip
BENCH_BIN := tests/benchmark/venv/bin
BENCH_OUT := tests/benchmark/outputs
SRC_IMG := $(BENCH_OUT)/rewrite_src.png

# ============================================================================
# Off-channel benchmark tests (independent venv, no project dep pollution)
# ============================================================================

# mflux reference CLI venv (``make bench-setup`` once)
bench-setup:
	python3 -m venv tests/benchmark/venv
	$(BENCH_PIP) install -r tests/benchmark/requirements.txt

$(SRC_IMG):
	@mkdir -p $(BENCH_OUT)
	$(PYTHON) $(CURDIR)/bin/danqing-generate \
		--model z-image-turbo \
		--prompt "a simple landscape" --seed 1 --steps 4 \
		--size 256x256 --output $(SRC_IMG)

bench-src: $(SRC_IMG)

# mflux PSNR/SSIM vs reference CLI (all models in cases.ALL_CASES)
bench-mflux: $(SRC_IMG)
	$(PYTHON) -m tests.benchmark mflux --all

bench-mflux-case: $(SRC_IMG)
	@if [ -z "$(ID)" ]; then \
		echo "Usage: make bench-mflux-case ID=<case-id>  (make bench-mflux --list via: python -m tests.benchmark mflux --list)"; \
		exit 2; \
	fi
	$(PYTHON) -m tests.benchmark mflux --case $(ID)

# Output sanity — reject white/black/near-flat images (cases.ALL_SANITY_CASES)
bench-sanity:
	$(PYTHON) -m tests.benchmark sanity --all

bench-sanity-case:
	@if [ -z "$(ID)" ]; then \
		echo "Usage: make bench-sanity-case ID=<case-id>  (list: python -m tests.benchmark sanity --list)"; \
		exit 2; \
	fi
	$(PYTHON) -m tests.benchmark sanity --case $(ID)

# ============================================================================
# Frontend (Vite + Vue 3 + TypeScript)
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
# API start/stop
# ============================================================================

start:
	./bin/launch.sh

stop:
	./bin/stop.sh

# ============================================================================
# Dual-platform import gate (see docs/dual_platform_architecture.md §8.5)
# ============================================================================

check-consistency:
	$(PYTHON) scripts/check_consistency.py

check-engine-imports:
	$(PYTHON) scripts/check_engine_backend_imports.py
	@echo "Engine backend import gate OK"

test-engine-unit:
	PYTHONPATH=. $(PYTHON) scripts/test_engine_unit.py

# ============================================================================
# Syntax check
# ============================================================================

lint:
	$(PYTHON) scripts/make_lint.py
	@echo "Lint OK"

# ============================================================================
# Build artifacts (see scripts/out_paths.py)
# ============================================================================

OUT_DIR := $(CURDIR)/out

clean:
	$(PYTHON) scripts/clean_build.py

# ============================================================================
# Desktop (Tauri 2 shell + PyInstaller sidecar)
# ============================================================================

DESKTOP_DIR := $(CURDIR)/desktop

desktop-prereqs:
	@command -v npm >/dev/null 2>&1 || (printf '%s\n' 'npm not found. Install Node.js: https://nodejs.org/' >&2; exit 1)
	@command -v cargo >/dev/null 2>&1 || (printf '%s\n' 'cargo not found. Install Rust: https://rustup.rs/  Then add ~/.cargo/bin to PATH, e.g.  source "$$HOME/.cargo/env"' >&2; exit 1)
	@echo "desktop prerequisites OK (npm + cargo)"

# MLX-only PyInstaller on macOS (no *_cuda / torch). Override: DANQING_PYINSTALLER_PROFILE=full
export DANQING_PYINSTALLER_PROFILE ?= mlx

desktop-sidecar: frontend-build
	DANQING_PYINSTALLER_PROFILE=$(DANQING_PYINSTALLER_PROFILE) $(PYTHON) scripts/build_sidecar.py

desktop-tauri: desktop-prereqs
	cd $(DESKTOP_DIR) && npm install && npm run build

# frontend dist -> MLX sidecar -> Tauri (outputs under out/)
desktop-bundle: frontend-build desktop-sidecar desktop-tauri
	@echo "Desktop bundle: $(OUT_DIR)/desktop/cargo/release/bundle/"

# ============================================================================
# Help
# ============================================================================

help:
	@echo "DanQing Studio v4 — Makefile"
	@echo ""
	@echo "  make bench-setup       mflux reference CLI venv (tests/benchmark/venv)"
	@echo "  make bench-src         Placeholder image for rewrite/upscale cases"
	@echo "  make bench-mflux       All mflux PSNR cases"
	@echo "  make bench-mflux-case ID=<id>  Single mflux case"
	@echo "  make bench-sanity      All output-sanity cases"
	@echo "  make bench-sanity-case ID=<id>  Single sanity case"
	@echo "  make check-consistency Registry / routes / i18n gate"
	@echo "  make check-engine-imports  mlx/torch import gate"
	@echo "  make test-engine-unit  Backend engine unit tests (scripts/test_engine_unit.py)"
	@echo "  make lint            Syntax check"
	@echo "  make start/stop      API start/stop"
	@echo "  make frontend-install  Install frontend dependencies (npm install)"
	@echo "  make frontend-dev    Start frontend dev server (Vite, port 5173)"
	@echo "  make frontend-build  Build frontend for production"
	@echo "  make frontend-typecheck  Run TypeScript type check"
	@echo "  make clean           Remove out/ and legacy dist/build artifacts"
	@echo "  make desktop-prereqs Check npm + cargo (Tauri build requirements)"
	@echo "  make desktop-sidecar PyInstaller -> out/sidecar/danqing-api (MLX on macOS)"
	@echo "  make desktop-tauri   Tauri bundle -> out/desktop/cargo/release/bundle/"
	@echo "  make desktop-bundle  Full desktop build (recommended)"
	@echo "  ./scripts/build_desktop.sh  Same as desktop-bundle (shell entry)"
