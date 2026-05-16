.PHONY: bench-ref bench-all bench-mflux-all bench-seedvr2-mflux bench-case bench-list bench-src bench-setup bench-sanity bench-sanity-list bench-ltx-video bench-ltx-video-list bench-ltx-video-case check-engine-imports test-engine-unit parity-harness lint start stop help clean frontend-install frontend-dev frontend-build frontend-typecheck desktop-prereqs desktop-sidecar desktop-tauri desktop-bundle

PYTHON := .venv/bin/python3
BENCH_PY := tests/benchmark/venv/bin/python3
BENCH_PIP := tests/benchmark/venv/bin/pip
BENCH_BIN := tests/benchmark/venv/bin
BENCH_OUT := tests/benchmark/outputs
SRC_IMG := $(BENCH_OUT)/rewrite_src.png

# ============================================================================
# Off-channel benchmark tests (independent venv, no project dep pollution)
# ============================================================================

# Initialize benchmark test environment
bench-setup:
	python3 -m venv tests/benchmark/venv
	$(BENCH_PIP) install -r tests/benchmark/requirements.txt

# Generate source image for rewrite/upscale tests
$(SRC_IMG):
	@mkdir -p $(BENCH_OUT)
	$(PYTHON) $(CURDIR)/bin/danqing-generate \
		--model z-image-turbo \
		--prompt "a simple landscape" --seed 1 --steps 4 \
		--size 256x256 --output $(SRC_IMG)

bench-src: $(SRC_IMG)

# Generate all reference images (runs reference CLI, not DanQing engine)
bench-ref: $(SRC_IMG)
	$(PYTHON) -m tests.benchmark.run --all --ref-only

# Single model comparison
bench-case:
	@if [ -z "$(ID)" ]; then \
		$(PYTHON) -m tests.benchmark.run; \
	else \
		$(PYTHON) -m tests.benchmark.run --case $(ID); \
	fi

# All model comparison (reference + DanQing)
bench-all: $(SRC_IMG)
	$(PYTHON) -m tests.benchmark.run --all

# 同 bench-all；名称强调 mflux 对照（退出码见 tests/benchmark/cases.py 豁免集合）
bench-mflux-all: $(SRC_IMG)
	$(PYTHON) -m tests.benchmark.run --all

# SeedVR2 超分 7b + 3b 与 mflux 对照（3b 无本地权重时 SKIP）
bench-seedvr2-mflux: $(SRC_IMG)
	$(PYTHON) -m tests.benchmark.run --case seedvr2-7b-upscale && $(PYTHON) -m tests.benchmark.run --case seedvr2-3b-upscale

# 无外部参考 CLI 的模型：仅跑 danqing 生成 + 像素健全性（白/黑/平场拒收）
bench-sanity:
	$(PYTHON) -m tests.benchmark.run --sanity

bench-sanity-list:
	$(PYTHON) -c "from tests.benchmark.cases import list_sanity_cases; print('\n'.join(list_sanity_cases()))"

# LTX: DanQing ``danqing-video-generate`` vs ``tests/benchmark/venv/bin/ltx-2-mlx``（需本地 MLX 权重 + ffmpeg）
bench-ltx-video:
	$(PYTHON) -m tests.benchmark.run --ltx-video

bench-ltx-video-list:
	$(PYTHON) -c "from tests.benchmark.ltx_video_cases import list_ltx_video_cases; print('\n'.join(list_ltx_video_cases()))"

bench-ltx-video-case:
	@if [ -z "$(ID)" ]; then \
		echo "Usage: make bench-ltx-video-case ID=<ltx-video-case-id>"; \
		$(MAKE) bench-ltx-video-list; \
		exit 2; \
	fi
	$(PYTHON) -m tests.benchmark.run --ltx-video --ltx-video-case $(ID)

# List all test cases
bench-list:
	$(PYTHON) -c "from tests.benchmark.cases import list_cases; print('\n'.join(list_cases()))"

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

check-engine-imports:
	$(PYTHON) scripts/check_engine_backend_imports.py
	@echo "Engine backend import gate OK"

test-engine-unit:
	PYTHONPATH=. $(PYTHON) -m unittest tests.test_mlx_affine_quant_inference tests.test_ltx_weights -v

# Greenfield parity smoke (no external ref CLI); extend with full bench-* as needed
parity-harness: bench-sanity

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
	@echo "  make bench-setup     Initialize benchmark test environment (independent venv)"
	@echo "  make bench-src       Generate source image (for rewrite/upscale)"
	@echo "  make bench-all       Full mflux PSNR suite (same as bench-mflux-all)"
	@echo "  make bench-mflux-all Full mflux PSNR suite (exit 0 if only exempt rewrite gaps fail)"
	@echo "  make bench-seedvr2-mflux  SeedVR2 7b+3b upscale vs mflux (3b SKIP if bundle missing)"
	@echo "  make bench-sanity    Output sanity (no mflux ref)"
	@echo "  make bench-list      List mflux benchmark case ids"
	@echo "  make bench-sanity-list  List sanity case ids"
	@echo "  make bench-ltx-video   LTX distilled MLX vs ltx-2-mlx CLI (frame PSNR, bench venv)"
	@echo "  make bench-ltx-video-list  List LTX video benchmark case ids"
	@echo "  make bench-ltx-video-case ID=<id>  Single LTX video case"
	@echo "  make check-engine-imports  Enforce mlx/torch only in runtime/ and *_mlx.py / *_cuda.py"
	@echo "  make test-engine-unit   Stdlib unit tests (MLX affine quant inference + LTX remap/import)"
	@echo "  make parity-harness     Alias: make bench-sanity (output sanity gate)"
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
