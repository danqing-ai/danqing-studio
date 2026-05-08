.PHONY: bench-ref bench-all bench-case bench-list bench-src bench-setup

PYTHON := .venv/bin/python3
BENCH_PY := tests/benchmark/venv/bin/python3
BENCH_PIP := tests/benchmark/venv/bin/pip
BENCH_BIN := tests/benchmark/venv/bin
BENCH_OUT := tests/benchmark/outputs
SRC_IMG := $(BENCH_OUT)/rewrite_src.png

# ============================================================================
# 旁路基准测试（独立 venv，不污染项目依赖）
# ============================================================================

# 初始化基准测试环境
bench-setup:
	python3 -m venv tests/benchmark/venv
	$(BENCH_PIP) install -r tests/benchmark/requirements.txt

# 生成 rewrite/upscale 用的源图
$(SRC_IMG):
	@mkdir -p $(BENCH_OUT)
	$(BENCH_BIN)/mflux-generate-z-image-turbo \
		--model models/Base/z-image-turbo-fp16 \
		--prompt "a simple landscape" --seed 1 --steps 4 \
		--width 256 --height 256 --output $(SRC_IMG)

bench-src: $(SRC_IMG)

# 生成所有 mflux 参考图（不跑丹青引擎）
bench-ref: $(SRC_IMG)
	$(PYTHON) -m tests.benchmark.run --all --ref-only

# 单个模型对比
bench-case:
	@if [ -z "$(ID)" ]; then \
		$(PYTHON) -m tests.benchmark.run; \
	else \
		$(PYTHON) -m tests.benchmark.run --case $(ID); \
	fi

# 全部模型对比（参考 + 丹青）
bench-all: $(SRC_IMG)
	$(PYTHON) -m tests.benchmark.run --all

# 列出所有用例
bench-list:
	$(PYTHON) -c "from tests.benchmark.cases import list_cases; print('\n'.join(list_cases()))"

# ============================================================================
# API 启动/停止
# ============================================================================

start:
	./bin/launch.sh

stop:
	./bin/stop.sh

# ============================================================================
# 语法检查
# ============================================================================

lint:
	$(PYTHON) -c "
import py_compile, os
fail = 0
for root, dirs, files in os.walk('backend/engine'):
    for f in files:
        if f.endswith('.py') and '__pycache__' not in root:
            try: py_compile.compile(os.path.join(root,f), doraise=True)
            except py_compile.PyCompileError as e: print(f'FAIL: {os.path.join(root,f)}'); fail=1
for f in ['backend/main.py','tests/benchmark/cases.py','tests/benchmark/runner.py','tests/benchmark/compare.py']:
    try: py_compile.compile(f, doraise=True)
    except py_compile.PyCompileError as e: print(f'FAIL: {f}'); fail=1
exit(fail)
"
	@echo "Lint OK"

# ============================================================================
# 帮助
# ============================================================================

help:
	@echo "DanQing Studio v4 — Makefile"
	@echo ""
	@echo "  make bench-setup     初始化基准测试环境（独立 venv）"
	@echo "  make bench-src       生成源图（改写/超分用）"
	@echo "  make bench-ref       生成 mflux CLI 参考图"
	@echo "  make bench-all       全模型对比测试"
	@echo "  make bench-case ID=<id>  单用例测试"
	@echo "  make bench-list      列出所有用例"
	@echo "  make lint            语法检查"
	@echo "  make start/stop      API 启动/停止"
