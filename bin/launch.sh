#!/bin/bash
# DanQing Studio v4 Launcher

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════╗${NC}"
echo -e "${BLUE}║    DanQing Studio v4 Launcher       ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════╝${NC}"
echo ""

if [[ "$(uname)" != "Darwin" ]]; then
    echo -e "${RED}Error: This application only supports macOS${NC}"
    exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
    echo -e "${YELLOW}Warning: Non-Apple Silicon detected, MLX acceleration may not be available${NC}"
fi

PYTHON311="/opt/homebrew/bin/python3.11"
if [ ! -f "$PYTHON311" ]; then
    PYTHON311="$(command -v python3.11 || true)"
fi

if [ -z "$PYTHON311" ] || [ ! -f "$PYTHON311" ]; then
    echo -e "${RED}Error: Python 3.11 not found${NC}"
    echo -e "${YELLOW}Please run: brew install python@3.11${NC}"
    exit 1
fi

PYTHON_VERSION=$("$PYTHON311" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${GREEN}✓ Python ${PYTHON_VERSION}${NC}"

VENV_DIR="$PROJECT_ROOT/.venv"
VENV_PYTHON="$VENV_DIR/bin/python3"
VENV_PIP="$VENV_DIR/bin/pip3"

NEED_CREATE=0
if [ ! -f "$VENV_PYTHON" ] || [ ! -f "$VENV_PIP" ]; then
    NEED_CREATE=1
else
    VENV_PY_VERSION=$("$VENV_PYTHON" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [ "$VENV_PY_VERSION" != "3.11" ]; then
        echo -e "${YELLOW}Virtual env Python version (${VENV_PY_VERSION}) mismatch, need 3.11${NC}"
        NEED_CREATE=1
    fi
fi

if [ "$NEED_CREATE" -eq 1 ]; then
    echo -e "${YELLOW}Creating virtual environment (Python 3.11)...${NC}"
    rm -rf "$VENV_DIR"
    "$PYTHON311" -m venv "$VENV_DIR"
    "$VENV_PYTHON" -m ensurepip --upgrade
    "$VENV_PIP" install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip -q
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

echo -e "${BLUE}Checking dependencies...${NC}"
if ! "$VENV_PYTHON" -c "import fastapi, uvicorn, mlx, pydantic" 2>/dev/null; then
    echo -e "${YELLOW}Installing dependencies to virtual environment...${NC}"
    "$VENV_PIP" install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt -q
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Dependencies ready${NC}"
fi

# Data dirs under custom workspace when configured (see default_config/workspace.pointer.json)
"$VENV_PYTHON" -c "
from pathlib import Path
import sys
sys.path.insert(0, '${PROJECT_ROOT}')
from backend.utils.config_paths import resolve_default_config_root
from backend.utils.workspace import prepare_data_directories
root = Path('${PROJECT_ROOT}').resolve()
default_cfg = resolve_default_config_root(bootstrap_root=root, bundle_root=None)
prepare_data_directories(root, default_config_root=default_cfg)
"

# Build frontend if needed (Vite -> out/frontend/dist/, see frontend/vite.config.ts)
FRONTEND_DIR="$PROJECT_ROOT/frontend"
FRONTEND_DIST="$PROJECT_ROOT/out/frontend/dist"
if [ -f "$FRONTEND_DIR/package.json" ]; then
    NEED_FRONTEND_BUILD=0
    if [ ! -f "$FRONTEND_DIST/index.html" ]; then
        NEED_FRONTEND_BUILD=1
    elif [ "$FRONTEND_DIR/package.json" -nt "$FRONTEND_DIST/index.html" ] \
        || [ "$FRONTEND_DIR/vite.config.ts" -nt "$FRONTEND_DIST/index.html" ]; then
        NEED_FRONTEND_BUILD=1
    elif find "$FRONTEND_DIR/src" -type f -newer "$FRONTEND_DIST/index.html" -print -quit 2>/dev/null | grep -q .; then
        NEED_FRONTEND_BUILD=1
    fi

    if [ "$NEED_FRONTEND_BUILD" -eq 1 ]; then
        echo -e "${BLUE}Building frontend -> out/frontend/dist ...${NC}"
        cd "$FRONTEND_DIR"
        if [ ! -d "node_modules" ]; then
            npm install
        fi
        npm run build
        cd "$PROJECT_ROOT"
        if [ ! -f "$FRONTEND_DIST/index.html" ]; then
            echo -e "${RED}Error: frontend build failed (missing $FRONTEND_DIST/index.html)${NC}" >&2
            exit 1
        fi
        echo -e "${GREEN}✓ Frontend built${NC}"
    else
        echo -e "${GREEN}✓ Frontend up to date (out/frontend/dist)${NC}"
    fi
fi

echo ""
echo -e "${GREEN}Starting DanQing Studio v4...${NC}"
echo -e "${BLUE}Access at: http://localhost:7860${NC}"
echo ""

exec "$VENV_PYTHON" -m uvicorn backend.main:app --host 0.0.0.0 --port 7860 --reload
