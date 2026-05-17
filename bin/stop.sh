#!/bin/bash
# DanQing Studio v4 Shutdown Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Stopping DanQing Studio...${NC}"

PIDS=$(pgrep -f "uvicorn.*backend.main" 2>/dev/null || true)

if [ -z "$PIDS" ]; then
    echo -e "${YELLOW}No running DanQing Studio process found${NC}"
    exit 0
fi

echo -e "${YELLOW}Found PID: $PIDS${NC}"

echo "$PIDS" | xargs kill 2>/dev/null || true

sleep 2

REMAINING=$(pgrep -f "uvicorn.*backend.main" 2>/dev/null || true)
if [ -n "$REMAINING" ]; then
    echo -e "${RED}Process not responding to SIGTERM, force killing...${NC}"
    echo "$REMAINING" | xargs kill -9 2>/dev/null || true
fi

echo -e "${GREEN}✓ DanQing Studio stopped${NC}"
