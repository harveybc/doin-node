#!/usr/bin/env bash
# Run the full DOIN benchmark suite.
# Usage: ./scripts/run_benchmarks.sh [--quick]
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="$(dirname "$REPO_ROOT")"

echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  DOIN Benchmark Suite${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
pip install -e "$WORKSPACE/doin-core" -q 2>/dev/null || { echo -e "${RED}Failed to install doin-core${NC}"; exit 1; }
pip install -e "$WORKSPACE/doin-plugins" -q 2>/dev/null || { echo -e "${RED}Failed to install doin-plugins${NC}"; exit 1; }
pip install -e "$REPO_ROOT" -q 2>/dev/null || { echo -e "${RED}Failed to install doin-node${NC}"; exit 1; }
pip install aiohttp -q 2>/dev/null
echo -e "${GREEN}Dependencies installed.${NC}"

ARGS=""
if [[ "${1:-}" == "--quick" ]]; then
    ARGS="--quick"
    echo -e "${YELLOW}Running in quick mode.${NC}"
fi

OUTPUT="$REPO_ROOT/benchmark_results.json"

echo -e "\n${YELLOW}Running benchmarks...${NC}\n"
cd "$REPO_ROOT"
python -m doin_node.benchmarks.runner $ARGS --output "$OUTPUT"

echo -e "\n${GREEN}Results: $OUTPUT${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
