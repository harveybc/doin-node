#!/usr/bin/env bash
# Run a single DOIN node with the quadratic test domain.
# Usage: ./scripts/run_single_test.sh
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="$(dirname "$REPO_ROOT")"

echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  DOIN Single Node Test (Quadratic Domain)${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
pip install -e "$WORKSPACE/doin-core" -q 2>/dev/null || { echo -e "${RED}Failed to install doin-core${NC}"; exit 1; }
pip install -e "$WORKSPACE/doin-plugins" -q 2>/dev/null || { echo -e "${RED}Failed to install doin-plugins${NC}"; exit 1; }
pip install -e "$REPO_ROOT" -q 2>/dev/null || { echo -e "${RED}Failed to install doin-node${NC}"; exit 1; }
echo -e "${GREEN}Dependencies installed.${NC}"

# Clean data dir
DATA_DIR="$REPO_ROOT/doin-data-single"
rm -rf "$DATA_DIR"

CONFIG="$REPO_ROOT/examples/quadratic_single_node.json"
echo -e "\n${YELLOW}Starting node on port 8470...${NC}"
echo -e "${YELLOW}Config: $CONFIG${NC}"
echo -e "${YELLOW}Data:   $DATA_DIR${NC}"
echo -e "\n${GREEN}Press Ctrl+C to stop.${NC}\n"

STATS_FILE="$REPO_ROOT/experiment_stats.csv"
echo -e "${YELLOW}Stats:  $STATS_FILE${NC}"

cd "$REPO_ROOT"
python -m doin_node.cli --config "$CONFIG" --log-level INFO

echo -e "\n${GREEN}Experiment stats written to: $STATS_FILE${NC}"
echo -e "${GREEN}Summary: ${STATS_FILE}.summary.json${NC}"
