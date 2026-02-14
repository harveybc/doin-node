#!/usr/bin/env bash
# Run two DOIN nodes that peer with each other.
# Usage: ./scripts/run_two_node_test.sh
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="$(dirname "$REPO_ROOT")"

echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  DOIN Two-Node Test (Quadratic Domain)${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
pip install -e "$WORKSPACE/doin-core" -q 2>/dev/null || { echo -e "${RED}Failed to install doin-core${NC}"; exit 1; }
pip install -e "$WORKSPACE/doin-plugins" -q 2>/dev/null || { echo -e "${RED}Failed to install doin-plugins${NC}"; exit 1; }
pip install -e "$REPO_ROOT" -q 2>/dev/null || { echo -e "${RED}Failed to install doin-node${NC}"; exit 1; }
echo -e "${GREEN}Dependencies installed.${NC}"

# Clean data dirs
rm -rf "$REPO_ROOT/doin-data-node-a" "$REPO_ROOT/doin-data-node-b"

cleanup() {
    echo -e "\n${YELLOW}Stopping nodes...${NC}"
    kill $PID_A $PID_B 2>/dev/null || true
    wait $PID_A $PID_B 2>/dev/null || true
    echo -e "${GREEN}Done.${NC}"
}
trap cleanup EXIT

# Start Node A (port 8470)
echo -e "\n${YELLOW}Starting Node A on port 8470...${NC}"
cd "$REPO_ROOT"
python -m doin_node.cli --config examples/quadratic_node_a.json --log-level INFO &
PID_A=$!

sleep 3

# Start Node B (port 8471, peers with A)
echo -e "${YELLOW}Starting Node B on port 8471 (bootstrap: localhost:8470)...${NC}"
python -m doin_node.cli --config examples/quadratic_node_b.json --log-level INFO &
PID_B=$!

echo -e "\n${GREEN}Both nodes running. Press Ctrl+C to stop.${NC}"
echo -e "${CYAN}  Node A: http://localhost:8470/status${NC}"
echo -e "${CYAN}  Node B: http://localhost:8471/status${NC}\n"

# Poll status every 30s
while kill -0 $PID_A 2>/dev/null && kill -0 $PID_B 2>/dev/null; do
    sleep 30
    echo -e "\n${CYAN}--- Status Check ---${NC}"
    echo -e "${YELLOW}Node A:${NC}"
    curl -s http://localhost:8470/status 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "  (unavailable)"
    echo -e "${YELLOW}Node B:${NC}"
    curl -s http://localhost:8471/status 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "  (unavailable)"
done

echo -e "${RED}A node exited unexpectedly.${NC}"
