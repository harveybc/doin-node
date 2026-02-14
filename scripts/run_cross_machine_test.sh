#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  DOIN Cross-Machine Test — Delta ↔ Dragon (Quadratic Domain)
# ═══════════════════════════════════════════════════════════════════
#
#  This script runs on ONE machine. Run it on both machines.
#
#  Usage:
#    On Delta (first node):
#      ./scripts/run_cross_machine_test.sh delta
#
#    On Dragon (second node — needs Delta's IP):
#      ./scripts/run_cross_machine_test.sh dragon <DELTA_IP>
#
#  The script will:
#    1. Install dependencies
#    2. Copy the right config
#    3. Start the node
#    4. Poll /status every 30s
#
#  Key metric: Compare time_to_target between:
#    - Single node (from earlier test)
#    - Two nodes (this test)
#    If two nodes converge faster → scalability proven!
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="$(dirname "$REPO_ROOT")"

ROLE="${1:-}"
PEER_IP="${2:-}"

if [[ -z "$ROLE" ]] || [[ "$ROLE" != "delta" && "$ROLE" != "dragon" ]]; then
    echo -e "${RED}Usage: $0 <delta|dragon> [peer_ip]${NC}"
    echo -e "  delta  — First node (no peers needed)"
    echo -e "  dragon — Second node (needs Delta's IP)"
    exit 1
fi

if [[ "$ROLE" == "dragon" && -z "$PEER_IP" ]]; then
    echo -e "${RED}Dragon needs Delta's IP: $0 dragon <DELTA_IP>${NC}"
    exit 1
fi

echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  DOIN Cross-Machine Test — ${ROLE^^}${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════${NC}"

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
pip install -e "$WORKSPACE/doin-core" -q 2>/dev/null || { echo -e "${RED}Failed to install doin-core${NC}"; exit 1; }
pip install -e "$WORKSPACE/doin-plugins" -q 2>/dev/null || { echo -e "${RED}Failed to install doin-plugins${NC}"; exit 1; }
pip install -e "$REPO_ROOT" -q 2>/dev/null || { echo -e "${RED}Failed to install doin-node${NC}"; exit 1; }
echo -e "${GREEN}Dependencies installed.${NC}"

# Set up config
CONFIG_SRC="$REPO_ROOT/examples/cross_machine_${ROLE}.json"
CONFIG="$REPO_ROOT/cross_machine_config.json"

if [[ "$ROLE" == "dragon" ]]; then
    # Replace <DELTA_IP> placeholder with actual IP
    sed "s/<DELTA_IP>/$PEER_IP/g" "$CONFIG_SRC" > "$CONFIG"
    echo -e "${GREEN}Config: Dragon bootstrapping to ${PEER_IP}:8470${NC}"
else
    cp "$CONFIG_SRC" "$CONFIG"
    echo -e "${GREEN}Config: Delta (seed node, no bootstrap peers)${NC}"
fi

# Clean data dir
DATA_DIR="$REPO_ROOT/doin-data-${ROLE}"
rm -rf "$DATA_DIR"

echo -e "\n${YELLOW}Starting ${ROLE^^} node on port 8470...${NC}"
echo -e "${YELLOW}Config: $CONFIG${NC}"
echo -e "${YELLOW}Data: $DATA_DIR${NC}"
echo -e "${YELLOW}Stats CSV: ./experiment_stats_${ROLE}.csv${NC}"
echo -e "${YELLOW}OLAP DB: ./olap_${ROLE}.db${NC}"

if [[ "$ROLE" == "dragon" ]]; then
    echo -e "${YELLOW}Bootstrap peer: ${PEER_IP}:8470${NC}"
fi

echo -e "\n${GREEN}Press Ctrl+C to stop.${NC}\n"

# Show our IP for the other machine
echo -e "${CYAN}My IPs (give one to the other machine):${NC}"
hostname -I 2>/dev/null || ip addr show | grep 'inet ' | awk '{print $2}' | cut -d/ -f1
echo ""

cd "$REPO_ROOT"
python -m doin_node.cli --config "$CONFIG" --log-level INFO
