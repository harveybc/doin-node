#!/bin/bash
# Watch DOIN optimization progress — shows only round completions and key events
# Usage: bash watch_optimization.sh [dragon|omega|both]

DRAGON_LOG="/home/openclaw/.openclaw/workspace/doin-node/dragon_node.log"
OMEGA_LOG_CMD="ssh -p 62024 harveybc@192.168.1.119 'tail -f /home/harveybc/doin-node/omega_node.log'"

filter='Round|fitness|CONVERGE|champion|best_perf|OPTIMAE|migration|improvement|🎯|📊|🏆|Error|WARNING'

case "${1:-both}" in
  dragon)
    echo "=== Watching Dragon ==="
    tail -f "$DRAGON_LOG" | grep --line-buffered -iE "$filter"
    ;;
  omega)
    echo "=== Watching Omega ==="
    ssh -p 62024 harveybc@192.168.1.119 "tail -f /home/harveybc/doin-node/omega_node.log" | grep --line-buffered -iE "$filter"
    ;;
  both)
    echo "=== Watching Dragon + Omega ==="
    (tail -f "$DRAGON_LOG" | grep --line-buffered -iE "$filter" | sed 's/^/[DRAGON] /') &
    (ssh -p 62024 harveybc@192.168.1.119 "tail -f /home/harveybc/doin-node/omega_node.log" | grep --line-buffered -iE "$filter" | sed 's/^/[OMEGA]  /') &
    wait
    ;;
esac
