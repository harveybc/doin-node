#!/bin/bash
PIP_NVIDIA=$(find /home/openclaw/.local/lib/python3.12/site-packages/nvidia -name "lib" -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="/home/openclaw/.local/lib/tf_cuda_shims:${PIP_NVIDIA}/home/harveybc/anaconda3/envs/tensorflow/lib"
export PREDICTOR_QUIET=1
export TF_CPP_MIN_LOG_LEVEL=3
cd /home/openclaw/.openclaw/workspace/doin-node
exec python3 -u -c "
import sys
sys.argv=['doin-node','--config','examples/predictor_single_node.json','--log-level','INFO','--olap-db','predictor_olap.db']
from doin_node.cli import main
main()
" >> dragon_node.log 2>&1
