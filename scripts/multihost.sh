#!/usr/bin/env bash
# Multi-host launcher: run the SAME command on every node.
# Usage: NNODES=2 RDZV=host0:29500 [GPUS=N] scripts/multihost.sh scripts/simpledagger.py [args...]
set -euo pipefail
# gsplat's JIT must use the system gcc: the PATH's conda-forge gcc 15 links a
# libstdc++ newer than the system's (CXXABI_1.3.15 import error otherwise)
export CXX="${CXX:-/usr/bin/g++}" CC="${CC:-/usr/bin/gcc}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec torchrun \
  --nnodes="${NNODES:?total number of nodes, e.g. NNODES=2}" \
  --nproc-per-node="${GPUS:-$(nvidia-smi -L | wc -l)}" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${RDZV:?rendezvous address, e.g. RDZV=host0:29500}" \
  "$@"
