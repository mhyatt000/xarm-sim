#!/usr/bin/env bash
# Single-host launcher: one torchrun rank per GPU on this machine.
# Usage: [GPUS=N] scripts/host.sh scripts/simpledagger.py [args...]
set -euo pipefail
# gsplat's JIT must use the system gcc: the PATH's conda-forge gcc 15 links a
# libstdc++ newer than the system's (CXXABI_1.3.15 import error otherwise)
export CXX="${CXX:-/usr/bin/g++}" CC="${CC:-/usr/bin/gcc}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec torchrun \
  --standalone \
  --nproc-per-node="${GPUS:-$(nvidia-smi -L | wc -l)}" \
  "$@"
