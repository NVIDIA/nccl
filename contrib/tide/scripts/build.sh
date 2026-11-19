#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build the tide._C extension in place against an NCCL >= 2.31.2 build (needs nvcc: CUDA_HOME, TORCH_CUDA_ARCH_LIST).
# Development build: the test-only fault hooks are compiled in (TIDE_FAULT_INJECTION=0 to leave them out,
# as `pip install` does by default).
# Usage: scripts/build.sh   (NCCL: the enclosing tree's build/, or NCCL_HOME)
set -euo pipefail
cd "$(dirname "$0")/.."
export MAX_JOBS="${MAX_JOBS:-16}" TIDE_FAULT_INJECTION="${TIDE_FAULT_INJECTION:-1}"
LOG=$(mktemp)
if ! python3 setup.py build_ext --inplace > "$LOG" 2>&1; then
  grep -v -e Wattributes -e "declared with greater visibility" "$LOG" | grep -i -B3 -A8 "error" | head -80
  echo "BUILD FAILED (full log: $LOG)"; exit 1
fi
rm -f "$LOG"
python3 -c "import tide, tide._C as C; print('tide._C ok, compiled nccl', C.nccl_compiled_version)" 2>&1 | grep -v -e FutureWarning -e "import pynvml"
