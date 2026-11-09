#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Single-node launcher: scripts/run_local.sh <nproc> <python script> [args...]
# Forces the whole process (torch's communicator included) onto the NCCL build in NCCL_HOME
# (default: the enclosing NCCL tree's build/).
set -euo pipefail
cd "$(dirname "$0")/.."
NCCL_HOME=${NCCL_HOME:-$(cd ../.. && pwd)/build}
NPROC=$1; shift
export LD_PRELOAD="${NCCL_HOME}/lib/libnccl.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONWARNINGS="ignore::FutureWarning"
exec torchrun --standalone --nproc_per_node "$NPROC" "$@"
