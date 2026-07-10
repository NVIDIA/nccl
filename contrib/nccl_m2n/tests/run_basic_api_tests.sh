#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# See LICENSE.txt for more license information.

# Thin launcher for the MPI variant of basic_api_test.
#
# Usage:
#   tests/run_basic_api_tests.sh [-N <ranks>] [-- <args forwarded to the binary>]
#
# Examples:
#   tests/run_basic_api_tests.sh -N 8 --filter full_replication
#   tests/run_basic_api_tests.sh -N 32 --algorithm direct
#
# To run the no-MPI single-host variant directly:
#   ./build/bin/basic_api_test_local [-N <ranks>] [other flags]

set -euo pipefail

N="${N:-8}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -N) N="$2"; shift 2;;
        --) shift; break;;
        -h|--help)
            cat <<EOF
Usage: $0 [-N <ranks>] [-- <args>]
Launches build/bin/basic_api_test_mpi via mpirun.
Anything after \`--\` (or any unrecognized flag) is forwarded to the binary.
Defaults: N=8.
EOF
            exit 0;;
        *) break;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build/bin/basic_api_test_mpi"

if [[ ! -x "$BIN" ]]; then
    echo "error: $BIN not found. Build with 'make tests' first." >&2
    exit 1
fi

exec mpirun -np "$N" "$BIN" "$@"
