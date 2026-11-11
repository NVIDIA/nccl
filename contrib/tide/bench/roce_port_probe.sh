#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Which LAG member carries the RoCE bytes: diff the per-member physical port counters of bond1
# (rail 0 = rank 0's NIC) around one 2-node run. RDMA bypasses the kernel, so tcpdump/AF_PACKET see
# nothing; ethtool *_phy counters do. Run on the first host of $HOSTFILE.
# usage: [EXTRA="K=V ..."] bench/roce_port_probe.sh <outdir> <tag> <bench script> [args...]
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd); OUT=${1:?outdir}; TAG=${2:?tag}; shift 2; mkdir -p "$OUT"
SLAVES=$(awk '/Slave Interface/ {print $3}' /proc/net/bonding/bond1)
snap() { for s in $SLAVES; do ethtool -S "$s" | awk -v s="$s" '/^ *(tx|rx)_bytes_phy:/ {print s, $1, $2}'; done; }
cd "$REPO"
snap > "$OUT/phy_${TAG}_before.txt"
NNODES=2 TMO=${TMO:-300} bash scripts/run_multinode.sh "probe_${TAG}" "$@" > "$OUT/run_${TAG}.log" 2>&1
snap > "$OUT/phy_${TAG}_after.txt"
echo "== $TAG  extra=[${EXTRA:-}]  $(grep -h -o '"nic_GBs_rank0": [0-9.]*\|"arm": "[a-z_]*", "GBs": [0-9.]*' "$OUT/run_${TAG}.log" | head -2 | tr '\n' ' ')"
grep -q "FATAL\|rc=[1-9]" "$OUT/run_${TAG}.log" && grep -h "FATAL\|rc=" "$OUT/run_${TAG}.log" | head -2
paste "$OUT/phy_${TAG}_before.txt" "$OUT/phy_${TAG}_after.txt" | awk '{printf "   %-6s %-14s %8.2f GiB\n", $1, $2, ($6-$3)/2^30}'
