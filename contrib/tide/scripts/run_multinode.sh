#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Hostfile-driven N-node torchrun launcher. Run it ON the first host of the hostfile.
# usage: [knobs] scripts/run_multinode.sh <tag> <script> [script args...]
# knobs: HOSTFILE(/etc/mpi/hostfile) NNODES(all) NPROC(8) PORT(29561) TMO(900) OUTDIR(./out)
#        LIB=nccl|none   (nccl: LD_PRELOAD $NCCL_HOME/lib/libnccl.so.2 on every node)
#        NCCL_HOME(the enclosing NCCL tree's build/)  EXTRA="K=V K=V"  DEBUG(WARN)  CTA_POLICY(unset)
#        WRAP="nsys profile ..." (command prefix in front of torchrun, every node)
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
TAG=${1:?tag}; shift; SCRIPT=${1:?script}; shift
HOSTFILE=${HOSTFILE:-/etc/mpi/hostfile}
mapfile -t HOSTS < <(awk 'NF && $1 !~ /^#/ {print $1}' "$HOSTFILE")
NNODES=${NNODES:-${#HOSTS[@]}}; HOSTS=("${HOSTS[@]:0:$NNODES}"); MASTER=${HOSTS[0]}
hostname -I | grep -qw "$MASTER" || { echo "FATAL: run on ${MASTER}; local IPs: $(hostname -I)"; exit 2; }
NCCL_HOME=${NCCL_HOME:-$(cd "$REPO/../.." && pwd)/build}
case "${LIB:-nccl}" in
  nccl) PRELOAD="${NCCL_HOME}/lib/libnccl.so.2" ;;
  none) PRELOAD="" ;;
  *) echo "bad LIB"; exit 2 ;;
esac
OUTDIR=${OUTDIR:-$REPO/out}; mkdir -p "$OUTDIR"; TS=$(date +%Y%m%d_%H%M%S); OUT=$OUTDIR/${TS}_${TAG}
# The NCCL_IB_* / socket settings are those of the RoCE rail fabric TIDE was developed on; EXTRA overrides them.
ENVS=(
  "PYTHONPATH=${REPO}" "PYTHONWARNINGS=ignore::FutureWarning" "NCCL_DEBUG=${DEBUG:-WARN}"
  "NCCL_IB_DISABLE=0" "NCCL_IB_GID_INDEX=3" "NCCL_IB_ECE_ENABLE=0" "NCCL_IB_TIMEOUT=21"
  "NCCL_IB_QPS_PER_CONNECTION=8" "NCCL_NET_OVERHEAD=1000" "NCCL_NET_PLUGIN=none"
  "NCCL_RUNTIME_CONNECT=0" "NCCL_SOCKET_IFNAME=bond0" "TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
  "NCCL_HOME=${NCCL_HOME}"
)
[ -n "$PRELOAD" ] && ENVS+=("LD_PRELOAD=${PRELOAD}")
[ -n "${CTA_POLICY:-}" ] && ENVS+=("NCCL_CTA_POLICY=${CTA_POLICY}")
# shellcheck disable=SC2206
[ -n "${EXTRA:-}" ] && ENVS+=(${EXTRA})
ARGS="$*"
launch() {  # $1 = node_rank   (fd limit: 64 ranks x several RMA contexts exhaust the default 1024)
  echo "cd ${REPO} && ulimit -n ${NOFILE:-65536} 2>/dev/null; env ${ENVS[*]} timeout ${TMO:-900} ${WRAP:-} torchrun --nnodes=${NNODES} --nproc_per_node=${NPROC:-8} \
    --node_rank=$1 --master_addr=${MASTER} --master_port=${PORT:-29561} ${SCRIPT} ${ARGS}"
}
gpu_busy() { local q="nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1"
  if [ "$1" = "$MASTER" ]; then bash -c "$q"; else ssh -o ConnectTimeout=8 -o BatchMode=yes "$1" "$q"; fi; }
t0=$(date +%s)
while :; do busy=0; for h in "${HOSTS[@]}"; do u=$(gpu_busy "$h" 2>/dev/null || echo 999999); [ "${u:-999999}" -gt "${IDLE_MAX_MIB:-512}" ] && busy=1; done
  [ $busy -eq 0 ] && break; [ $(( $(date +%s) - t0 )) -ge ${IDLE_TMO:-180} ] && { echo "FATAL: GPUs not idle"; exit 44; }; sleep 3; done
echo "=== ${TAG} nnodes=${NNODES} lib=${LIB:-nccl} script=${SCRIPT} args=[${ARGS}] extra=[${EXTRA:-}] $(date +%H:%M:%S) ===" | tee "${OUT}.log"
PIDS=()
for i in $(seq 1 $((NNODES - 1))); do
  ssh -o ConnectTimeout=8 -o BatchMode=yes "${HOSTS[$i]}" "$(launch $i)" > "${OUT}.node${i}.log" 2>&1 & PIDS+=($!)
done
eval "$(launch 0)" >> "${OUT}.log" 2>&1; rc=$?
for p in "${PIDS[@]}"; do wait "$p" || rc=$((rc | 1)); done
grep -h -e "^{" -e PASS -e FAIL "${OUT}.log" | cut -c1-400
echo "rc=${rc} log=${OUT}.log"; exit $rc
