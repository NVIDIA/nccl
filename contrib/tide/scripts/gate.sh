#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# The regression gate: every leg must exit 0 (negatives exit 0 only when their check catches the
# injected fault, so they need the fault-injection build scripts/build.sh makes).
# usage: scripts/gate.sh offline | core1 | coremulti | ep | decouple | 1 | multi | all
#   offline needs no GPU; the others run on the first host of HOSTFILE (default /etc/mpi/hostfile).
# knobs: N (multi-node legs' node count, default: every host of HOSTFILE), ONLY=<regex> (tags to run),
#        GATE_ENV="K=V ..." (added to every leg),
#        OUTDIR (logs + gate_summary.txt), plus scripts/run_multinode.sh's (NCCL_HOME, TMO, ...)
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
STAGE=${1:?offline | core1 | coremulti | ep | decouple | 1 | multi | all}
HOSTFILE=${HOSTFILE:-/etc/mpi/hostfile}
N=${N:-$(awk 'NF && $1 !~ /^#/' "$HOSTFILE" 2>/dev/null | wc -l)}
OUTDIR=${OUTDIR:-$REPO/out/gate_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUTDIR"
SUMMARY=$OUTDIR/gate_summary.txt
fails=0

record() {  # tag rc
  local verdict=PASS; [ "$2" -eq 0 ] || { verdict=FAIL; fails=$((fails + 1)); }
  printf '%-24s %s rc=%s\n' "$1" "$verdict" "$2" | tee -a "$SUMMARY"
}

leg() {  # tag nodes "ENV=V ..." script [args...]
  local tag=$1 nodes=$2 env=$3; shift 3
  [ -n "${ONLY:-}" ] && ! [[ $tag =~ $ONLY ]] && return
  local nproc=8; [[ $1 == test/test_fsdp_compat.py ]] && nproc=1
  ( cd "$REPO" && NNODES=$nodes NPROC=$nproc OUTDIR=$OUTDIR PORT=$((29600 + nodes * 10)) TMO=${TMO:-900} \
      EXTRA="$env ${GATE_ENV:-}" bash scripts/run_multinode.sh "$tag" "$@" ) > "$OUTDIR/$tag.out" 2>&1
  record "$tag" $?
}

dc_fault() {  # tag nodes "ENV=V ..." script [args...]: PASS iff the job fails, a rank logged the device network phase's FATAL and it ended in time
  local tag=$1 nodes=$2 env=$3; shift 3
  [ -n "${ONLY:-}" ] && ! [[ $tag =~ $ONLY ]] && return
  local tmo=${DC_FAULT_TMO:-240} t0 rc dt
  t0=$(date +%s)
  ( cd "$REPO" && NNODES=$nodes NPROC=8 OUTDIR=$OUTDIR PORT=$((29600 + nodes * 10)) TMO=$tmo \
      EXTRA="$env ${GATE_ENV:-}" bash scripts/run_multinode.sh "$tag" "$@" ) > "$OUTDIR/$tag.out" 2>&1
  rc=$?
  dt=$(( $(date +%s) - t0 ))
  if [ $rc -ne 0 ] && [ $dt -lt "$tmo" ] && cat "$OUTDIR"/*"$tag"*.log 2>/dev/null | grep -q "TIDE-RS net\] rank .* FATAL"; then
    record "$tag" 0
  else
    echo "dc_fault $tag: rc=$rc after ${dt}s (deadline ${tmo}s)" >> "$OUTDIR/$tag.out"
    record "$tag" 1
  fi
}

offline() {
  local bin=$OUTDIR/test_waves rc
  ( cd "$REPO" && g++ -std=c++17 -O2 -Icsrc test/test_waves.cpp csrc/collective/rs/waves.cpp -o "$bin" && "$bin" ) \
      > "$OUTDIR/waves.out" 2>&1
  record waves $?
  ( cd "$REPO" && python3 -m py_compile tide/*.py test/*.py bench/*.py docs/figures/plot.py docs/figures/diagrams.py docs/figures/logo.py ) > "$OUTDIR/py_compile.out" 2>&1; rc=$?
  record py_compile $rc
  ( cd "$REPO" && for f in scripts/*.sh bench/*.sh; do bash -n "$f" || exit 1; done ) > "$OUTDIR/sh_syntax.out" 2>&1
  record sh_syntax $?
  # every tracked source / script carries the SPDX header
  ( cd "$REPO" && missing=$(git ls-files 'csrc/*' 'tide/*' 'test/*' 'bench/*' 'scripts/*' setup.py pyproject.toml docs/usage.md docs/overview.md docs/benchmarks.md docs/numerics.md docs/figures/plot.py docs/figures/diagrams.py docs/figures/logo.py \
        | xargs grep -L "SPDX-License-Identifier" || true); [ -z "$missing" ] || { echo "no SPDX header: $missing"; exit 1; } ) \
      > "$OUTDIR/spdx.out" 2>&1
  record spdx $?
}

single() {
  local F="TIDE_FAULT_RANK=1 TIDE_FAULT_US=300000"
  leg 1n_fsdp_compat   1 "" test/test_fsdp_compat.py
  leg 1n_ag            1 "" test/test_ag_bitwise.py --shard-mib 64 --segments 30 --iters 10
  leg 1n_ag_staged     1 "" test/test_ag_bitwise.py --shard-mib 64 --segments 30 --iters 10 --staged
  leg 1n_ag_nowrite    1 "TIDE_FAULT=nowrite" test/test_ag_bitwise.py --shard-mib 16 --segments 8 --iters 3
  leg 1n_fence         1 "" test/test_fence.py
  leg 1n_fence_burst   1 "" test/test_fence.py --pattern burst --lanes 3
  leg 1n_fence_conn1   1 "CUDA_DEVICE_MAX_CONNECTIONS=1" test/test_fence.py --pattern burst --lanes 3
  leg 1n_fence_nofence 1 "TIDE_FAULT=nofence" test/test_fence.py
  leg 1n_fence_nodone  1 "TIDE_FAULT=nodone,slow TIDE_FAULT_RANK=1 TIDE_FAULT_US=20000" test/test_fence.py
  leg 1n_concurrent    1 "" test/test_concurrent_rs.py --iters 200
  leg 1n_rs            1 "" test/test_rs_bitwise.py --shard-mib 32 --iters 3 --repeat 2
  leg 1n_rs_fp32       1 "" test/test_rs_bitwise.py --shard-mib 32 --iters 2 --repeat 2 --dtype fp32
  leg 1n_rs_plans3     1 "" test/test_rs_bitwise.py --shard-mib 16 --iters 2 --repeat 2 --plans 3
  leg 1n_rs_tuning     1 "" test/test_rs_tuning.py
  leg 1n_rs_invariance 1 "" test/test_rs_invariance.py
  leg 1n_rs_nonfinite  1 "" test/test_rs_nonfinite.py
  leg 1n_rs_nofence    1 "TIDE_FAULT=nofence,slowpush $F" test/test_rs_bitwise.py --shard-mib 32 --iters 2 --repeat 2
  leg 1n_rs_nofree     1 "TIDE_RS_CHUNK_MIB=4 TIDE_RS_STAGE_SLOTS=2 TIDE_FAULT=nofree,slowsum $F" test/test_rs_bitwise.py --shard-mib 32 --iters 2 --repeat 2
  leg 1n_rs_nowrite    1 "TIDE_FAULT=nowrite" test/test_rs_bitwise.py --shard-mib 16 --iters 2 --repeat 1
  leg 1n_fsdp          1 "" test/test_fsdp_adapter.py
  leg 1n_fsdp_rootauto 1 "" test/test_fsdp_adapter.py --root-auto --slots 2
  leg 1n_fsdp_unins    1 "" test/test_fsdp_adapter.py --uninstall-after 3
  leg 1n_fsdp_rs       1 "" test/test_fsdp_adapter.py --rs
  leg 1n_fsdp_slowopt  1 "" test/test_fsdp_adapter.py --slow-opt-ms 50
  leg 1n_fsdp_rs_slowopt 1 "" test/test_fsdp_adapter.py --rs --slow-opt-ms 50
  leg 1n_fsdp_reinst   1 "" test/test_fsdp_adapter.py --rs --uninstall-after 3 --reinstall
  leg 1n_fsdp_reinst_stock 1 "" test/test_fsdp_adapter.py --rs --no-mp --vocab 4099 --uninstall-after 2 --reinstall --stock-between 2
  leg 1n_abort         1 "TIDE_FAULT=abort" test/test_abort.py
  leg 1n_fsdp_accum    1 "" test/test_fsdp_adapter.py --rs --accum 3 --steps 3
  leg 1n_fsdp_accumag  1 "" test/test_fsdp_adapter.py --accum 2 --steps 3
  leg 1n_fsdp_nomp     1 "" test/test_fsdp_adapter.py --rs --no-mp --vocab 4099 --accum 2 --steps 4
}

multi() {
  [ "$N" -ge 2 ] || { echo "multi: need >= 2 hosts in $HOSTFILE (N=$N)"; fails=$((fails + 1)); return; }
  local n=${N}n F="TIDE_FAULT_RANK=9 TIDE_FAULT_US=300000"  # rank 9: second node, so the fault crosses the NIC
  leg ${n}_ag             "$N" "" test/test_ag_bitwise.py --shard-mib 32 --segments 30 --iters 10
  leg ${n}_ag_pace0       "$N" "TIDE_NIC_PACE=0" test/test_ag_bitwise.py --shard-mib 32 --segments 30 --iters 10
  leg ${n}_ag_lead0       "$N" "TIDE_NIC_PACE=1 TIDE_NIC_PACE_LEAD_MIB=0" test/test_ag_bitwise.py --shard-mib 32 --segments 30 --iters 10
  leg ${n}_ag_lead64      "$N" "TIDE_NIC_PACE=1 TIDE_NIC_PACE_LEAD_MIB=64" test/test_ag_bitwise.py --shard-mib 32 --segments 30 --iters 10
  leg ${n}_ag_soak        "$N" "" test/test_ag_bitwise.py --shard-mib 16 --segments 30 --iters 100
  leg ${n}_ag_nowrite     "$N" "TIDE_FAULT=nowrite" test/test_ag_bitwise.py --shard-mib 16 --segments 8 --iters 3
  leg ${n}_fence          "$N" "" test/test_fence.py
  leg ${n}_fence_burst    "$N" "" test/test_fence.py --pattern burst --lanes 3
  leg ${n}_fence_conn1    "$N" "CUDA_DEVICE_MAX_CONNECTIONS=1" test/test_fence.py --pattern burst --lanes 3
  leg ${n}_fence_nocredit "$N" "TIDE_FAULT=nocredit,slow TIDE_FAULT_RANK=8 TIDE_FAULT_US=20000" test/test_fence.py --pattern burst
  leg ${n}_fence_nocr_pace "$N" "TIDE_NIC_PACE=1 TIDE_FAULT=nocredit,slow TIDE_FAULT_RANK=8 TIDE_FAULT_US=20000" test/test_fence.py --pattern burst
  leg ${n}_concurrent     "$N" "" test/test_concurrent_rs.py --iters 200
  leg ${n}_rs             "$N" "" test/test_rs_bitwise.py --shard-mib 32 --iters 3 --repeat 2
  leg ${n}_rs128          "$N" "" test/test_rs_bitwise.py --shard-mib 128 --iters 2 --repeat 2
  leg ${n}_rs_k8          "$N" "TIDE_RS_KWAY=8" test/test_rs_bitwise.py --shard-mib 32 --iters 3 --repeat 2
  leg ${n}_rs_fp32        "$N" "" test/test_rs_bitwise.py --shard-mib 32 --dtype fp32 --iters 2 --repeat 2
  leg ${n}_rs_plans3      "$N" "" test/test_rs_bitwise.py --shard-mib 32 --iters 2 --repeat 2 --plans 3
  leg ${n}_rs_tuning      "$N" "" test/test_rs_tuning.py
  leg ${n}_rs_tuning_env  "$N" "TIDE_RS_KWAY=3 TIDE_RS_CTAS=3 TIDE_RS_PARTS=3" test/test_rs_tuning.py
  leg ${n}_rs_invariance  "$N" "" test/test_rs_invariance.py
  leg ${n}_rs_nonfinite   "$N" "" test/test_rs_nonfinite.py
  leg ${n}_W1_nonfinite   "$N" "TIDE_RS_WAVE_NODES=1" test/test_rs_nonfinite.py
  leg ${n}_rs_nowrite     "$N" "TIDE_FAULT=nowrite" test/test_rs_bitwise.py --shard-mib 32 --iters 2 --repeat 1
  leg ${n}_rs_nocredit    "$N" "TIDE_FAULT=nocredit,slow $F" test/test_rs_bitwise.py --shard-mib 32 --iters 2 --repeat 2
  leg ${n}_rs_conn1       "$N" "CUDA_DEVICE_MAX_CONNECTIONS=1" test/test_rs_bitwise.py --shard-mib 32 --iters 3 --repeat 2
  # Several landing waves on a small fleet (TIDE_RS_WAVE_NODES below the node count): parity reuse,
  # the own-only last wave, zeroing across waves and calls.
  local W
  for W in 1 2 3; do
    [ "$W" -lt "$N" ] || continue
    local w="TIDE_RS_WAVE_NODES=$W"
    leg ${n}_W${W}_rs        "$N" "$w" test/test_rs_bitwise.py --shard-mib 32 --iters 4 --repeat 2
    leg ${n}_W${W}_rs128     "$N" "$w" test/test_rs_bitwise.py --shard-mib 128 --iters 2 --repeat 2
    leg ${n}_W${W}_plans3    "$N" "$w" test/test_rs_bitwise.py --shard-mib 32 --iters 2 --repeat 2 --plans 3
    leg ${n}_W${W}_fsdp      "$N" "$w" test/test_fsdp_adapter.py --rs --accum 2 --steps 4
  done
  local w1="TIDE_RS_WAVE_NODES=1"
  leg ${n}_W1_nozero      "$N" "$w1 TIDE_FAULT=nozero" test/test_rs_bitwise.py --shard-mib 32 --iters 3 --repeat 1
  leg ${n}_W1_nodone      "$N" "$w1 TIDE_FAULT=nodone,slow $F" test/test_rs_bitwise.py --shard-mib 32 --iters 3 --repeat 2
  leg ${n}_W1_nofence     "$N" "$w1 TIDE_FAULT=nofence,slowpush $F" test/test_rs_bitwise.py --shard-mib 32 --iters 2 --repeat 2
  leg ${n}_W1_slow        "$N" "$w1 TIDE_FAULT=slow $F" test/test_rs_bitwise.py --shard-mib 32 --iters 2 --repeat 2
  leg ${n}_fsdp           "$N" "" test/test_fsdp_adapter.py
  leg ${n}_fsdp_rootauto  "$N" "" test/test_fsdp_adapter.py --root-auto --slots 2 --slow-rank 9 --slow-ms 5
  leg ${n}_fsdp_unins     "$N" "" test/test_fsdp_adapter.py --uninstall-after 3
  leg ${n}_fsdp_rs        "$N" "" test/test_fsdp_adapter.py --rs --accum 2 --steps 4
  leg ${n}_fsdp_frozen    "$N" "" test/test_fsdp_adapter.py --rs --frozen --fp32-group --ac --accum 2 --steps 4
  leg ${n}_fsdp_tail      "$N" "TIDE_RING_TAIL=1" test/test_fsdp_adapter.py --rs --frozen --fp32-group --ac --last-no-reshard --block-prefetch --accum 2 --steps 4
  leg ${n}_fsdp_rs_slow   "$N" "" test/test_fsdp_adapter.py --rs --slow-rank 9 --slow-ms 5
  leg ${n}_fsdp_slowopt   "$N" "" test/test_fsdp_adapter.py --slow-opt-ms 50
  leg ${n}_fsdp_tail_slowopt "$N" "TIDE_RING_TAIL=1" test/test_fsdp_adapter.py --rs --frozen --fp32-group --ac --last-no-reshard --block-prefetch --accum 2 --steps 4 --slow-opt-ms 50
  leg ${n}_fsdp_reinst    "$N" "" test/test_fsdp_adapter.py --rs --uninstall-after 3 --reinstall
  leg ${n}_fsdp_nomp      "$N" "" test/test_fsdp_adapter.py --rs --no-mp --vocab 4099 --accum 2 --steps 4
  leg ${n}_fsdp_nomp_reinst "$N" "" test/test_fsdp_adapter.py --rs --no-mp --vocab 4099 --uninstall-after 3 --reinstall
}

# Rail rounds of up to 8 blocks (8: >= 9 nodes at TIDE_RS_KWAY >= 8): the sum kernel against a host reference, bitwise, on one GPU
tma_rail() {
  local tag=1n_tma_rail bin=$OUTDIR/test_rs_tma_rail
  [ -n "${ONLY:-}" ] && ! [[ $tag =~ $ONLY ]] && return
  ( cd "$REPO" && "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" -std=c++17 -O2 -arch=native -Icsrc -I"${NCCL_HOME:-$REPO/../../build}/include" \
      test/test_rs_tma_rail.cu csrc/collective/rs/kernels.cu -o "$bin" && CUDA_VISIBLE_DEVICES=0 "$bin" ) \
      > "$OUTDIR/$tag.out" 2>&1
  record $tag $?
}

# The short gate: what a change to the reduce-scatter / adapter must pass, plus the multi-wave paths a
# fleet of >= 9 (incast) / 17 (full-wave parity reuse) nodes exercises at the default wave size
# (on <= 8 nodes forced with one-node waves, without which nozero has nothing to catch).
core1() {
  leg 1n_rs            1 "" test/test_rs_bitwise.py --shard-mib 32 --iters 3 --repeat 2
  leg 1n_rs_tuning     1 "" test/test_rs_tuning.py
  leg 1n_rs_invariance 1 "" test/test_rs_invariance.py
  leg 1n_rs_nonfinite  1 "" test/test_rs_nonfinite.py
  tma_rail
  leg 1n_fsdp_rs       1 "" test/test_fsdp_adapter.py --rs --accum 2 --steps 4
  leg 1n_fsdp_frozen   1 "" test/test_fsdp_adapter.py --rs --frozen --fp32-group --ac --accum 2 --steps 4
  leg 1n_fsdp_wide     1 "" test/test_fsdp_adapter.py --rs --fp32-group --wide 300 --accum 2 --steps 4
  leg 1n_fsdp_tail     1 "TIDE_RING_TAIL=1" test/test_fsdp_adapter.py --rs --frozen --fp32-group --ac --last-no-reshard --block-prefetch --accum 2 --steps 4
  leg 1n_fsdp_tailfb   1 "" test/test_fsdp_adapter.py --rs --last-no-reshard --reverse-forward --accum 2 --steps 4
  leg 1n_fsdp_tail_eval 1 "TIDE_RING_TAIL=1" test/test_fsdp_adapter.py --rs --frozen --fp32-group --ac --last-no-reshard --block-prefetch --accum 2 --steps 4 --eval-batches 3 --ema-swap
  leg 1n_fsdp_hsdp     1 "" test/test_fsdp_adapter.py --rs --replicate 2 --accum 2 --steps 4
  leg 1n_fsdp_arena_nccl 1 "TIDE_ARENA_TORCH=0" test/test_fsdp_adapter.py --rs --accum 2 --steps 4
  leg 1n_fsdp_env_fallback 1 "" test/test_fsdp_adapter.py --rs --env-problem-rank 3 --steps 2
  leg 1n_fsdp_nomp     1 "" test/test_fsdp_adapter.py --rs --no-mp --vocab 4099 --accum 2 --steps 4
  leg 1n_fsdp_reinst_stock 1 "" test/test_fsdp_adapter.py --rs --no-mp --vocab 4099 --uninstall-after 2 --reinstall --stock-between 2
  leg 1n_fsdp_slowopt  1 "" test/test_fsdp_adapter.py --slow-opt-ms 50
  leg 1n_fsdp_tail_slowopt 1 "TIDE_RING_TAIL=1" test/test_fsdp_adapter.py --rs --frozen --fp32-group --ac --last-no-reshard --block-prefetch --accum 2 --steps 4 --slow-opt-ms 50
  # copy-in (--copyin: the second run, bitwise against the first); the negative skips the slot's wait for the reads
  leg 1n_fsdp_copyin   1 "" test/test_fsdp_adapter.py --rs --copyin --poison --accum 2 --steps 4
  leg 1n_fsdp_copyin_tail_eval 1 "TIDE_RING_TAIL=1" test/test_fsdp_adapter.py --rs --copyin --frozen --fp32-group --ac --last-no-reshard --block-prefetch --accum 2 --steps 4 --eval-batches 3 --ema-swap
  leg 1n_fsdp_copyin_neg 1 "TIDE_FAULT=noreadwait,slowpush TIDE_FAULT_RANK=1 TIDE_FAULT_US=300000" test/test_fsdp_adapter.py --rs --copyin --poison --steps 3
  leg 1n_abort         1 "TIDE_FAULT=abort" test/test_abort.py
}
coremulti() {
  [ "$N" -ge 2 ] || { echo "coremulti: need >= 2 hosts (N=$N)"; fails=$((fails + 1)); return; }
  local n=${N}n F="TIDE_FAULT_RANK=9 TIDE_FAULT_US=300000" w=""
  [ "$N" -le 8 ] && w="TIDE_RS_WAVE_NODES=1"
  leg ${n}_ag             "$N" "" test/test_ag_bitwise.py --shard-mib 32 --segments 30 --iters 10
  leg ${n}_rs             "$N" "" test/test_rs_bitwise.py --shard-mib 32 --iters 3 --repeat 2
  leg ${n}_rs_k8          "$N" "TIDE_RS_KWAY=8" test/test_rs_bitwise.py --shard-mib 32 --iters 3 --repeat 2
  leg ${n}_rs_invariance  "$N" "" test/test_rs_invariance.py
  leg ${n}_rs_nonfinite   "$N" "" test/test_rs_nonfinite.py
  [ -n "$w" ] && leg ${n}_W1_rs "$N" "$w" test/test_rs_bitwise.py --shard-mib 32 --iters 4 --repeat 2
  leg ${n}_rs_nozero      "$N" "$w TIDE_FAULT=nozero" test/test_rs_bitwise.py --shard-mib 32 --iters 3 --repeat 1
  leg ${n}_rs_nodone      "$N" "$w TIDE_FAULT=nodone,slow $F" test/test_rs_bitwise.py --shard-mib 32 --iters 3 --repeat 2
  leg ${n}_fsdp_rs        "$N" "" test/test_fsdp_adapter.py --rs --accum 2 --steps 4
  leg ${n}_fsdp_frozen    "$N" "" test/test_fsdp_adapter.py --rs --frozen --fp32-group --ac --accum 2 --steps 4
  leg ${n}_fsdp_tail      "$N" "TIDE_RING_TAIL=1" test/test_fsdp_adapter.py --rs --frozen --fp32-group --ac --last-no-reshard --block-prefetch --accum 2 --steps 4
  leg ${n}_fsdp_tail_eval "$N" "TIDE_RING_TAIL=1" test/test_fsdp_adapter.py --rs --frozen --fp32-group --ac --last-no-reshard --block-prefetch --accum 2 --steps 4 --eval-batches 3 --ema-swap
  leg ${n}_fsdp_hsdp      "$N" "" test/test_fsdp_adapter.py --rs --replicate 2 --accum 2 --steps 4
  leg ${n}_fsdp_hsdp_tail "$N" "TIDE_RING_TAIL=1" test/test_fsdp_adapter.py --rs --replicate 2 --frozen --fp32-group --ac --last-no-reshard --block-prefetch --accum 2 --steps 4 --eval-batches 3 --ema-swap
  leg ${n}_fsdp_arena_nccl "$N" "TIDE_ARENA_TORCH=0" test/test_fsdp_adapter.py --rs --accum 2 --steps 4
  leg ${n}_fsdp_env_fallback "$N" "" test/test_fsdp_adapter.py --rs --env-problem-rank 9 --steps 2
  leg ${n}_fsdp_wide      "$N" "" test/test_fsdp_adapter.py --rs --fp32-group --wide 300 --accum 2 --steps 4
  leg ${n}_fsdp_hsdp_wide "$N" "" test/test_fsdp_adapter.py --rs --replicate 2 --fp32-group --wide 300 --accum 2 --steps 4
  leg ${n}_fsdp_nomp      "$N" "" test/test_fsdp_adapter.py --rs --no-mp --vocab 4099 --accum 2 --steps 4
  leg ${n}_fsdp_slowopt   "$N" "" test/test_fsdp_adapter.py --slow-opt-ms 50
  leg ${n}_fsdp_tail_slowopt "$N" "TIDE_RING_TAIL=1" test/test_fsdp_adapter.py --rs --frozen --fp32-group --ac --last-no-reshard --block-prefetch --accum 2 --steps 4 --slow-opt-ms 50
  leg ${n}_rs_outbox      "$N" "TIDE_RS_OUTBOX=64" test/test_rs_bitwise.py --shard-mib 32 --iters 3 --repeat 2
  leg ${n}_rs_slots4      "$N" "TIDE_RS_CHUNK_MIB=1 TIDE_RS_STAGE_SLOTS=4" test/test_rs_bitwise.py --shard-mib 128 --iters 2 --repeat 2
  leg ${n}_rs_nofree      "$N" "TIDE_RS_CHUNK_MIB=1 TIDE_FAULT=nofree,slowsum $F" test/test_rs_bitwise.py --shard-mib 128 --iters 2 --repeat 2
  # without expert groups the ring slots share one receive stage by default (an all-gather writing it before the
  # previous one scattered out shows bitwise in the legs above); one stage per slot, the default with expert groups:
  local sh="TIDE_FSDP_SHARED_STAGE=0"
  leg ${n}_fsdp_sepstage  "$N" "$sh" test/test_fsdp_adapter.py --accum 2 --steps 4
  leg ${n}_fsdp_sepstage_slow "$N" "$sh" test/test_fsdp_adapter.py --slow-rank 9 --slow-ms 5
  leg ${n}_fsdp_sepstage_tail "$N" "$sh TIDE_RING_TAIL=1" test/test_fsdp_adapter.py --rs --frozen --fp32-group --ac --last-no-reshard --block-prefetch --accum 2 --steps 4
  leg ${n}_fsdp_sepstage_hsdp "$N" "$sh" test/test_fsdp_adapter.py --replicate 2 --accum 2 --steps 4
  leg ${n}_fsdp_shstage_slow "$N" "" test/test_fsdp_adapter.py --slow-rank 9 --slow-ms 5
  # passes over the sub-slices (rs_invariance checks them against one pass bitwise): with one wave per
  # part, part q + 2 reuses part q's parity (nodone has something to catch); two- / three-node waves add
  # the zeroing across parts, an odd wave count shifting the parities from part to part
  local pa="TIDE_RS_PARTS=4"
  leg ${n}_parts_rs       "$N" "$pa" test/test_rs_bitwise.py --shard-mib 32 --iters 3 --repeat 2
  leg ${n}_parts_rs128    "$N" "$pa" test/test_rs_bitwise.py --shard-mib 128 --iters 2 --repeat 2
  leg ${n}_parts_plans3   "$N" "$pa" test/test_rs_bitwise.py --shard-mib 32 --iters 2 --repeat 2 --plans 3
  leg ${n}_parts_nodone   "$N" "$pa TIDE_FAULT=nodone,slow $F" test/test_rs_bitwise.py --shard-mib 32 --iters 3 --repeat 2
  leg ${n}_parts_nocredit "$N" "$pa TIDE_FAULT=nocredit,slow $F" test/test_rs_bitwise.py --shard-mib 32 --iters 2 --repeat 2
  leg ${n}_parts_tuning   "$N" "TIDE_RS_PARTS=3" test/test_rs_tuning.py
  leg ${n}_parts_fsdp     "$N" "$pa" test/test_fsdp_adapter.py --rs --accum 2 --steps 4
  leg ${n}_parts_fsdp_hsdp "$N" "$pa" test/test_fsdp_adapter.py --rs --replicate 2 --accum 2 --steps 4
  if [ "$N" -ge 4 ] && [ $((N % 2)) -ne 1 ]; then  # no wave of the own node alone (repeatWaves would refuse)
    leg ${n}_parts_W2_rs     "$N" "$pa TIDE_RS_WAVE_NODES=2" test/test_rs_bitwise.py --shard-mib 32 --iters 4 --repeat 2
    leg ${n}_parts_W2_nozero "$N" "$pa TIDE_RS_WAVE_NODES=2 TIDE_FAULT=nozero" test/test_rs_bitwise.py --shard-mib 32 --iters 3 --repeat 1
  fi
  if [ "$N" -ge 5 ] && [ $((N % 3)) -ne 1 ]; then
    leg ${n}_parts_W3_rs     "$N" "TIDE_RS_PARTS=3 TIDE_RS_WAVE_NODES=3" test/test_rs_bitwise.py --shard-mib 32 --iters 4 --repeat 2
  fi
  # pipelined calls (read phase, then network phase; two outbox / own-spare banks): back-to-back calls overlap, so the
  # negatives also catch a bank or a spare reused too early; the adapter legs compare them bitwise with one-phase calls
  local pp="TIDE_RS_PIPELINE=2"
  leg ${n}_pipe_rs        "$N" "$pp" test/test_rs_bitwise.py --shard-mib 32 --iters 3 --repeat 2
  leg ${n}_pipe_outbox    "$N" "$pp TIDE_RS_OUTBOX=2" test/test_rs_bitwise.py --shard-mib 32 --iters 3 --repeat 2
  leg ${n}_pipe_noputwait "$N" "$pp TIDE_RS_OUTBOX=2 TIDE_FAULT=noputwait,slow $F" test/test_rs_bitwise.py --shard-mib 32 --iters 2 --repeat 2
  [ -n "$w" ] && leg ${n}_pipe_W1_rs "$N" "$pp $w" test/test_rs_bitwise.py --shard-mib 32 --iters 4 --repeat 2
  leg ${n}_pipe_parts_rs  "$N" "$pp $pa" test/test_rs_bitwise.py --shard-mib 32 --iters 3 --repeat 2
  leg ${n}_pipe_invariance "$N" "$pp" test/test_rs_invariance.py
  leg ${n}_pipe_nonfinite "$N" "$pp" test/test_rs_nonfinite.py
  leg ${n}_pipe_nodone    "$N" "$pp $w TIDE_FAULT=nodone,slow $F" test/test_rs_bitwise.py --shard-mib 32 --iters 3 --repeat 2
  leg ${n}_pipe_nocredit  "$N" "$pp $pa TIDE_FAULT=nocredit,slow $F" test/test_rs_bitwise.py --shard-mib 32 --iters 2 --repeat 2
  leg ${n}_pipe_nozero    "$N" "$pp $w TIDE_FAULT=nozero" test/test_rs_bitwise.py --shard-mib 32 --iters 3 --repeat 1
  leg ${n}_pipe_nofree    "$N" "$pp TIDE_RS_CHUNK_MIB=1 TIDE_FAULT=nofree,slowsum $F" test/test_rs_bitwise.py --shard-mib 128 --iters 2 --repeat 2
  leg ${n}_fsdp_async     "$N" "" test/test_fsdp_adapter.py --rs --poison --accum 2 --steps 4
  leg ${n}_fsdp_async_merged "$N" "" test/test_fsdp_adapter.py --rs --merged-stream --poison --accum 2 --steps 4
  [ -n "$w" ] && leg ${n}_fsdp_async_W1 "$N" "$w" test/test_fsdp_adapter.py --rs --merged-stream --poison --accum 2 --steps 4
  leg ${n}_fsdp_async_parts "$N" "$pa TIDE_RS_KWAY=2" test/test_fsdp_adapter.py --rs --merged-stream --poison --accum 2 --steps 4
  leg ${n}_fsdp_async_slow "$N" "TIDE_FAULT=slow $F" test/test_fsdp_adapter.py --rs --merged-stream --poison --steps 3
  leg ${n}_fsdp_async_slowsum "$N" "TIDE_FAULT=slowsum $F" test/test_fsdp_adapter.py --rs --poison --steps 3
  leg ${n}_fsdp_async_neg "$N" "TIDE_FAULT=earlyread,slowpush $F" test/test_fsdp_adapter.py --rs --merged-stream --poison --steps 3
  leg ${n}_fsdp_async_hsdp "$N" "" test/test_fsdp_adapter.py --rs --replicate 2 --poison --accum 2 --steps 4
  leg ${n}_fsdp_async_tail_eval "$N" "TIDE_RING_TAIL=1" test/test_fsdp_adapter.py --rs --merged-stream --frozen --fp32-group --ac --last-no-reshard --accum 2 --steps 4 --eval-batches 3 --ema-swap
  leg ${n}_fsdp_async_bp  "$N" "TIDE_RING_TAIL=1" test/test_fsdp_adapter.py --rs --block-prefetch --ac --last-no-reshard --accum 2 --steps 4
  leg ${n}_fsdp_async_2l  "$N" "" test/test_fsdp_adapter.py --rs --merged-stream --layers 2 --poison --accum 2 --steps 4
  leg ${n}_fsdp_async_split "$N" "" test/test_fsdp_adapter.py --rs --split-comm --poison --accum 2 --steps 4
  # TIDE_RS_NET=device: the same plans and sums with the network phase in GIN device kernels (bitwise), the negatives
  # still caught through the device-side credit / waveDone waits
  local dc="$pp TIDE_RS_NET=device"
  leg ${n}_dc_rs          "$N" "$dc" test/test_rs_bitwise.py --shard-mib 32 --iters 3 --repeat 2
  leg ${n}_dc_outbox      "$N" "$dc TIDE_RS_OUTBOX=2 TIDE_RS_NET_TRACE=2" test/test_rs_bitwise.py --shard-mib 32 --iters 3 --repeat 2
  [ -n "$w" ] && leg ${n}_dc_W1_rs "$N" "$dc $w" test/test_rs_bitwise.py --shard-mib 32 --iters 4 --repeat 2
  leg ${n}_dc_parts_rs    "$N" "$dc $pa" test/test_rs_bitwise.py --shard-mib 32 --iters 3 --repeat 2
  leg ${n}_dc_parts_k2_rs "$N" "$dc $pa TIDE_RS_KWAY=2" test/test_rs_bitwise.py --shard-mib 32 --iters 3 --repeat 2
  leg ${n}_dc_invariance  "$N" "$dc" test/test_rs_invariance.py
  leg ${n}_dc_nonfinite   "$N" "$dc" test/test_rs_nonfinite.py
  leg ${n}_dc_nodone      "$N" "$dc $w TIDE_FAULT=nodone,slow $F" test/test_rs_bitwise.py --shard-mib 32 --iters 3 --repeat 2
  leg ${n}_dc_nocredit    "$N" "$dc $pa TIDE_FAULT=nocredit,slow $F" test/test_rs_bitwise.py --shard-mib 32 --iters 2 --repeat 2
  leg ${n}_dc_nozero      "$N" "$dc $w TIDE_FAULT=nozero" test/test_rs_bitwise.py --shard-mib 32 --iters 3 --repeat 1
  leg ${n}_fsdp_async_dc  "$N" "TIDE_RS_NET=device" test/test_fsdp_adapter.py --rs --merged-stream --poison --accum 2 --steps 4
  leg ${n}_fsdp_async_dc_parts "$N" "TIDE_RS_NET=device $pa TIDE_RS_KWAY=2" test/test_fsdp_adapter.py --rs --merged-stream --poison --accum 2 --steps 4
  leg ${n}_fsdp_async_dc_hsdp "$N" "TIDE_RS_NET=device" test/test_fsdp_adapter.py --rs --replicate 2 --poison --accum 2 --steps 4
  leg ${n}_fsdp_async_dc_tail_eval "$N" "TIDE_RS_NET=device TIDE_RING_TAIL=1" test/test_fsdp_adapter.py --rs --merged-stream --frozen --fp32-group --ac --last-no-reshard --accum 2 --steps 4 --eval-batches 3 --ema-swap
  # copy-in (--copyin: the second run, bitwise against the first, no early all-gather), also with the memory-mode
  # sizing (device network phase, 2 sum CTAs, 1 landing CTA); the negative skips the slot's wait for the reads, on two
  # communicators: on one, NCCL orders the next all-gather's credit wait behind this reduce-scatter's host network phase,
  # which waits for every local rank's reads, so the skipped wait shows only by timing luck
  leg ${n}_fsdp_copyin    "$N" "" test/test_fsdp_adapter.py --rs --copyin --poison --accum 2 --steps 4
  leg ${n}_fsdp_copyin_merged "$N" "" test/test_fsdp_adapter.py --rs --copyin --merged-stream --poison --accum 2 --steps 4
  leg ${n}_fsdp_copyin_hsdp "$N" "" test/test_fsdp_adapter.py --rs --copyin --replicate 2 --poison --accum 2 --steps 4
  leg ${n}_fsdp_copyin_tail_eval "$N" "TIDE_RING_TAIL=1" test/test_fsdp_adapter.py --rs --copyin --merged-stream --frozen --fp32-group --ac --last-no-reshard --accum 2 --steps 4 --eval-batches 3 --ema-swap
  leg ${n}_fsdp_copyin_dc "$N" "TIDE_RS_NET=device TIDE_RS_CTAS=2 TIDE_RS_MC_CTAS=1" test/test_fsdp_adapter.py --rs --copyin --merged-stream --poison --accum 2 --steps 4
  leg ${n}_fsdp_copyin_neg "$N" "TIDE_FAULT=noreadwait,slowpush $F" test/test_fsdp_adapter.py --rs --copyin --poison --steps 3 --split-comm
  # the fail-stop path: one rank stops launching the device network kernels from its second call; every peer's wait times
  # out, that kernel never returns (no completion event), the monitor thread exits the process and the job ends in time
  dc_fault ${n}_dc_stall "$N" "$dc TIDE_FAULT=dcstall TIDE_FAULT_RANK=9 TIDE_RS_NET_TIMEOUT_S=20" test/test_rs_bitwise.py --shard-mib 32 --iters 3 --repeat 2
}

# Expert groups: TIDE over the subgroups r % ep (FSDP's dp_shard_mod_ep); ep = 8 is one rank per node
# (direct all-gather, rail reduce-scatter), 4 / 2 the regular multi-node algorithm on 2 / 4 ranks per node.
ep() {
  [ "$N" -ge 2 ] || { echo "ep: need >= 2 hosts (N=$N)"; fails=$((fails + 1)); return; }
  local n=${N}n e F="TIDE_FAULT_RANK=1 TIDE_FAULT_US=300000"  # comm rank 1: the second node of every subgroup
  for e in 8 4 2; do
    leg ${n}_ep${e}_ag       "$N" "" test/test_ag_bitwise.py --ep $e --shard-mib 32 --segments 30 --iters 10
    leg ${n}_ep${e}_ag_own   "$N" "" test/test_ag_bitwise.py --ep $e --shard-mib 32 --segments 30 --iters 10 --own-sep
    leg ${n}_ep${e}_fence    "$N" "" test/test_fence.py --ep $e --pattern burst --lanes 3
    leg ${n}_ep${e}_rs       "$N" "" test/test_rs_bitwise.py --ep $e --shard-mib 32 --iters 3 --repeat 2
    leg ${n}_ep${e}_rs_fp32  "$N" "" test/test_rs_bitwise.py --ep $e --shard-mib 32 --dtype fp32 --iters 2 --repeat 2
    leg ${n}_ep${e}_plans3   "$N" "" test/test_rs_bitwise.py --ep $e --shard-mib 32 --iters 2 --repeat 2 --plans 3
    leg ${n}_ep${e}_fsdp     "$N" "" test/test_fsdp_adapter.py --ep $e --rs --accum 2 --steps 4
    leg ${n}_ep${e}_fsdp_div "$N" "" test/test_fsdp_adapter.py --ep $e --rs --divide-factor $((N * 8))
    leg ${n}_ep${e}_fsdp_frozen "$N" "" test/test_fsdp_adapter.py --ep $e --rs --frozen --block-prefetch --fp32-group --ac
    leg ${n}_ep${e}_fsdp_tail "$N" "TIDE_RING_TAIL=1" test/test_fsdp_adapter.py --ep $e --rs --frozen --block-prefetch --fp32-group --ac --last-no-reshard
  done
  leg ${n}_ep8_ag_soak       "$N" "" test/test_ag_bitwise.py --ep 8 --shard-mib 16 --segments 30 --iters 100
  leg ${n}_ep8_rs128         "$N" "" test/test_rs_bitwise.py --ep 8 --shard-mib 128 --iters 2 --repeat 2
  leg ${n}_ep8_rs_k8         "$N" "TIDE_RS_KWAY=8" test/test_rs_bitwise.py --ep 8 --shard-mib 32 --iters 3 --repeat 2
  leg ${n}_ep8_ag_nowrite    "$N" "TIDE_FAULT=nowrite" test/test_ag_bitwise.py --ep 8 --shard-mib 16 --segments 8 --iters 3
  leg ${n}_ep8_fence_nofence "$N" "TIDE_FAULT=nofence" test/test_fence.py --ep 8
  leg ${n}_ep8_rs_nowrite    "$N" "TIDE_FAULT=nowrite" test/test_rs_bitwise.py --ep 8 --shard-mib 32 --iters 2 --repeat 1
  leg ${n}_ep8_rs_nocredit   "$N" "TIDE_FAULT=nocredit,slow $F" test/test_rs_bitwise.py --ep 8 --shard-mib 32 --iters 2 --repeat 2
  leg ${n}_ep8_fsdp_reinst   "$N" "" test/test_fsdp_adapter.py --ep 8 --rs --uninstall-after 3 --reinstall
  leg ${n}_ep8_fsdp_nomp     "$N" "" test/test_fsdp_adapter.py --ep 8 --rs --no-mp --vocab 4099 --accum 2 --steps 4
  leg ${n}_ep8_fsdp_split    "$N" "" test/test_fsdp_adapter.py --ep 8 --rs --split-comm
  leg ${n}_ep8_fsdp_ag       "$N" "" test/test_fsdp_adapter.py --ep 8
  leg ${n}_ep8_fsdp_tail_eval "$N" "TIDE_RING_TAIL=1" test/test_fsdp_adapter.py --ep 8 --rs --frozen --block-prefetch --fp32-group --ac --last-no-reshard --eval-batches 3 --ema-swap
  leg ${n}_ep8_fsdp_hsdp     "$N" "TIDE_RING_TAIL=1" test/test_fsdp_adapter.py --ep 8 --replicate 2 --rs --frozen --block-prefetch --fp32-group --ac --last-no-reshard
  leg ${n}_ep2_fsdp_shstage  "$N" "TIDE_FSDP_SHARED_STAGE=1" test/test_fsdp_adapter.py --ep 2 --block-prefetch --accum 2 --steps 4
  # the device network phase over expert subgroups: the rail lane (ep 8) stays one-phase, the multi-rank one (ep 2)
  # runs pipelined with device kernels too
  leg ${n}_ep8_fsdp_async_dc "$N" "TIDE_RS_NET=device" test/test_fsdp_adapter.py --ep 8 --rs --merged-stream --poison --accum 2 --steps 4
  leg ${n}_ep2_fsdp_async_dc "$N" "TIDE_RS_NET=device" test/test_fsdp_adapter.py --ep 2 --rs --merged-stream --poison --accum 2 --steps 4
  # copy-in on the expert lanes too: the rail lane (one-phase) and a pipelined one
  leg ${n}_ep8_fsdp_copyin   "$N" "" test/test_fsdp_adapter.py --ep 8 --rs --copyin --poison --accum 2 --steps 4
  leg ${n}_ep2_fsdp_copyin   "$N" "" test/test_fsdp_adapter.py --ep 2 --rs --copyin --poison --accum 2 --steps 4
  leg ${n}_ep2_dc_rs         "$N" "TIDE_RS_PIPELINE=2 TIDE_RS_NET=device" test/test_rs_bitwise.py --ep 2 --shard-mib 32 --iters 3 --repeat 2
  # passes over the sub-slices on 4 / 2 ranks per node; the rail lane (ep 8) has its own parts
  leg ${n}_ep2_parts_rs      "$N" "TIDE_RS_PARTS=4" test/test_rs_bitwise.py --ep 2 --shard-mib 32 --iters 3 --repeat 2
  leg ${n}_ep4_parts_fsdp    "$N" "TIDE_RS_PARTS=3" test/test_fsdp_adapter.py --ep 4 --rs --accum 2 --steps 4
}

# Decoupled all-gathers (the FSDP2 adapter's only kind; test_ag_bitwise / test_fence take --decouple): the release
# gates `ready`, the exit wait leaves the caller's stream. One connection puts every stream on one hardware queue:
# aliasing may serialize, never deadlock.
decouple() {
  local c1="CUDA_DEVICE_MAX_CONNECTIONS=1"
  local T="--rs --frozen --fp32-group --ac --last-no-reshard --block-prefetch --accum 2 --steps 4"
  leg 1n_dec_ag            1 "" test/test_ag_bitwise.py --decouple --shard-mib 64 --segments 30 --iters 10
  leg 1n_dec_fence         1 "" test/test_fence.py --decouple
  leg 1n_dec_fence_burst   1 "" test/test_fence.py --decouple --pattern burst --lanes 3
  leg 1n_dec_fence_conn1   1 "$c1" test/test_fence.py --decouple --pattern burst --lanes 3
  leg 1n_dec_fence_nofence 1 "TIDE_FAULT=nofence" test/test_fence.py --decouple
  leg 1n_dec_fence_nodone  1 "TIDE_FAULT=nodone,slow TIDE_FAULT_RANK=1 TIDE_FAULT_US=20000" test/test_fence.py --decouple
  leg 1n_dec_fence_nofree  1 "TIDE_FAULT=nofree" test/test_fence.py --decouple
  leg 1n_dec_fsdp          1 "" test/test_fsdp_adapter.py --rs --accum 2 --steps 4
  leg 1n_dec_fsdp_tail     1 "TIDE_RING_TAIL=1" test/test_fsdp_adapter.py $T
  leg 1n_dec_fsdp_slowopt  1 "" test/test_fsdp_adapter.py --slow-opt-ms 50
  leg 1n_dec_fsdp_reinst   1 "" test/test_fsdp_adapter.py --rs --uninstall-after 3 --reinstall
  leg 1n_dec_fsdp_conn1    1 "$c1" test/test_fsdp_adapter.py $T
  [ "$N" -ge 2 ] || return
  local n=${N}n e
  leg ${n}_dec_ag          "$N" "" test/test_ag_bitwise.py --decouple --shard-mib 32 --segments 30 --iters 10
  leg ${n}_dec_ag_soak     "$N" "" test/test_ag_bitwise.py --decouple --shard-mib 16 --segments 30 --iters 100
  leg ${n}_dec_fence       "$N" "" test/test_fence.py --decouple
  leg ${n}_dec_fence_burst "$N" "" test/test_fence.py --decouple --pattern burst --lanes 3
  leg ${n}_dec_fence_conn1 "$N" "$c1" test/test_fence.py --decouple --pattern burst --lanes 3
  leg ${n}_dec_fence_nocredit "$N" "TIDE_FAULT=nocredit,slow TIDE_FAULT_RANK=8 TIDE_FAULT_US=20000" test/test_fence.py --decouple --pattern burst
  leg ${n}_dec_fence_nodone "$N" "TIDE_FAULT=nodone,slow TIDE_FAULT_RANK=9 TIDE_FAULT_US=20000" test/test_fence.py --decouple
  leg ${n}_dec_fence_nofree "$N" "TIDE_FAULT=nofree" test/test_fence.py --decouple
  leg ${n}_dec_fsdp        "$N" "" test/test_fsdp_adapter.py --rs --accum 2 --steps 4
  leg ${n}_dec_fsdp_rootauto "$N" "" test/test_fsdp_adapter.py --root-auto --slots 2 --slow-rank 9 --slow-ms 5
  leg ${n}_dec_fsdp_tail   "$N" "TIDE_RING_TAIL=1" test/test_fsdp_adapter.py $T
  leg ${n}_dec_fsdp_rs_slow "$N" "" test/test_fsdp_adapter.py --rs --slow-rank 9 --slow-ms 5
  leg ${n}_dec_fsdp_tail_slowopt "$N" "TIDE_RING_TAIL=1" test/test_fsdp_adapter.py $T --slow-opt-ms 50
  leg ${n}_dec_fsdp_reinst "$N" "" test/test_fsdp_adapter.py --rs --uninstall-after 3 --reinstall
  leg ${n}_dec_fsdp_conn1  "$N" "$c1" test/test_fsdp_adapter.py $T
  for e in 8 4; do
    leg ${n}_dec_ep${e}_fence "$N" "" test/test_fence.py --decouple --ep $e --pattern burst --lanes 3
    leg ${n}_dec_ep${e}_fsdp_tail "$N" "TIDE_RING_TAIL=1" test/test_fsdp_adapter.py --ep $e --rs --frozen --block-prefetch --fp32-group --ac --last-no-reshard
  done
  # decoupled all-gathers with the pipelined reduce-scatter's device network phase: plain / parts / the K1 trace /
  # validation + EMA swap
  local K="TIDE_RS_NET=device" A="--rs --merged-stream --poison --accum 2 --steps 4"
  leg ${n}_dec_fsdp_async_dc       "$N" "$K" test/test_fsdp_adapter.py $A
  leg ${n}_dec_fsdp_async_dc_parts "$N" "$K TIDE_RS_PARTS=4 TIDE_RS_KWAY=2" test/test_fsdp_adapter.py $A
  leg ${n}_dec_fsdp_async_dc_trace "$N" "$K TIDE_RS_PARTS=4 TIDE_RS_KWAY=2 TIDE_RS_NET_TRACE=4" test/test_fsdp_adapter.py $A
  leg ${n}_dec_fsdp_async_dc_eval  "$N" "$K TIDE_RING_TAIL=1" test/test_fsdp_adapter.py --rs --merged-stream --frozen --fp32-group --ac --last-no-reshard --accum 2 --steps 4 --eval-batches 3 --ema-swap
}

echo "gate $STAGE: repo $(git -C "$REPO" rev-parse --short HEAD) N=$N out=$OUTDIR" | tee -a "$SUMMARY"
case "$STAGE" in
  offline) offline ;;
  1) single ;;
  multi) multi ;;
  all) offline; single; multi ;;
  core1) core1 ;;
  coremulti) coremulti ;;
  ep) ep ;;
  decouple) decouple ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
echo "gate $STAGE: $fails failed ($SUMMARY)" | tee -a "$SUMMARY"
[ "$fails" -eq 0 ]
