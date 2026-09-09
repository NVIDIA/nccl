#!/bin/bash
#
# NIIN comprehensive test suite — functional + performance.
# Produces a CSV with all results for cross-platform comparison.
#
# Usage:
#   ./run_all.sh [--p2p] [--ib] [--all] [--build-only] [--skip-perf]
#
# Modes:
#   --p2p        Run with P2P/LSA (default GPU 0,1 same NUMA)
#   --ib         Run with IB loopback (disable P2P, force network)
#   --all        Run both modes (default)
#
# Env vars:
#   NCCL_HOME      Path to NCCL repo root
#   NVSHMEM_HOME   Path to NVSHMEM source (for compat tests)
#   MPI_HOME       Path to MPI install
#   ARCH           GPU arch (auto-detected if not set)
#   HCA            IB HCA for loopback (default: first active mlx5)
#   GPU_PAIR       GPU pair to use (default: "0,1")
#   NP             Number of PEs (default: 2)
#   RESULTS_DIR    Output directory (default: results/)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NIIN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NCCL_HOME="${NCCL_HOME:-$(cd "$NIIN_DIR/../.." && pwd)}"
NVSHMEM_HOME="${NVSHMEM_HOME:-/storage/benjaming/repo/nvshmem}"
MPI_HOME="${MPI_HOME:-/cm/shared/apps/openmpi4/gcc/4.1.8}"
GPU_PAIR="${GPU_PAIR:-0,1}"
NP="${NP:-2}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results}"
TIMEOUT_FUNC=60
TIMEOUT_PERF=120
RUN_P2P=false
RUN_IB=false
BUILD_ONLY=false
SKIP_PERF=false
SKIP_FUNC=false

# Auto-detect arch
if [ -z "${ARCH:-}" ]; then
  gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  case "$gpu_name" in
    *H100*|*H200*) ARCH=sm_90 ;;
    *L40*|*L4*|*RTX*4*) ARCH=sm_89 ;;
    *A100*|*A30*) ARCH=sm_80 ;;
    *V100*) ARCH=sm_70 ;;
    *) ARCH=sm_89; echo "Warning: unknown GPU '$gpu_name', defaulting to $ARCH" ;;
  esac
fi

# Auto-detect HCA
if [ -z "${HCA:-}" ]; then
  HCA=$(ibstat 2>/dev/null | grep -B1 "State: Active" | grep "^CA" | head -1 | awk '{print $2}' | tr -d "'")
  if [ -z "$HCA" ]; then HCA="mlx5_1"; fi
fi

for arg in "$@"; do
  case $arg in
    --p2p) RUN_P2P=true ;;
    --ib) RUN_IB=true ;;
    --all) RUN_P2P=true; RUN_IB=true ;;
    --build-only) BUILD_ONLY=true ;;
    --skip-perf) SKIP_PERF=true ;;
    --skip-func) SKIP_FUNC=true ;;
  esac
done

# Default: run both
if ! $RUN_P2P && ! $RUN_IB; then
  RUN_P2P=true
  RUN_IB=true
fi

mkdir -p "$RESULTS_DIR"

NVCC_FLAGS="
  -I $NIIN_DIR/test/compat
  -I $NIIN_DIR/include
  -I $NCCL_HOME/build/include
  -I $MPI_HOME/include
  -L $NCCL_HOME/build/lib -lnccl
  -L $MPI_HOME/lib -lmpi
  --expt-relaxed-constexpr -std=c++17 -arch=$ARCH
  -DNIIN_HAS_MPI
  -lcuda -lcudart -lrt
  -Xlinker -rpath,$NCCL_HOME/build/lib
  -Xlinker -rpath,$MPI_HOME/lib"

MPI_RUN="mpirun -np $NP --allow-run-as-root"
BUILD_DIR="/tmp/niin_test_suite"

# ===========================================================================
# System info
# ===========================================================================
echo "=== NIIN Test Suite ==="
echo "Date:       $(date -Iseconds)"
echo "Host:       $(hostname)"
echo "NCCL:       $NCCL_HOME"
echo "NVSHMEM:    $NVSHMEM_HOME"
echo "MPI:        $MPI_HOME"
echo "Arch:       $ARCH"
echo "GPU pair:   $GPU_PAIR"
echo "HCA:        $HCA"
echo "Results:    $RESULTS_DIR"
echo

# System info CSV
cat > "$RESULTS_DIR/system_info.csv" << SYSEOF
key,value
hostname,$(hostname)
date,$(date -Iseconds)
gpu,$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
gpu_count,$(nvidia-smi -L | wc -l)
driver,$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
cuda,$(nvcc --version 2>&1 | grep "release" | sed 's/.*release //' | sed 's/,.*//')
arch,$ARCH
gpu_pair,$GPU_PAIR
hca,$HCA
nccl_version,$(grep "NCCL_VERSION" $NCCL_HOME/build/include/nccl.h 2>/dev/null | head -1 | sed 's/.*NCCL_VERSION(//' | sed 's/)//')
p2p_topology,$(nvidia-smi topo -m 2>/dev/null | head -6 | tail -4 | tr '\t' ',' || echo "unavailable")
SYSEOF

# ===========================================================================
# XFAIL lists
# ===========================================================================
# P2P mode: atomics with specific type variants, multi-subtest hangs
XFAIL_P2P="atomic_add atomic_and atomic_inc atomic_or atomic_swap atomic_xor
           g_extended
           test_any test_some wait_until_any wait_until_some
           put_signal wait
           info"

# IB mode: all the above plus atomics and scalar/strided gets over network
XFAIL_IB="$XFAIL_P2P
          atomic_add atomic_and atomic_compare_swap atomic_fetch atomic_set
          g iget
          ptr put_signal_nbi
          sync_test test_all_any_some wait_until wait
          signal_add"

is_xfail() {
  local name="$1" list="$2"
  for xf in $list; do
    if [ "$name" = "$xf" ]; then return 0; fi
  done
  return 1
}

# ===========================================================================
# Build
# ===========================================================================
echo "=== Building tests ==="
mkdir -p "$BUILD_DIR"

# Discover NVSHMEM tests
NVSHMEM_TESTS=()
if [ -d "$NVSHMEM_HOME/test" ]; then
  for f in "$NVSHMEM_HOME"/test/device/query/*.cu \
           "$NVSHMEM_HOME"/test/device/sync/*.cu; do
    [ -f "$f" ] && NVSHMEM_TESTS+=("$f")
  done
  for f in "$NVSHMEM_HOME"/test/device/pt-to-pt/*.cu; do
    [ -f "$f" ] || continue
    grep -q "ring_alltoall.h\|data_check.h\|perf_utils.h" "$f" 2>/dev/null && continue
    echo "$(basename ${f%.cu})" | grep -q "qp_specific" && continue
    NVSHMEM_TESTS+=("$f")
  done
fi

# Build functional tests
BUILT_FUNC=()
echo "test,build_status,build_error" > "$RESULTS_DIR/build_results.csv"
for src in "${NVSHMEM_TESTS[@]}"; do
  name=$(basename ${src%.cu})
  output=$(nvcc "$src" -o "$BUILD_DIR/$name" $NVCC_FLAGS 2>&1)
  if [ $? -eq 0 ]; then
    echo -n "."
    echo "$name,OK," >> "$RESULTS_DIR/build_results.csv"
    BUILT_FUNC+=("$name")
  else
    echo -n "x"
    err=$(echo "$output" | head -1 | tr ',' ';' | cut -c1-200)
    echo "$name,FAIL,$err" >> "$RESULTS_DIR/build_results.csv"
  fi
done
echo
echo "Built ${#BUILT_FUNC[@]} of ${#NVSHMEM_TESTS[@]} functional tests"

# Build perftests
make -C "$NIIN_DIR/perftest" \
  NCCL_HOME="$NCCL_HOME" MPI_HOME="$MPI_HOME" ARCH="$ARCH" \
  clean all 2>&1 | tail -1
echo

if $BUILD_ONLY; then
  echo "Build complete. Exiting."
  exit 0
fi

# ===========================================================================
# Run functional tests
# ===========================================================================
run_func_tests() {
  local mode="$1" env_prefix="$2" xfail_list="$3" csv="$4"
  echo "=== Functional tests ($mode) ==="
  echo "test,mode,status,detail" > "$csv"

  local pass=0 xpass=0 xfail=0 fail=0
  for name in "${BUILT_FUNC[@]}"; do
    output=$(timeout $TIMEOUT_FUNC env CUDA_VISIBLE_DEVICES=$GPU_PAIR $env_prefix \
             $MPI_RUN "$BUILD_DIR/$name" 2>&1)
    rc=$?
    errors=$(echo "$output" | grep -i "error\|incorrect\|FAIL\|trap\|Segmentation" | \
             grep -v "mca_\|btl_\|orte_\|help-mpi\|MCA\|OpenFabrics\|device params\|aggregate\|cpcs\|pstat" | head -1)
    detail=$(echo "$errors" | tr ',' ';' | cut -c1-200)

    if [ $rc -eq 0 ] && [ -z "$errors" ]; then
      if is_xfail "$name" "$xfail_list"; then
        echo "  $name: XPASS"; echo "$name,$mode,XPASS,expected_fail_but_passed" >> "$csv"
        xpass=$((xpass+1))
      else
        echo "  $name: PASS"; echo "$name,$mode,PASS," >> "$csv"
        pass=$((pass+1))
      fi
    elif [ $rc -eq 124 ]; then
      if is_xfail "$name" "$xfail_list"; then
        echo "  $name: XFAIL (timeout)"; echo "$name,$mode,XFAIL,timeout" >> "$csv"
        xfail=$((xfail+1))
      else
        echo "  $name: FAIL (timeout)"; echo "$name,$mode,FAIL,timeout" >> "$csv"
        fail=$((fail+1))
      fi
    else
      if is_xfail "$name" "$xfail_list"; then
        echo "  $name: XFAIL (rc=$rc)"; echo "$name,$mode,XFAIL,$detail" >> "$csv"
        xfail=$((xfail+1))
      else
        echo "  $name: FAIL (rc=$rc)"; echo "$name,$mode,FAIL,$detail" >> "$csv"
        fail=$((fail+1))
      fi
    fi
  done

  echo
  echo "  $mode: $pass PASS, $xpass XPASS, $xfail XFAIL, $fail FAIL"
  echo
}

# ===========================================================================
# Run perftests
# ===========================================================================
run_perf_tests() {
  local mode="$1" env_prefix="$2" csv="$3"
  local perf_dir="$NIIN_DIR/perftest/build"
  echo "=== Performance tests ($mode) ==="
  echo "test,mode,scope,size_bytes,value,unit" > "$csv"

  # Put latency
  if [ -f "$perf_dir/shmem_put_latency" ]; then
    echo "  put_latency..."
    output=$(timeout $TIMEOUT_PERF env CUDA_VISIBLE_DEVICES=$GPU_PAIR $env_prefix \
             $MPI_RUN "$perf_dir/shmem_put_latency" -b 4 -e 65536 -i 200 -w 50 2>&1)
    scope=""
    echo "$output" | while IFS= read -r line; do
      if echo "$line" | grep -q "^shmem_put_latency"; then
        scope=$(echo "$line" | sed 's/.*(\(.*\))/\1/')
      elif echo "$line" | grep -qE "^[0-9]"; then
        sz=$(echo "$line" | awk '{print $1}')
        val=$(echo "$line" | awk '{print $2}')
        echo "put_latency,$mode,$scope,$sz,$val,us" >> "$csv"
      fi
    done
  fi

  # Put BW
  if [ -f "$perf_dir/shmem_put_bw" ]; then
    echo "  put_bw..."
    local heap_env=""
    if [ "$mode" = "ib" ]; then heap_env="NVSHMEM_SYMMETRIC_SIZE=1073741824"; fi
    output=$(timeout $TIMEOUT_PERF env CUDA_VISIBLE_DEVICES=$GPU_PAIR \
             NVSHMEM_SYMMETRIC_SIZE=1073741824 $env_prefix \
             $MPI_RUN "$perf_dir/shmem_put_bw" -b 1024 -e 67108864 -i 50 -w 10 -n 16 -t 256 2>&1)
    scope=""
    echo "$output" | while IFS= read -r line; do
      if echo "$line" | grep -q "^shmem_put_bw"; then
        scope=$(echo "$line" | sed 's/.*(\(.*\))/\1/')
      elif echo "$line" | grep -qE "^[0-9]"; then
        sz=$(echo "$line" | awk '{print $1}')
        val=$(echo "$line" | awk '{print $2}')
        echo "put_bw,$mode,$scope,$sz,$val,GB/s" >> "$csv"
      fi
    done
  fi

  # Get latency
  if [ -f "$perf_dir/shmem_get_latency" ]; then
    echo "  get_latency..."
    output=$(timeout $TIMEOUT_PERF env CUDA_VISIBLE_DEVICES=$GPU_PAIR $env_prefix \
             $MPI_RUN "$perf_dir/shmem_get_latency" -b 4 -e 65536 -i 200 -w 50 2>&1)
    scope=""
    echo "$output" | while IFS= read -r line; do
      if echo "$line" | grep -q "^shmem_get_latency"; then
        scope=$(echo "$line" | sed 's/.*(\(.*\))/\1/')
      elif echo "$line" | grep -qE "^[0-9]"; then
        sz=$(echo "$line" | awk '{print $1}')
        val=$(echo "$line" | awk '{print $2}')
        echo "get_latency,$mode,$scope,$sz,$val,us" >> "$csv"
      fi
    done
  fi

  # Put signal latency
  if [ -f "$perf_dir/shmem_put_signal_latency" ]; then
    echo "  put_signal_latency..."
    output=$(timeout $TIMEOUT_PERF env CUDA_VISIBLE_DEVICES=$GPU_PAIR $env_prefix \
             $MPI_RUN "$perf_dir/shmem_put_signal_latency" -b 4 -e 65536 -i 200 -w 50 2>&1)
    scope=""
    echo "$output" | while IFS= read -r line; do
      if echo "$line" | grep -q "^shmem_put_signal"; then
        scope=$(echo "$line" | sed 's/.*(\(.*\))/\1/')
      elif echo "$line" | grep -qE "^[0-9]"; then
        sz=$(echo "$line" | awk '{print $1}')
        val=$(echo "$line" | awk '{print $2}')
        echo "put_signal_latency,$mode,$scope,$sz,$val,us" >> "$csv"
      fi
    done
  fi

  echo "  done"
  echo
}

# ===========================================================================
# Execute
# ===========================================================================

if $RUN_P2P; then
  P2P_ENV=""
  if ! $SKIP_FUNC; then
    run_func_tests "p2p" "$P2P_ENV" "$XFAIL_P2P" "$RESULTS_DIR/func_p2p.csv"
  fi
  if ! $SKIP_PERF; then
    run_perf_tests "p2p" "$P2P_ENV" "$RESULTS_DIR/perf_p2p.csv"
  fi
fi

if $RUN_IB; then
  IB_ENV="NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_HCA=$HCA"
  if ! $SKIP_FUNC; then
    run_func_tests "ib" "$IB_ENV" "$XFAIL_IB" "$RESULTS_DIR/func_ib.csv"
  fi
  if ! $SKIP_PERF; then
    run_perf_tests "ib" "$IB_ENV" "$RESULTS_DIR/perf_ib.csv"
  fi
fi

# ===========================================================================
# Merge into a single summary
# ===========================================================================
echo "=== Generating summary ==="

# Merge functional
{
  echo "test,mode,status,detail"
  for f in "$RESULTS_DIR"/func_*.csv; do
    [ -f "$f" ] && tail -n+2 "$f"
  done
} > "$RESULTS_DIR/functional_results.csv"

# Merge perf
{
  echo "test,mode,scope,size_bytes,value,unit"
  for f in "$RESULTS_DIR"/perf_*.csv; do
    [ -f "$f" ] && tail -n+2 "$f"
  done
} > "$RESULTS_DIR/performance_results.csv"

# Summary counts
echo
echo "=== Summary ==="
for f in "$RESULTS_DIR"/func_*.csv; do
  [ -f "$f" ] || continue
  mode=$(basename "$f" .csv | sed 's/func_//')
  pass=$(grep -c ",PASS," "$f" || true)
  xpass=$(grep -c ",XPASS," "$f" || true)
  xfail=$(grep -c ",XFAIL," "$f" || true)
  fail=$(grep -c ",FAIL," "$f" || true)
  echo "  Functional ($mode): $pass PASS, $xpass XPASS, $xfail XFAIL, $fail FAIL"
done
echo
echo "Results in: $RESULTS_DIR/"
ls -la "$RESULTS_DIR"/*.csv
echo
echo "=== Done ==="
