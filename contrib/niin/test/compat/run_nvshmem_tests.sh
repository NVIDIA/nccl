#!/bin/bash
#
# Run NVSHMEM device tests against NIIN and produce a results CSV.
#
# Usage:
#   ./run_nvshmem_tests.sh [--build-only] [--transport nvl|pcie|ib]
#
# Options:
#   --build-only              Compile tests but don't run them
#   --transport <mode>        Transport preset: nvl, pcie, ib
#
# Env vars:
#   NCCL_HOME      Path to NCCL build (default: autodetect from script location)
#   NVSHMEM_HOME   Path to NVSHMEM source (default: /storage/benjaming/repo/nvshmem)
#   MPI_HOME       Path to MPI install (default: /cm/shared/apps/openmpi4/gcc/4.1.8)
#   ARCH           GPU arch (default: sm_89)
#   HCA            IB HCA to use (default: mlx5_1)
#   GPU_PAIR       CUDA_VISIBLE_DEVICES pair to use (default depends on transport)
#   NP             Number of PEs (default: 2)
#   RESULTS_CSV    Output CSV path (default: ../results/compat_results.csv)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NIIN_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
NCCL_HOME="${NCCL_HOME:-$(cd "$NIIN_DIR/../.." && pwd)}"
NVSHMEM_HOME="${NVSHMEM_HOME:-/storage/benjaming/repo/nvshmem}"
MPI_HOME="${MPI_HOME:-/cm/shared/apps/openmpi4/gcc/4.1.8}"
ARCH="${ARCH:-sm_89}"
HCA="${HCA:-mlx5_1}"
GPU_PAIR="${GPU_PAIR:-}"
NP="${NP:-2}"
RESULTS_CSV="${RESULTS_CSV:-$SCRIPT_DIR/../results/compat_results.csv}"
BUILD_DIR="/tmp/niin_compat_tests"
BUILD_ONLY=false
TRANSPORT="nvl"
TIMEOUT_SEC=60

while [ $# -gt 0 ]; do
  arg="$1"
  case $arg in
    --build-only) BUILD_ONLY=true ;;
    --ib) TRANSPORT="ib" ;;
    --transport)
      shift
      TRANSPORT="${1:-}"
      ;;
  esac
  shift
done

case "$TRANSPORT" in
  nvl|pcie|ib) ;;
  *)
    echo "Unknown transport: $TRANSPORT" >&2
    exit 1
    ;;
esac

if [ -z "$GPU_PAIR" ]; then
  case "$TRANSPORT" in
    nvl|ib) GPU_PAIR="4,5" ;;
    pcie)   GPU_PAIR="4,6" ;;
  esac
fi

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
case "$TRANSPORT" in
  ib)
    MPI_ENV="CUDA_VISIBLE_DEVICES=$GPU_PAIR NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_HCA=$HCA OMPI_MCA_btl=^openib"
    ;;
  nvl)
    MPI_ENV="CUDA_VISIBLE_DEVICES=$GPU_PAIR NCCL_IB_DISABLE=1 OMPI_MCA_btl=^openib"
    ;;
  pcie)
    MPI_ENV="CUDA_VISIBLE_DEVICES=$GPU_PAIR NCCL_IB_DISABLE=1 OMPI_MCA_btl=^openib"
    ;;
esac

# Known expected failures:
# - info: NIIN version format differs from NVSHMEM (build fail)
# - g_extended still has correctness gaps on some transports
XFAIL_TESTS="info g_extended"

is_xfail() {
  local name="$1"
  for xf in $XFAIL_TESTS; do
    if [ "$name" = "$xf" ]; then return 0; fi
  done
  return 1
}

# ---------------------------------------------------------------------------
# Discover tests
# ---------------------------------------------------------------------------
NVSHMEM_TEST_DIR="$NVSHMEM_HOME/test"
TESTS=()

# Query tests
for f in "$NVSHMEM_TEST_DIR"/device/query/*.cu; do
  [ -f "$f" ] && TESTS+=("device/query/$(basename ${f%.cu})")
done

# Sync tests
for f in "$NVSHMEM_TEST_DIR"/device/sync/*.cu; do
  [ -f "$f" ] && TESTS+=("device/sync/$(basename ${f%.cu})")
done

# PT-to-PT tests (only ones that don't need ring_alltoall)
for f in "$NVSHMEM_TEST_DIR"/device/pt-to-pt/*.cu; do
  name=$(basename ${f%.cu})
  # Skip tests that need ring_alltoall.h / data_check.h
  if grep -q "ring_alltoall.h\|data_check.h\|perf_utils.h" "$f" 2>/dev/null; then continue; fi
  # Skip QP tests (NCCL has no QP API)
  if echo "$name" | grep -q "qp_specific"; then continue; fi
  TESTS+=("device/pt-to-pt/$name")
done

mkdir -p "$BUILD_DIR"
mkdir -p "$(dirname "$RESULTS_CSV")"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
echo "=== Building ${#TESTS[@]} tests ==="
echo "test,category,build_status" > "${RESULTS_CSV}.build"

BUILD_PASS=0
BUILD_FAIL=0
BUILD_SKIP=0
BUILT_TESTS=()

for t in "${TESTS[@]}"; do
  name=$(basename "$t")
  category=$(dirname "$t" | tr '/' '_')
  src="$NVSHMEM_TEST_DIR/$t.cu"

  echo -n "  $name: "
  output=$(nvcc "$src" -o "$BUILD_DIR/$name" $NVCC_FLAGS 2>&1)
  if [ $? -eq 0 ]; then
    echo "OK"
    echo "$name,$category,OK" >> "${RESULTS_CSV}.build"
    BUILD_PASS=$((BUILD_PASS + 1))
    BUILT_TESTS+=("$name")
  else
    # Check if it's a known missing symbol
    echo "FAIL"
    echo "$output" | head -3 | sed 's/^/    /'
    echo "$name,$category,FAIL" >> "${RESULTS_CSV}.build"
    BUILD_FAIL=$((BUILD_FAIL + 1))
  fi
done

echo
echo "Build: $BUILD_PASS OK, $BUILD_FAIL failed, $BUILD_SKIP skipped"
echo

if $BUILD_ONLY; then
  rm -f "${RESULTS_CSV}.build"
  exit 0
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "=== Running ${#BUILT_TESTS[@]} tests ==="
echo "test,status,detail" > "$RESULTS_CSV"

RUN_PASS=0
RUN_FAIL=0
RUN_XFAIL=0

for name in "${BUILT_TESTS[@]}"; do
  echo -n "  $name: "

  output=$(timeout $TIMEOUT_SEC env $MPI_ENV $MPI_RUN "$BUILD_DIR/$name" 2>&1)
  rc=$?

  # Filter out MPI noise for error detection
  errors=$(echo "$output" | grep -i "error\|incorrect\|FAIL\|trap\|Segmentation" | \
           grep -v "mca_\|btl_\|orte_\|help-mpi\|MCA\|OpenFabrics\|device params\|aggregate\|cpcs\|pstat")

  if [ $rc -eq 0 ] && [ -z "$errors" ]; then
    if is_xfail "$name"; then
      echo "XPASS (expected fail but passed)"
      echo "$name,XPASS,expected_fail_but_passed" >> "$RESULTS_CSV"
      RUN_PASS=$((RUN_PASS + 1))
    else
      echo "PASS"
      echo "$name,PASS," >> "$RESULTS_CSV"
      RUN_PASS=$((RUN_PASS + 1))
    fi
  elif [ $rc -eq 124 ]; then
    if is_xfail "$name"; then
      echo "XFAIL (timeout, expected)"
      echo "$name,XFAIL,timeout" >> "$RESULTS_CSV"
      RUN_XFAIL=$((RUN_XFAIL + 1))
    else
      echo "FAIL (timeout)"
      echo "$name,FAIL,timeout" >> "$RESULTS_CSV"
      RUN_FAIL=$((RUN_FAIL + 1))
    fi
  else
    detail=$(echo "$errors" | head -1 | tr ',' ';')
    if is_xfail "$name"; then
      echo "XFAIL (rc=$rc, expected)"
      echo "$name,XFAIL,$detail" >> "$RESULTS_CSV"
      RUN_XFAIL=$((RUN_XFAIL + 1))
    else
      echo "FAIL (rc=$rc)"
      if [ -n "$errors" ]; then echo "    $(echo "$errors" | head -1)"; fi
      echo "$name,FAIL,$detail" >> "$RESULTS_CSV"
      RUN_FAIL=$((RUN_FAIL + 1))
    fi
  fi
done

echo
echo "=========================================="
echo "  $RUN_PASS passed, $RUN_XFAIL xfail, $RUN_FAIL failed"
echo "=========================================="
echo
echo "Results written to: $RESULTS_CSV"

rm -f "${RESULTS_CSV}.build"

# Exit with failure if any unexpected failures
exit $RUN_FAIL
