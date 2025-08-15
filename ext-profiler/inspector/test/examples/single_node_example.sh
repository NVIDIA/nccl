#!/bin/bash

# Single Node NCCL Inspector Example
# This script demonstrates how to run NCCL Inspector on a single node

# NCCL Inspector Configuration
export NCCL_PROFILER_PLUGIN=/path/to/nccl/ext-profiler/inspector/libnccl-profiler-inspector.so
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500

# Run NCCL performance test with inspector enabled
NCCL_DEBUG=WARN \
./build/test/perf/all_reduce_perf -b 8 -e 16G -f 2 -g 8
