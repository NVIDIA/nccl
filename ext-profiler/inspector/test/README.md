# NCCL Inspector Test Directory

This directory contains test files and examples for the NCCL Inspector plugin.

## Directory Structure

- **slurm/**: SLURM job scripts and test configurations for multi-node testing
  - **nccl-tests/**: NCCL performance test configurations and scripts

## Test Types

### Single Node Tests
Basic functionality tests that can be run on a single node to verify inspector operation.

### Multi-Node Tests (SLURM)
Distributed testing using SLURM job scheduler for multi-node scenarios.

## Running Tests

### Single Node Testing
```bash
# Basic functionality test
NCCL_DEBUG=WARN \
NCCL_PROFILER_PLUGIN=../libnccl-profiler-inspector.so \
NCCL_INSPECTOR_ENABLE=1 \
NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500 \
./all_reduce_perf -b 8 -e 16G -f 2 -g 8
```

### Multi-Node Testing
Use the SLURM scripts in the `slurm/` directory for distributed testing.

## Test Validation

After running tests, verify:
1. Inspector logs are generated in the expected output directory
2. JSON output files contain valid performance data
3. No errors in the inspector log output
4. Performance metrics are reasonable for the hardware configuration

## Troubleshooting

- Check that all required environment variables are set
- Verify the inspector plugin path is correct
- Ensure output directories are writable
- Monitor NCCL debug output for any initialization issues
