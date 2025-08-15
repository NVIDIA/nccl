# NCCL Inspector Example Scripts

This directory contains example scripts demonstrating how to use the NCCL Inspector plugin in different scenarios.

## Available Examples

### 1. Single Node Example
**File**: `single_node_example.sh`

A simple example showing how to run NCCL Inspector on a single node with basic NCCL performance tests.

### 2. Multi-Node SLURM Example
**File**: `multi_node_slurm_example.sh`

A comprehensive example demonstrating how to run NCCL Inspector in a multi-node SLURM environment with various NCCL collective operations.

### 3. Training Workload Example
**File**: `training_workload_example.sh`

An example showing how to integrate NCCL Inspector with real distributed training workloads using SLURM.

## Usage

1. Copy the relevant example script to your working directory
2. Modify the paths and configuration variables to match your environment
3. Make the script executable: `chmod +x script_name.sh`
4. Run the script according to your environment (direct execution or SLURM submission)

## Key Configuration Variables

All examples use these core NCCL Inspector environment variables:

- `NCCL_PROFILER_PLUGIN`: Path to the inspector plugin library
- `NCCL_INSPECTOR_ENABLE=1`: Enables the inspector plugin
- `NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS`: Dump interval in microseconds
- `NCCL_INSPECTOR_DUMP_DIR`: Output directory for inspector logs

## Notes

- These scripts use generic paths (`/path/to/...`) that need to be customized for your environment
- The SLURM examples include cluster-specific configurations that may need adjustment
- The training workload example is a template that requires customization for your specific training framework and model
