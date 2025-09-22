# NCCL Example Tuner Plugin

This example plugin shows a practical example of a CSV file-based tuning approach, allowing selective overrides for tuning parameters based on all tuning inputs without recompiling.

## Features

- **File-based Configuration**: Read tuning parameters from a CSV configuration file
- **Size-based Tuning**: Specify different configurations based on message size ranges
- **Dimension-aware Tuning**: Match configurations based on number of nodes and ranks
- **Optional Channels Configuration**: Set specific channel counts or use -1 to keep NCCL's default
- **Environment Variable Support**: Specify config file location via `NCCL_TUNER_CONFIG_FILE`
- **Fallback Behavior**: Gracefully handles missing config files and invalid entries

## Building

```bash
make
```

This will create `libnccl-tuner-example.so` that can be loaded by NCCL.

## Configuration File Format

The configuration file uses CSV (Comma-Separated Values) format with one configuration per line:

```
collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff
```

### Parameters

- **collective_type**: The collective operation type
  - `broadcast`, `reduce`, `allgather`, `reducescatter`, `allreduce`

- **min_bytes/max_bytes**: The message size range (in bytes) for which this config applies
  - Use `0` for minimum and `4294967295` for maximum (covers all sizes)

- **algorithm**: The NCCL algorithm to use
  - `tree`, `ring`, `collnet_direct`, `collnet_chain`, `nvls`, `nvls_tree`, `pat`

- **protocol**: The NCCL protocol to use
  - `ll`, `ll128`, `simple`

- **channels**: Number of channels (SMs) to use
  - Use a positive integer to specify exact channel count
  - Use `-1` to keep NCCL's default channel selection

- **nNodes**: Number of nodes to match
  - Use a positive integer to match specific node count
  - Use `-1` to match any number of nodes

- **nRanks**: Number of ranks to match
  - Use a positive integer to match specific rank count
  - Use `-1` to match any number of ranks

- **numPipeOps**: Number of pipeline operations to match (optional)
  - Use a positive integer to match specific pipeline operation count
  - Use `-1` to match any number of pipeline operations
  - If omitted, configuration will match any numPipeOps value

- **regBuff**: Whether user buffer can be registered (optional)
  - Use `0` to match only non-registered buffers
  - Use `1` to match only registered buffers
  - Use `-1` to match either registered or non-registered buffers
  - If omitted, configuration will match any regBuff value

### Example Configuration

```csv
# Single-node, small allreduce: use tree algorithm, registered buffers only
allreduce,0,65536,tree,simple,2,1,-1,-1,1

# 4-node, 32-rank setup: medium allreduce, single pipeline op, non-registered buffers
allreduce,65537,1048576,ring,simple,4,4,32,1,0

# Any topology: large allreduce with LL128, multiple pipeline ops, any buffer type
allreduce,1048577,4294967295,ring,ll128,-1,-1,-1,4,-1

# Single-node broadcast: prefer tree, any pipeOps, registered buffers (backward compatible)
broadcast,0,32768,tree,simple,-1,1,-1

# Multi-node broadcast: optimized for non-registered buffers, single pipeline op
broadcast,32769,4294967295,ring,simple,2,-1,-1,1,0
```

Comments start with `#` and empty lines are ignored. The CSV format makes it easy to edit configurations in spreadsheet applications like Excel, Google Sheets, or LibreOffice Calc.

### Backward Compatibility

Configurations without the numPipeOps and/or regBuff parameters are fully supported:
- 8 fields: matches any numPipeOps and regBuff values
- 9 fields: matches any regBuff value
- 10 fields: full parameter specification

This ensures existing configuration files continue to work without modification.

## Usage

### Method 1: Default Config File
Place your configuration in `nccl_tuner.conf` in the current working directory.

### Method 2: Environment Variable
Set the `NCCL_TUNER_CONFIG_FILE` environment variable to specify the config file path:

```bash
export NCCL_TUNER_CONFIG_FILE=/path/to/your/tuner.conf
mpirun -np 4 your_nccl_application
```

## Editing Configuration Files

### Generating Configuration Files from Raw Data

A python script to generate valid CSV configs has been provided. [Using optimize_config.py](scripts/README.md).

### Spreadsheet Tips:
- Use column headers: `collective_type,min_bytes,max_bytes,algorithm,protocol,channels,nNodes,nRanks,numPipeOps,regBuff`
- Save as CSV format (not Excel format) for the plugin to read
- Use data validation to prevent typos in algorithm/protocol names

## Logging

The plugin uses NCCL's logging system. To see tuner-related messages:

```bash
export NCCL_DEBUG=INFO
```

This will show when configurations are loaded and applied, including the topology information.

For detailed debugging output during tuning decisions:

```bash
export NCCL_DEBUG=TRACE
```

This will show verbose information about which configurations are being evaluated and matched.

## Dimension Matching

Configurations are only applied when the topology matches:

- **Exact Match**: Configuration specifies `nNodes=4,nRanks=32`, only applied when communicator has exactly 4 nodes and 32 ranks
- **Wildcard Nodes**: Configuration specifies `nNodes=-1,nRanks=8`, applied to any topology with exactly 8 ranks
- **Wildcard Ranks**: Configuration specifies `nNodes=2,nRanks=-1`, applied to any 2-node topology regardless of ranks per node
- **Wildcard Both**: Configuration specifies `nNodes=-1,nRanks=-1`, applied to any topology

This allows you to create specialized configurations for different cluster setups while maintaining flexibility.

## Default Behavior

If no configuration file is found or no matching configuration exists for a collective operation, the plugin falls back to preferring the ring algorithm with simple protocol. All configured algorithm/protocol combinations are given a low cost (0.0) to make them preferred by NCCL's selection logic.

When channels is set to `-1`, NCCL's default channel selection logic is preserved, allowing the system to automatically determine the optimal number of channels based on hardware and message size.

## Troubleshooting

1. **Config file not found**: Check the file path and permissions
2. **Configurations not applied**: Verify the collective type, size ranges, algorithm/protocol names, and topology parameters
3. **Plugin not loaded**: Ensure `LD_LIBRARY_PATH` includes the plugin directory and that `NCCL_TUNER_PLUGIN` either specifies the plugin name, or an absolute path to the plugin shared library.
4. **No effect on performance**: Check that NCCL is actually using the tuner plugin with `NCCL_DEBUG=INFO`
5. **Topology mismatch**: Verify that nNodes and nRanks match your actual setup, or use -1 for wildcards
6. **CSV parsing errors**: Ensure no spaces after commas, or quote fields containing spaces
