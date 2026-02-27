# Basic NCCL Tuner Plugin

This directory contains a minimal placeholder implementation of an NCCL tuner plugin. It serves as a starting point for developing custom tuner plugins by providing the essential function stubs and interface structure required by NCCL.

## Purpose

This basic plugin is designed to:
- Provide a minimal working example of the NCCL tuner plugin interface
- Serve as a template for developing custom tuner plugins
- Demonstrate the required function signatures and structure
- Implement placeholder functionality that can be extended


## Implementation Details

The plugin implements the following functions:

### `pluginInit`
```c
ncclResult_t pluginInit(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction, void **context)
```
- **Purpose**: Initialize the plugin with communicator information
- **Current Implementation**: Simple placeholder that returns success
- **Parameters**:
  - `nRanks`: Total number of ranks in the communicator
  - `nNodes`: Total number of nodes in the communicator
  - `logFunction`: NCCL debug logging function
  - `context`: Plugin context pointer (output)

### `pluginGetCollInfo`
```c
ncclResult_t pluginGetCollInfo(void* context, ncclFunc_t collType, size_t nBytes,
                              int numPipeOps, float** collCostTable, int numAlgo, int numProto,
                              int regBuff, int* nChannels)
```
- **Purpose**: Modify cost tables for collective operations
- **Current Implementation**:
  - Sets RING+SIMPLE algorithm to cost 0.0 (highest preference)
  - Sets channel count to 1
- **Parameters**:
  - `context`: Plugin context from init
  - `collType`: Type of collective operation
  - `nBytes`: Message size in bytes
  - `numPipeOps`: Number of pipeline operations
  - `collCostTable`: Cost table to modify
  - `numAlgo`: Number of algorithms
  - `numProto`: Number of protocols
  - `regBuff`: Whether buffer can be registered
  - `nChannels`: Number of channels to use (output)

### `pluginDestroy`
```c
ncclResult_t pluginDestroy(void* context)
```
- **Purpose**: Clean up plugin resources
- **Current Implementation**: Simple placeholder that returns success

## Cost Table Structure

The plugin demonstrates how to modify NCCL's cost tables:

```c
float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;
```

The cost table is a 2D array where:
- First dimension: Algorithm index (e.g., `NCCL_ALGO_RING`)
- Second dimension: Protocol index (e.g., `NCCL_PROTO_SIMPLE`)
- Values: Cost for that algorithm/protocol combination

### Cost Values
- **0.0**: Highest preference (lowest cost)
- **Positive values**: Relative costs (lower is better)
- **`NCCL_ALGO_PROTO_IGNORE`**: Disable this combination

## Building

```bash
make
```

This creates `libnccl-tuner-basic.so` which can be loaded by NCCL.

## Usage

### Loading the Plugin

```bash
export LD_LIBRARY_PATH=/path/to/basic:$LD_LIBRARY_PATH
mpirun -np 4 your_nccl_application
```

```bash
export NCCL_TUNER_PLUGIN=basic
export NCCL_TUNER_PLUGIN=libnccl-tuner-basic.so
export NCCL_TUNER_PLUGIN=/path/to/your/plugin/libnccl-tuner-basic.so
```

### Verifying Plugin Loading

Enable NCCL debug output to see if the plugin is loaded:

```bash
export NCCL_DEBUG=INFO
```

You should see messages indicating the tuner plugin is being used.

## Extending the Plugin

This basic plugin provides a foundation that you can extend:

### 1. Add Configuration Logic

Modify `pluginGetCollInfo` to implement your tuning strategy:

```c
__hidden ncclResult_t pluginGetCollInfo(void* context, ncclFunc_t collType, size_t nBytes,
                              int numPipeOps, float** collCostTable, int numAlgo, int numProto,
                              int regBuff, int* nChannels) {
  // Your custom tuning logic here
  if (nBytes < 1024) {
    // Small message optimization
    table[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = 0.0;
  } else {
    // Large message optimization
    table[NCCL_ALGO_RING][NCCL_PROTO_LL128] = 0.0;
  }

  // Dynamic channel selection
  *nChannels = (nBytes > 1024*1024) ? 4 : 1;

  return ncclSuccess;
}
```

### 2. Add Context Management

Use the context pointer to store plugin state:

```c
struct pluginContext {
  int initialized;
  size_t nRanks;
  size_t nNodes;
  // Add your plugin-specific data here
};
```

### 3. Add File-Based Configuration

Read configuration from files, environment variables, or other sources.

### 4. Add Topology Awareness

Use the `nRanks` and `nNodes` parameters to implement topology-specific tuning.

## File Structure

```
basic/
├── README.md          # This file
├── plugin.c           # Plugin implementation
├── Makefile           # Build configuration
└── nccl/              # NCCL header files
    └── tuner.h        # Tuner plugin interface definitions
```

## Next Steps

1. **Understand the Interface**: Study the function signatures and parameters
2. **Implement Your Logic**: Add your tuning strategy to `pluginGetCollInfo`
3. **Test Thoroughly**: Verify your plugin works with different message sizes and topologies
4. **Add Error Handling**: Implement proper error checking and resource management
5. **Document Your Changes**: Update this README with your specific implementation details

## Comparison with Example Plugin

- **Basic Plugin**: Minimal implementation, good for learning and simple use cases
- **Example Plugin**: Full-featured CSV-based configuration system, good for production use

Choose the basic plugin if you want to:
- Learn the tuner plugin interface
- Implement simple, hardcoded tuning strategies
- Build a custom plugin from scratch

Choose the example plugin if you want:
- File-based configuration
- Complex tuning strategies
- Production-ready features

## Resources

- [Parent Directory README](../README.md) - General tuner plugin development guide
- [Example Plugin](../example/README.md) - Fully featured implementation

This basic plugin provides the foundation you need to start developing custom NCCL tuner plugins. Extend it with your specific tuning logic and requirements.
