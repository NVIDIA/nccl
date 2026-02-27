# NCCL Tuner Plugin Development

This directory contains resources and examples for developing NCCL tuner plugins. Tuner plugins allow you to customize NCCL's algorithm and protocol selection behavior to optimize performance for specific workloads and hardware configurations.

## Overview

NCCL tuner plugins provide a way to influence NCCL's automatic algorithm and protocol selection by modifying the cost tables that NCCL uses to make decisions. This allows you to:

- Override default algorithm/protocol combinations for specific collective operations
- Customize tuning based on message size, topology, and other parameters
- Implement sophisticated tuning strategies without recompiling NCCL
- Optimize performance for specific hardware configurations or workloads

## Tuner Plugin Interface

NCCL tuner plugins must implement the `ncclTuner_t` interface defined in `nccl_tuner.h` within `nccl/src/include/plugin`. These definitions have been forked to `tuner.h` in each example plugin, and it is expected that any plugin implementor forks the internal NCCL definitions as well. The current interface includes:

```c
// Initialize the tuner plugin
ncclResult_t (*init)(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction, void **context);

// Get and modify collective operation cost information
ncclResult_t (*getCollInfo)(void* context, ncclFunc_t collType, size_t nBytes,
                           int numPipeOps, float** collCostTable, int numAlgo, int numProto,
                           int regBuff, int* nChannels);

// Clean up plugin resources
ncclResult_t (*destroy)(void* context);
```

## Development Guidelines

### 1. Plugin Structure

A typical tuner plugin should:
- Include the necessary forked NCCL headers (`tuner.h`)
- Implement all required interface functions
- Export the plugin structure with appropriate version
- Handle all input parameters gracefully

### 2. Cost Table Modification

The `getCollInfo` function receives a cost table that maps algorithm/protocol combinations to performance costs. Lower costs indicate preferred combinations. You can:

- Set costs to `0.0` to make combinations highly preferred
- Set costs to `NCCL_ALGO_PROTO_IGNORE` to disable combinations
- Use relative costs to create preferences between options

### 3. Channel Management

The `nChannels` parameter allows you to:
- Set a specific number of channels to use
- Return the original value to preserve NCCL's default behavior
- Implement dynamic channel selection based on message size or topology

### 4. Error Handling

Always return appropriate `ncclResult_t` values:
- `ncclSuccess` for successful or ignored operations
- `ncclInternalError` for plugin-specific errors. Returning an error is only advisable on plugin initialization and destruction, as the penalty users can pay for the overhead of a failed plugin call can be immense.
- Other NCCL error codes as appropriate

## Getting Started

### Option 1: Start with the Example Plugin

If you're new to tuner plugin development, start with the `example/` directory:

```bash
cd example/
make
```

This provides a CSV-based configuration system that you can customize or use as a template.

## Building and Testing

### Build Requirements

- GCC or compatible C compiler
- NCCL headers (included in `nccl/` subdirectories)
- Make

## Option 2: Use the Basic Plugin

For more customized tuning needs, you might want to start with a clean baseline. In that case, base off the basic plugin in the `basic/` directory:

```bash
cd basic/
make
```

### Build Process

Each plugin directory contains a Makefile:

```bash
cd basic/    # or example/
make
```

This generates a shared library (`.so` file) that can be loaded by NCCL.

### Loading the Plugin

Set the `LD_LIBRARY_PATH` to include your plugin directory:

```bash
export LD_LIBRARY_PATH=/path/to/your/plugin:$LD_LIBRARY_PATH
```

Set `NCCL_TUNER_PLUGIN` to either the plugin name, or the absolute path to the plugin file. Any of the below can work:

```bash
export NCCL_TUNER_PLUGIN=example
export NCCL_TUNER_PLUGIN=libnccl-tuner-example.so
export NCCL_TUNER_PLUGIN=/path/to/your/plugin/libnccl-tuner-example.so
```

NCCL will automatically discover and load the plugin based on the exported symbol names.

## Advanced Topics

### Plugin Versioning

NCCL supports multiple plugin interface versions. Make sure your plugin exports the correct version:

```c
const ncclTuner_v4_t ncclTunerPlugin_v4 = {
    .name = "YourPluginName",
    .init = yourInitFunction,
    .getCollInfo = yourGetCollInfoFunction,
    .destroy = yourDestroyFunction
};
```

### Multi-GPU and Multi-Node Considerations

Your plugin receives topology information (`nRanks`, `nNodes`) during initialization. Use this to:
- Implement topology-aware tuning strategies
- Handle single-node vs. multi-node optimizations differently
- Scale channel counts based on available hardware

### Performance Optimization

- Keep plugin logic lightweight to avoid impacting NCCL performance
- Cache expensive computations when possible
- Use the logging system for debugging but avoid excessive output in production

## Debugging and Logging

Use NCCL's debug logging system:

```bash
export NCCL_DEBUG=INFO    # General information
export NCCL_DEBUG_SUBSYS=TUNING
```

Within your plugin, use the provided `ncclDebugLogger_t` function for consistent logging.

## Best Practices

1. **Test thoroughly**: Verify your plugin works with various message sizes and topologies
2. **Handle edge cases**: Ensure your plugin behaves correctly with unusual input parameters
3. **Document your approach**: Clearly document your tuning strategy and configuration options
4. **Version your plugin**: Use meaningful version numbers and maintain backward compatibility
5. **Performance validation**: Measure the impact of your tuning decisions on real workloads

## Contributing

When developing new tuner plugins:
- Follow the existing code style and structure
- Include comprehensive documentation
- Add example configurations and test cases
- Consider contributing useful plugins back to the community

## Resources

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- Example plugin implementations in this directory

For questions and support, refer to the NCCL community resources and documentation.
