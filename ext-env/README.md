# NCCL Environment Plugin Documentation

This page describes the NCCL Environment plugin API and how to implement an environment plugin for NCCL.

# Overview

To allow NCCL to customize environment variable handling and provide enhanced configuration management, NCCL provides an environment plugin interface. Environment plugins allow users to implement custom environment variable resolution, validation, and transformation logic without modifying NCCL core code.

Environment plugins come as a shared library called `libnccl-env.so`. That shared library contains one or more implementations of the NCCL ENV API, in the form of versioned structs, filled with pointers to all required functions.

# Plugin architecture

## Plugin name and supporting multiple environment plugins

When NCCL is initialized, it will look for a `libnccl-env.so` library and dynamically load it, then look for symbols inside the library.

The `NCCL_ENV_PLUGIN` environment variable allows multiple plugins to coexist. If set, NCCL will look for a library with a name of `libnccl-env-${NCCL_ENV_PLUGIN}.so`. It is therefore advised to name the library following that pattern, with a symlink pointing `libnccl-env.so` to `libnccl-env-${NCCL_ENV_PLUGIN}.so`. That way, if there are multiple plugins in the path, setting `NCCL_ENV_PLUGIN` will allow users to select the right plugin.

## Struct versioning

Once a library is found, NCCL will look for a symbol named `ncclEnvPlugin_vX`, with `X` increasing over time. The versioning ensures that the plugin and the NCCL core are compatible.

Plugins are encouraged to provide multiple of those symbols, implementing multiple versions of the NCCL ENV API, so that the same plugin can be compiled and support a wide range of NCCL versions.

Conversely, and to ease transition, NCCL can choose to support different plugin versions, looking for the latest ncclEnv struct version, but also looking for older ones so that older plugins would still work.

## Headers management

To help users build plugins effortlessly, plugins should copy the `ncclEnv_vX` definitions they support to their internal includes. An example is shown in `ext-env/example/` where we keep all headers in the `nccl/` directory and provide thin layers to implement old versions on top of newer ones.

The `nccl/` directory is populated with `env_vX.h` files extracting all relevant definitions from old API versions. It also provides error codes in `err.h`.

# API (v1)

Below is the main `ncclEnv_v1` struct. Each function is explained in later sections.

```c
typedef struct {
  const char* name;

  // Initialize the environment plugin
  // Input
  //  - ncclMajor: NCCL major version number
  //  - ncclMinor: NCCL minor version number
  //  - ncclPatch: NCCL patch version number
  //  - suffix: NCCL version suffix string
  ncclResult_t (*init)(uint8_t ncclMajor, uint8_t ncclMinor, uint8_t ncclPatch, const char* suffix);

  // Finalize the environment plugin
  ncclResult_t (*finalize)(void);

  // Get environment variable value
  // Input
  //  - name: environment variable name
  // Output
  //  - returns: pointer to environment variable value string, or NULL if not found. The plugin is responsible for keeping the
  //             returned value (address) valid until it is no longer needed by NCCL. This happens when NCCL calls ``finalize``
  //             or ``getEnv`` again on the same variable name. In any other case, modifying the variable (e.g., through
  //             ``setenv``) is considered undefined behavior since NCCL might access the returned address after the plugin has
  //             reset the variable.
  const char* (*getEnv)(const char* name);
} ncclEnv_v1_t;
```

## Error codes

All plugin functions use NCCL error codes as return value. `ncclSuccess` should be returned upon success.

Otherwise, plugins can return one of the following:
- `ncclSystemError` is returned when a system call fails, such as memory allocation or file I/O errors.
- `ncclInternalError` is returned when the plugin encounters an internal error during initialization or finalization.
- `ncclInvalidUsage` should be returned when the error is most likely a user error, such as invalid configuration.
- `ncclInvalidArgument` should be returned when invalid arguments are passed to plugin functions.

## Operation overview

NCCL will call the `init` function first during initialization, passing the NCCL version information. This allows the plugin to initialize its internal state and validate compatibility with the NCCL version.

The `getEnv` function is called whenever NCCL needs to retrieve an environment variable value. This provides the plugin with the opportunity to implement custom environment variable resolution, validation, or transformation logic. The plugin should keep every variable accessible thoroughout the plugin lifetime (i.e., until NCCL calls finalize).

When NCCL is finalized, the `finalize` function is called to allow the plugin to clean up any resources and perform any necessary cleanup operations.

## API Functions

### Initialization

#### name

The `name` field should point to a character string with the name of the environment plugin. This will be used for all logging, especially when `NCCL_DEBUG=INFO` is set.

#### init

As soon as NCCL finds the plugin and the correct ncclEnv symbol, it calls its `init` function. This allows the plugin to initialize its internal context and validate compatibility with the NCCL version.

The function receives:
- `ncclMajor`, `ncclMinor`, `ncclPatch`: NCCL version numbers for compatibility checking
- `suffix`: NCCL version suffix string (e.g., "+cuda12.0")

If the `init` function does not return `ncclSuccess`, NCCL will fall back to the internal environment plugin.


#### finalize

When the environment plugin is no longer needed, a call to `finalize` allows the plugin to clean up resources and perform any necessary cleanup operations.

### Environment variable handling

#### getEnv

The `getEnv` function is called whenever NCCL needs to retrieve an environment variable value. This function provides the plugin with the opportunity to implement custom environment variable resolution logic.

The function receives:
- `name`: The name of the environment variable to retrieve

The function should return:
- A pointer to the environment variable value string if found
- `NULL` if the environment variable is not set or not found

This allows plugins to implement various features such as:
- Environment variable validation and sanitization
- Dynamic environment variable resolution
- Configuration file integration
- Environment variable transformation or substitution
- Hierarchical configuration management

The returned memory address for the variable should not be modified by the plugin until it is safe to do so. That is, when NCCL
calls the plugin ``finalize`` or ``getEnv`` function for the same variable again. In any other case, modifying the variable is
considered undefined behavior.

# Plugin implementation examples

## Basic environment plugin

A basic environment plugin that simply delegates to the system `getenv` function:

```c
#include "nccl_env.h"

static ncclResult_t ncclEnvInit(uint8_t ncclMajor, uint8_t ncclMinor, uint8_t ncclPatch, const char* suffix) {
  return ncclSuccess;
}

static ncclResult_t ncclEnvFinalize(void) {
  return ncclSuccess;
}

static const char* ncclEnvGetEnv(const char* name) {
  return getenv(name);
}

const ncclEnv_v1_t ncclEnvPlugin_v1 = {
  .name = "ncclEnvBasic",
  .init = ncclEnvInit,
  .finalize = ncclEnvFinalize,
  .getEnv = ncclEnvGetEnv,
};
```

## Loading the plugin

Set the `LD_LIBRARY_PATH` to include your plugin directory:

```bash
export LD_LIBRARY_PATH=/path/to/your/plugin:$LD_LIBRARY_PATH
```

Set `NCCL_ENV_PLUGIN` to either the plugin name or the absolute path to the plugin file:

```bash
export NCCL_ENV_PLUGIN=myenv
export NCCL_ENV_PLUGIN=libnccl-env-myenv.so
export NCCL_ENV_PLUGIN=/path/to/your/plugin/libnccl-env-myenv.so
```

NCCL will automatically discover and load the plugin based on the exported symbol names.

# Advanced topics

## Plugin versioning

NCCL supports multiple plugin interface versions. Make sure your plugin exports the correct version:

```c
const ncclEnv_v1_t ncclEnvPlugin_v1 = {
  .name = "YourPluginName",
  .init = yourInitFunction,
  .finalize = yourFinalizeFunction,
  .getEnv = yourGetEnvFunction,
};
```

## Environment variable caching

For performance reasons, plugins may want to implement caching of environment variable values. However, care should be taken to ensure that cached values remain consistent with the actual environment state.

## Integration with configuration management systems

Environment plugins can integrate with external configuration management systems by:
- Reading configuration from files or databases
- Implementing hierarchical configuration resolution
- Supporting configuration hot-reloading
- Providing configuration validation and schema enforcement

# Best practices

1. **Test thoroughly**: Verify your plugin works with various environment variable configurations
2. **Handle edge cases**: Ensure your plugin behaves correctly with unusual or malformed input
3. **Document your approach**: Clearly document your environment variable handling strategy
4. **Version your plugin**: Use meaningful version numbers and maintain backward compatibility
5. **Performance optimization**: Keep plugin logic lightweight to avoid impacting NCCL performance
6. **Error handling**: Implement robust error handling and graceful degradation
7. **Security**: Validate and sanitize environment variable values appropriately

# Known limitations

- Environment plugins are called synchronously during NCCL initialization and environment variable access
- Plugins should avoid blocking operations in the `getEnv` function
- The plugin interface does not support asynchronous environment variable updates

# Contributing

When developing new environment plugins:
- Follow the existing code style and structure
- Include comprehensive documentation
- Add example configurations and test cases
- Consider contributing useful plugins back to the community

# Resources

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- Example plugin implementations in this directory

For questions and support, refer to the NCCL community resources and documentation.
