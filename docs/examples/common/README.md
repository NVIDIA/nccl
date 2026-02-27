# NCCL Common Utilities

## Description
This directory contains shared utilities and helper functions used across all NCCL examples. These utilities provide common functionality for error handling, device management, and MPI integration.

## Components

### Headers (`include/`)
- **utils.h**: General utility functions
- **nccl_utils.h**: NCCL error checking macros
- **mpi_utils.h**: MPI error checking macros

### Source Files (`src/`)
- **utils.cc**: General utility functions

## Key Features

### Error Checking Macros
```c
#define NCCLCHECK(cmd)  // NCCL error checking
#define CUDACHECK(cmd)  // CUDA error checking
#define MPICHECK(cmd)   // MPI error checking
```

## Usage in Examples
Include the headers in your example source files:
```c
#include "utils.h"
#include "mpi_utils.h"
```

## Notes
- All utilities include comprehensive error checking
- Functions are designed to be thread-safe
- Memory management functions handle null pointers safely
- MPI utilities are only needed for multi-process examples
