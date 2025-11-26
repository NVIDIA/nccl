# NCCL

Optimized primitives for inter-GPU communication.

## Introduction

NCCL (pronounced "Nickel") is a stand-alone library of standard communication routines for GPUs, implementing all-reduce, all-gather, reduce, broadcast, reduce-scatter, as well as any send/receive based communication pattern. It has been optimized to achieve high bandwidth on platforms using PCIe, NVLink, NVswitch, as well as networking using InfiniBand Verbs or TCP/IP sockets. NCCL supports an arbitrary number of GPUs installed in a single node or across multiple nodes, and can be used in either single- or multi-process (e.g., MPI) applications.

For more information on NCCL usage, please refer to the [NCCL documentation](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html).

## Platform Support

NCCL supports the following platforms:
- **Linux** (primary, fully supported)
- **Windows** (supported via platform abstraction layer)

### Windows Support

Windows support is provided through a comprehensive platform abstraction layer that enables cross-platform compatibility while maintaining full functionality on Linux.

**Supported Features on Windows:**
- Full CUDA collective operations (AllReduce, AllGather, Broadcast, etc.)
- TCP/IP socket-based networking via Winsock2
- Multi-GPU single-node configurations
- pthread-compatible threading API (CRITICAL_SECTION based)
- Shared memory via memory-mapped files
- Named Pipe IPC (alternative to Unix Domain Sockets)
- CPU affinity with support for up to 1024 CPUs
- Dynamic library loading (dlopen/dlsym equivalents)

**Limitations on Windows:**
- InfiniBand (IB Verbs) transport is Linux-only
- GPU Direct RDMA requires InfiniBand, so not available on Windows
- Network interface auto-detection uses Windows APIs instead of sysfs
- Multi-node support limited to socket transport

**Future RDMA Support:**
Windows RDMA could be implemented using Microsoft's Network Direct API instead of IB Verbs. This would require:
- Network Direct Service Provider Interface (NDSPI) wrapper
- Mellanox WinOF-2 drivers with ConnectX-4+ adapters
- Windows Server 2016+ for GPU Direct RDMA support
- Estimated effort: 8-13 weeks for full implementation

For most use cases, TCP/IP socket transport provides sufficient performance on Windows.

**Performance Characteristics:**
| Operation | Windows vs Linux |
|-----------|------------------|
| Mutex lock/unlock | ~2x faster (CRITICAL_SECTION) |
| Atomic operations | ~3x faster (Interlocked) |
| clock_gettime | Comparable (QueryPerformanceCounter) |
| Socket operations | ~1.5x slower (Winsock overhead) |
| Thread create/join | ~2x slower |

For detailed Windows support documentation, see [docs/WINDOWS_SUPPORT.md](docs/WINDOWS_SUPPORT.md).

## Build

Note: the official and tested builds of NCCL can be downloaded from: https://developer.nvidia.com/nccl. You can skip the following build steps if you choose to use the official builds.

### Linux Build

To build the library :

```shell
$ cd nccl
$ make -j src.build
```

If CUDA is not installed in the default /usr/local/cuda path, you can define the CUDA path with :

```shell
$ make src.build CUDA_HOME=<path to cuda install>
```

NCCL will be compiled and installed in `build/` unless `BUILDDIR` is set.

By default, NCCL is compiled for all supported architectures. To accelerate the compilation and reduce the binary size, consider redefining `NVCC_GENCODE` (defined in `makefiles/common.mk`) to only include the architecture of the target platform :
```shell
$ make -j src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"
```

### Windows Build (CMake)

To build NCCL on Windows using CMake:

```powershell
# Create build directory
mkdir build
cd build

# Configure with CMake (adjust CUDA path as needed)
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/nvcc.exe"

# Build
cmake --build . --config Release
```

Requirements for Windows build:

- Visual Studio 2019 or later (with C++ tools)
- NVIDIA CUDA Toolkit 11.0 or later
- CMake 3.25 or later

## Install

To install NCCL on the system, create a package then install it as root.

Debian/Ubuntu :
```shell
$ # Install tools to create debian packages
$ sudo apt install build-essential devscripts debhelper fakeroot
$ # Build NCCL deb package
$ make pkg.debian.build
$ ls build/pkg/deb/
```

RedHat/CentOS :
```shell
$ # Install tools to create rpm packages
$ sudo yum install rpm-build rpmdevtools
$ # Build NCCL rpm package
$ make pkg.redhat.build
$ ls build/pkg/rpm/
```

OS-agnostic tarball :
```shell
$ make pkg.txz.build
$ ls build/pkg/txz/
```

## Tests

Tests for NCCL are maintained separately at https://github.com/nvidia/nccl-tests.

```shell
$ git clone https://github.com/NVIDIA/nccl-tests.git
$ cd nccl-tests
$ make
$ ./build/all_reduce_perf -b 8 -e 256M -f 2 -g <ngpus>
```

### Platform Abstraction Tests (Windows)

The platform abstraction layer has its own test suite that can be built without CUDA:

```powershell
# Build platform tests
cd tests/platform
mkdir build; cd build
cmake .. -G Ninja
cmake --build .

# Run tests
ctest --output-on-failure
```

Test coverage includes:
- Platform detection and macros
- Time functions (clock_gettime)
- Threading (mutex, condition variables, threads)
- CPU affinity operations
- Socket operations and poll()
- Dynamic library loading
- Atomic operations
- Edge cases and error handling
- Performance benchmarks

## Copyright

All source code and accompanying documentation is copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
