# NCCL Windows Support

This document describes the Windows platform support added to NCCL and the workarounds implemented for Linux-specific functionality.

## Overview

NCCL has been ported to Windows with a platform abstraction layer that provides cross-platform compatibility. The implementation maintains full functionality on Linux while providing reasonable alternatives on Windows for most features.

## Platform Detection

Platform detection is handled in `src/include/platform.h`:
- `NCCL_PLATFORM_WINDOWS` - Set to 1 on Windows
- `NCCL_PLATFORM_LINUX` - Set to 1 on Linux/Unix
- `NCCL_PLATFORM_POSIX` - Set to 1 on POSIX systems

## Windows-Specific Headers

The following platform abstraction headers are provided in `src/include/platform/`:

| Header | Purpose |
|--------|---------|
| `win32_defs.h` | Basic Windows type definitions and compatibility macros |
| `win32_misc.h` | Time functions, environment variables, random numbers, CPU affinity |
| `win32_thread.h` | pthread-compatible threading primitives |
| `win32_socket.h` | BSD sockets compatibility with Winsock2 |
| `win32_dl.h` | Dynamic library loading (dlopen/dlsym equivalents) |
| `win32_shm.h` | Shared memory (memory-mapped files) |
| `win32_ipc.h` | Inter-process communication (Named Pipes) |

## Workarounds for Linux-Specific Features

### 1. InfiniBand Transport (Linux-only)

**Limitation:** InfiniBand (IB) transport is not supported on Windows because the libibverbs and mlx5 libraries are Linux-specific.

**Workaround:** The entire `net_ib.cc` implementation is wrapped in platform guards. On Windows:
- `ncclNetIb` is defined with stub functions that return `ncclInternalError`
- `ncclIbDevices()` returns 0 devices
- Warning is logged when IB initialization is attempted

**Impact:** Multi-node communication must use socket-based transport on Windows.

### 2. Network Interface Speed Detection

**Limitation:** Linux uses sysfs (`/sys/class/net/<dev>/speed`) to query network interface speed.

**Workaround:** Windows implementation uses the `GetIfTable2()` API from `iphlpapi.dll`:
- `ncclGetInterfaceSpeed()` function queries `MIB_IF_TABLE2`
- Returns `ReceiveLinkSpeed` from `MIB_IF_ROW2` structure
- Falls back to default values if interface not found

**Location:** `src/include/platform/win32_socket.h`

### 3. Network Vendor Detection

**Limitation:** Linux reads vendor IDs from sysfs (`/sys/class/net/<dev>/device/vendor`).

**Workaround:** On Windows, vendor detection is disabled:
- Default thread/socket configuration is used
- `TRACE()` message indicates vendor detection is unavailable
- Users can override via `NCCL_NTHREADS` and `NCCL_NSOCKS_PERTHREAD` environment variables

**Impact:** May not achieve optimal socket configuration for cloud VMs (AWS, GCP).

### 4. CPU Affinity

**Limitation:** Linux uses `sched_setaffinity()`, `sched_getaffinity()`, and `cpu_set_t` from `<sched.h>`.

**Workaround:** Full Windows implementation in `win32_misc.h`:
- `cpu_set_t` - Structure supporting up to 1024 CPUs (16 processor groups)
- `CPU_SET`, `CPU_ZERO`, `CPU_CLR`, `CPU_ISSET` - Macros matching Linux behavior
- `CPU_COUNT`, `CPU_EQUAL`, `CPU_AND`, `CPU_OR` - Additional operations
- `sched_setaffinity()` - Uses `SetThreadAffinityMask()`
- `sched_getaffinity()` - Uses `GetProcessAffinityMask()`
- `ncclSetThreadAffinity()`, `ncclGetThreadAffinity()` - NCCL-specific functions

**Limitation:** Windows processor groups are handled via first group only (first 64 CPUs).

### 5. Inter-Process Communication (IPC)

**Limitation:** Linux uses Unix Domain Sockets with `SCM_RIGHTS` for file descriptor passing.

**Workaround:** Windows uses Named Pipes with `DuplicateHandle()`:
- Named Pipes for local IPC communication
- `DuplicateHandle()` for passing handles between processes
- Requires target process handle for duplication

**Location:** `src/include/platform/win32_ipc.h`

**Limitation:** Handle passing requires different workflow than Unix FD passing.

### 6. Shared Memory

**Limitation:** Linux uses POSIX shared memory (`shm_open`, `mmap`).

**Workaround:** Windows uses memory-mapped files:
- `CreateFileMapping()` / `OpenFileMapping()` for shared memory objects
- `MapViewOfFile()` for memory mapping
- Unique names derived from keys

**Location:** `src/include/platform/win32_shm.h`

### 7. Time Functions

**Limitation:** Linux uses `clock_gettime()` with `CLOCK_MONOTONIC`.

**Workaround:** Windows implementation in `win32_misc.h`:
- `QueryPerformanceCounter()` for high-resolution monotonic time
- `GetSystemTimeAsFileTime()` for wall-clock time

### 8. Dynamic Library Loading

**Limitation:** Linux uses `dlopen()`, `dlsym()`, `dlclose()`.

**Workaround:** Windows implementation in `win32_dl.h`:
- `LoadLibrary()` / `GetProcAddress()` / `FreeLibrary()`
- `dlerror()` equivalent using `GetLastError()` and `FormatMessage()`

## Build Configuration

### CMake Support

The CMakeLists.txt files support Windows builds:
- Automatic platform detection
- Conditional compilation for platform-specific code
- Windows SDK/toolchain support

### Required Windows SDKs

- Windows 10 SDK (or later)
- CUDA Toolkit with Windows support
- Visual Studio 2019 or later (for MSVC compiler)

## Known Limitations

1. **Multi-node support is limited** - Without InfiniBand, socket transport is the only option
2. **Processor groups** - Only first 64 CPUs are used for affinity on systems with >64 CPUs
3. **GPU Direct RDMA** - Not available without IB transport
4. **Performance** - Socket transport may have higher latency than IB

## Testing

### Platform Abstraction Tests

The platform abstraction layer has a comprehensive test suite in `tests/platform/`:

```bash
# Build standalone tests (no CUDA required)
cd tests/platform
mkdir build && cd build
cmake ..
cmake --build .
ctest --output-on-failure
```

#### Test Files

| File | Description |
|------|-------------|
| `test_platform_standalone.cpp` | Single-file comprehensive test |
| `test_platform_detection.cpp` | Platform macro and type tests |
| `test_time_functions.cpp` | clock_gettime, nanosleep tests |
| `test_socket_abstraction.cpp` | Socket, poll, interface tests |
| `test_thread_abstraction.cpp` | pthread mutex/cond/thread tests |
| `test_cpu_affinity.cpp` | CPU set operations and affinity |
| `test_misc_functions.cpp` | getpid, dlopen, atomic ops tests |
| `validate_headers.cpp` | Header coexistence validation |

#### Validation Status

All tests pass on Windows with:
- Visual Studio 2019+
- Windows 10 SDK 10.0.26100.0
- MSVC 19.50+ compiler

When running NCCL on Windows:

1. Ensure Winsock is initialized (`WSAStartup`)
2. Use TCP/IP socket transport (`NCCL_SOCKET_IFNAME` environment variable)
3. Test with single-node multi-GPU configurations first
4. Monitor for warning messages about unavailable features

## Contributing

When adding new features:
1. Use platform macros (`NCCL_PLATFORM_WINDOWS`, `NCCL_PLATFORM_LINUX`)
2. Add Windows implementations to appropriate `win32_*.h` headers
3. Add tests in `tests/platform/`
4. Document any new limitations in this file
