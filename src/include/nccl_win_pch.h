/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_WIN_PCH_H_
#define NCCL_WIN_PCH_H_

/* Windows precompiled header for NCCL host compilation.
 * Pre-instantiates CUDA runtime and CCCL template headers once so that each
 * translation unit reuses the PCH instead of re-instantiating them, which
 * avoids C1060 (out of heap space) from CCCL's massive C++20 template library.
 */
#ifdef NCCL_OS_WINDOWS

/* Windows API — include before CUDA to avoid winsock redefinition */
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>

/* C++ standard library */
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

/* CUDA runtime — pulls in CCCL; compiled once into the PCH */
#include <cuda.h>
#include <cuda_runtime.h>

#endif /* NCCL_OS_WINDOWS */
#endif /* NCCL_WIN_PCH_H_ */
