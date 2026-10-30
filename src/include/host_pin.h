/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_HOST_PIN_H_
#define NCCL_HOST_PIN_H_

#include "nccl.h"
#include <stddef.h>

// Zeroed, device-mapped host memory (src/misc/host_pin.cc). Its registration is
// dropped while every memory manager of the process is suspended and restored at the
// same address on resume.
ncclResult_t ncclHostPinAlloc(void** ptr, size_t size);
ncclResult_t ncclHostPinFree(void* ptr);
// Same lifetime for a host range its owner allocated (e.g. a shared-memory segment).
ncclResult_t ncclHostPinRegister(void* ptr, size_t size);
ncclResult_t ncclHostPinUnregister(void* ptr);
ncclResult_t ncclHostPinManagerCreated();
ncclResult_t ncclHostPinManagerDestroyed(bool suspended);
ncclResult_t ncclHostPinManagerSuspended();
ncclResult_t ncclHostPinManagerResumed();

#endif
