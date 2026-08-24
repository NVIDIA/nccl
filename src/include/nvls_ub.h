/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_NVLS_UB_H_
#define NCCL_NVLS_UB_H_

#include "nccl.h"
#include "device.h"
#include "mc_arena.h"
#include <cuda.h>
#include <stddef.h>
#include <stdint.h>

struct ncclComm;
struct ncclReg;

// NVLS user buffer (UB) registration. User buffers are bound into ranges of the
// UB partition of the NVLS domain's shared multicast group, so a registration costs
// an arena range rather than a whole MC object.

#if CUDART_VERSION >= 12010

// The UB carveout to reserve in the MC group, resolving NCCL_NVLS_UB_SIZE; 0 disables UB,
// and comms that cannot register (e.g. MNNVL) resolve to 0.
size_t ncclNvlsUbSize(struct ncclComm* comm);

// Per-rank scratch the registration protocol needs from the NVLS shared memory.
size_t ncclNvlsUbScratchTypeSize(void);

// Check if this comm participates in UB registration; uniform across local ranks, so
// callers may skip the collective protocol.
bool ncclNvlsUbEnabled(struct ncclComm* comm);

// Whether this record can be multicast-bound, latching the first verdict on it
// (NVLS_REG_POSSIBLE or NVLS_REG_NO_SUPPORT). The multi-segment policy is the caller's.
bool ncclNvlsUbEligible(struct ncclComm* comm, const void* buff, struct ncclReg* reg);

// Collectively resolve multicast addresses for sendbuff/recvbuff, binding new arena
// ranges as needed. Callers gate on ncclNvlsUbEnabled and pass records that
// passed ncclNvlsUbEligible, or NULL. A NULL buffer must be NULL on every rank (it
// derives from the collective type). Best effort: unless every buffer resolved,
// *outRegBufUsed is 0 and the outputs are cleared.
ncclResult_t ncclNvlsUbRegister(struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendbuffSize,
                                size_t recvbuffSize, struct ncclReg* sendReg, struct ncclReg* recvReg,
                                int* outRegBufUsed, void** outRegBufSend, void** outRegBufRecv);

#endif // CUDART_VERSION >= 12010

#endif // NCCL_NVLS_UB_H_
