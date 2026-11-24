/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_REGISTER_H_
#define NCCL_REGISTER_H_

#include "device.h"

#include <cuda.h>
#include <stdint.h>

struct ncclMcArenaReg;

int64_t ncclParamLocalRegister();
int64_t ncclParamGraphRegister();

enum {
  NET_REG_COMPLETE = 0x01,
  NVLS_REG_COMPLETE = 0x02,
  NVLS_REG_POSSIBLE = 0x04,
  NVLS_REG_NO_SUPPORT = 0x08,
  COLLNET_REG_COMPLETE = 0x10,
  IPC_REG_COMPLETE = 0x20
};

struct ncclPeerRegIpcAddr {
  uintptr_t* devPeerRmtAddrs;
  uintptr_t* hostPeerRmtAddrs;
};

struct ncclRegNetHandles {
  void* handle;
  struct ncclProxyConnector* proxyConn;
  struct ncclRegNetHandles* next;
};

struct ncclReg {
  // common attributes
  uintptr_t begAddr, endAddr; // page aligned
  int localRefs;
  int graphRefs;
  uint32_t state;
  // net reg
  struct ncclRegNetHandles* netHandleHead;
  // nvls reg
  // Committed arena registration, owned here: NVLS_REG_COMPLETE implies non-NULL.
  struct ncclMcArenaReg* nvlsUbReg;
  // collnet reg
  void* collnetHandle;
  // gin reg
  void** ginMhandles;
  void** ginHandles;
  struct ncclProxyConnector* collnetProxyconn;
  // general ipc reg
  struct ncclPeerRegIpcAddr regIpcAddrs;
  struct ncclIpcRegInfo** ipcInfos;  // Dynamically allocated, sized to ipcInfosSize
  int ipcInfosSize;                  // Size of ipcInfos array (localRanks or nRanks for cross-clique)
};

struct ncclRegCache {
  struct ncclReg** slots;
  int capacity, population;
  uintptr_t pageSize;
};

// Device arrays of remote peer addresses for the registration records (regIpcAddrs.devPeerRmtAddrs, ipcIndexSize
// pointers each). As its own cuMem allocation every array would pay the allocation granularity for a few bytes,
// so the records of a communicator carve equal arrays out of shared blocks. An array returns to the free list
// with its record and is zeroed there, so that a kernel which still reads it after the deregistration faults as it
// did on the unmapped allocation; the blocks live as long as the communicator.
struct ncclRegIpcAddrBlock {
  struct ncclRegIpcAddrBlock* next;
  uintptr_t* base;
};

struct ncclRegIpcAddrPool {
  struct ncclRegIpcAddrBlock* blocks; // the head block is the one arrays are still carved from
  int arrayEntries; // pointers per array (ipcIndexSize), fixed for the communicator
  int arraysPerBlock;
  int nextArray; // arrays of the head block already handed out
  uintptr_t** freeArrays; // arrays returned by record cleanup, reused first
  int nFree, freeCap;
  cudaEvent_t clearEvent; // after the last zeroing; waited for before the blocks are freed
};

ncclResult_t ncclRegIpcAddrsAlloc(struct ncclComm* comm, int arrayEntries, cudaStream_t stream, uintptr_t** addrs);
// clear: zero the array on the communicator's host stream (not while the communicator is being freed)
ncclResult_t ncclRegIpcAddrsFree(struct ncclComm* comm, uintptr_t* addrs, bool clear);

ncclResult_t ncclRegCleanup(struct ncclComm* comm);
ncclResult_t ncclCommGraphRegister(const ncclComm_t comm, void* buff, size_t size, void** handle);
ncclResult_t ncclCommGraphDeregister(const ncclComm_t comm, struct ncclReg* handle);
ncclResult_t ncclRegLocalIsValid(struct ncclReg* reg, bool* isValid);

#endif
