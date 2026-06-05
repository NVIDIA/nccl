/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_OS_SOCKET_PAIR_H_
#define NCCL_OS_SOCKET_PAIR_H_

#include "nccl.h"
#include "os.h"
#include <cstddef>

// Platform-agnostic descriptor for local socket pair endpoints
// On Linux: file descriptor (int)
// On Windows: socket (SOCKET)
typedef ncclSocketDescriptor ncclSocketPairDescriptor;

// Invalid descriptor constant
#define NCCL_SOCKET_PAIR_INVALID NCCL_INVALID_SOCKET

// Creates a socket pair: two connected endpoints for data transfer
// pair[0] is intended for reading, pair[1] for writing (following pipe() convention)
// Either endpoint can technically perform both operations, but typical usage is unidirectional:
// one side writes to pair[1], the other side reads from pair[0]
// Returns ncclSuccess on success, error code on failure
ncclResult_t ncclOsSocketPairCreate(ncclSocketPairDescriptor pair[2]);

// Closes both endpoints of a socket pair
// Skips any descriptor that is already NCCL_SOCKET_PAIR_INVALID
// Resets both descriptors to NCCL_SOCKET_PAIR_INVALID after closing
// Returns ncclSuccess on success, error code on failure
ncclResult_t ncclOsSocketPairClose(ncclSocketPairDescriptor pair[2]);

// Writes data to the socket pair
// Returns ncclSuccess on success, error code on failure
// On success, *written contains the number of bytes written (may be less than len)
// Callers must loop to ensure all data is written
ncclResult_t ncclOsSocketPairWrite(ncclSocketPairDescriptor descriptor, const void* buf, size_t len, size_t* written);

// Reads data from the socket pair
// Returns ncclSuccess on success, error code on failure
// On success, *nread contains the number of bytes read (may be less than len; 0 indicates EOF)
// Callers must loop to ensure all expected data is read
ncclResult_t ncclOsSocketPairRead(ncclSocketPairDescriptor descriptor, void* buf, size_t len, size_t* nread);

#endif // NCCL_OS_SOCKET_PAIR_H_
