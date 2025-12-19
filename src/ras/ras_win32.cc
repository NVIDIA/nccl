/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * Windows RAS stub implementation.
 * RAS (Remote Asynchronous Signaling) is not supported on Windows.
 */

#include "platform.h"

#if NCCL_PLATFORM_WINDOWS

#include "nccl.h"
#include "ras.h"

ncclResult_t ncclRasCommInit(struct ncclComm *comm, struct rasRankInit *myRank)
{
    (void)comm;
    (void)myRank;
    // RAS not supported on Windows - return success to allow normal operation
    return ncclSuccess;
}

ncclResult_t ncclRasCommFini(const struct ncclComm *comm)
{
    (void)comm;
    return ncclSuccess;
}

ncclResult_t ncclRasAddRanks(struct rasRankInit *ranks, int nranks)
{
    (void)ranks;
    (void)nranks;
    return ncclSuccess;
}

#endif // NCCL_PLATFORM_WINDOWS
