#ifndef _NCCL_DEVICE_GIN__SCRATCH_A2A__TYPES_H_
#define _NCCL_DEVICE_GIN__SCRATCH_A2A__TYPES_H_

#include "../../include/nccl_device/core.h"

struct ncclGinSyncHandle {
  // signals to sync with remote peers
  ncclGinSignal_t railSignals;
};

#endif // _NCCL_DEVICE_GIN__SCRATCH_A2A__TYPES_H_
