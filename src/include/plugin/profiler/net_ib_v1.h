/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NET_IB_V1_H_
#define NET_IB_V1_H_

#define NCCL_PROFILER_NET_IB_VER 1

enum {
  ncclProfileQp = (1 << 0),
};

// The data structure version is encoded in the plugin identifier bitmask and
// passed to NCCL core through the profiler callback. NCCL copies the plugin
// identifier in the event descriptor before calling the profiler startEvent
// function. The profiler should inspect the plugin id to find out the source
// plugin as well as the version of the event struct
typedef struct {
  uint8_t type;        // event type (plugin defined)
  union {
    struct {
      int device;      // network device id
      uint64_t wr_id;  // work request id
      int opcode;      // ibv opcode
      int qpNum;       // QP number
      size_t length;   // work request data length
    } qp;
  };
} ncclProfilerNetIbDescr_v1_t;

#endif
