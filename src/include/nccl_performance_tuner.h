#ifndef NCCL_PERFORMANCE_TUNER_H_
#define NCCL_PERFORMANCE_TUNER_H_

#include "nccl.h"
#include "nccl_net.h"

// Symbol name for NCCL performance tuner plugin
#define NCCL_PERFORMANCE_TUNER_SYMBOL "ncclPerformanceTunerSymbol"

#define NCCL_NUM_FUNCTIONS 5 // Send/Recv not included for now
typedef enum { ncclFuncBroadcast, ncclFuncReduce, ncclFuncAllGather, ncclFuncReduceScatter, ncclFuncAllReduce, ncclFuncSendRecv, ncclFuncSend, ncclFuncRecv, ncclNumFuncs} ncclFunc_t;

#define NCCL_NUM_ALGORITHMS 5 // Tree/Ring/CollNet*
#define NCCL_ALGO_TREE 0
#define NCCL_ALGO_RING 1
#define NCCL_ALGO_COLLNET_DIRECT 2
#define NCCL_ALGO_COLLNET_CHAIN 3
#define NCCL_ALGO_NVLS 4

#define NCCL_NUM_PROTOCOLS 3 // Simple/LL/LL128
#define NCCL_PROTO_LL 0
#define NCCL_PROTO_LL128 1
#define NCCL_PROTO_SIMPLE 2

// API to be implemented by external performance tuner
typedef struct {
  // Name of the performance tuner
  const char* name;

  // Initializes tuner states.
  // nRanks: number of ranks in current communicator. Each communicator initialize its own tuner.
  // nNodes: number of nodes in current communicator.
  // logFunction: a logFunction can be useful to integrate logging together with NCCL core.
  ncclResult_t (*init)(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction);

  // Gets info (algo, protocol, number of ctas and threads) for a given collective.
  // Inputs:
  //   - collType: collective type , e.g., allreduce, allgather…
  //   - nBytes: collective size in bytes
  // Outputs:
  //   - algo: selected algorithm to be used for the given collective
  //   - protocol: selected protocol to be used for the given collective
  //   - nChannels: number of channels to be used for the given collective
  //   - nThreads: number of threads to be used for the given collective
  //
  // If getCollInfo() returns other than ncclSuccess, NCCL core will fall back
  // to the default topo-based tuning for the given collective.
  ncclResult_t (*getCollInfo)(ncclFunc_t collType, size_t nBytes, int *algo,
                              int *protocol, int *nChannels, int *nThreads);

  // Terminates the plugin and cleans up any resources that the plugin allocated.
  ncclResult_t (*destroy)();
} ncclPerformanceTuner_v1_t;

typedef ncclPerformanceTuner_v1_t ncclPerformanceTuner_t;
#endif
