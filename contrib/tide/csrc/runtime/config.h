// SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>

namespace tide {

// Every TIDE_* environment knob of the extension (with the Python-side ones, by class: docs/usage.md,
// Environment variables). A Comm parses the environment when it is constructed (a later communicator sees
// later changes, which bench/calibrate.py relies on) and its reduce-scatter sizing may then come from
// tide/solver.py (Comm::setRsTuning), except for knobs set in the environment; code without a communicator
// (copy descriptors, kernel selection) reads envConfig(), parsed once. Fault injection: runtime/comm_internal.h.
struct Config {
  // -- deployment ---------------------------------------------------------------------------------
  int rsDevctl = 0;              // TIDE_RS_NET=device|host: device = the pipelined network phase runs in GIN device kernels
                                 // (GDAKI only); host (default) = rounds issued by the host
  double rsDevctlTimeoutS = 300; // TIDE_RS_NET_TIMEOUT_S: a device wait longer than this is a protocol fault (process exits)
  // -- tuning, calibration and debug: the solver or an automatic rule picks the sizing ones unless set
  int rmaCtx = 4;                // TIDE_RMA_CTX: QPs a shard is striped over (num_rma_ctx > 0 wins)
  int paceMode = -1;             // TIDE_NIC_PACE: 1 always, 0 never, auto (-1): shards of at least two leads
  size_t paceLead = 8u << 20;    // TIDE_NIC_PACE_LEAD_MIB: bytes still in flight when the pace signal leaves
  int nicGroups = 4;             // TIDE_NIC_GROUPS: ranges the last all-gather round is sent and scattered in
  int commCtas = 1;              // TIDE_COMM_CTAS: channel cap of the NCCL communicator (0 = NCCL default)
  int rsCtas = 0;                // TIDE_RS_CTAS: K_sum CTAs (0: 4; 8 on a single node, 12 with one rank per node)
  int rsMcCtas = 4;              // TIDE_RS_MC_CTAS: CTAs of one wave's multicast landing (0: skip it, debug)
  int rsKway = 2;                // TIDE_RS_KWAY: remote destinations per NIC round (solver: 2, doubled up to rs_k_max on
                                 // many nodes; Calib.incast_max with one rank per node)
  size_t rsChunk = 16u << 20;    // TIDE_RS_CHUNK_MIB: staging chunk hint per source
  size_t stridedRows = 4;        // TIDE_STRIDED_ROWS: rows from which a strided descriptor replaces flat ones (0: never)
  int rsRailParts = 4;           // TIDE_RS_RAIL_PARTS: one-rank-per-node reduce-scatter in n staging parts (blocks 1/n of a piece)
  int rsParts = 1;               // TIDE_RS_PARTS: multicast reduce-scatter in n passes over the sub-slices (inbox / outbox
                                 // blocks 1/n of a sub-slice, two inbox parities from n = 2 on)
  int rsOutbox = 0;              // TIDE_RS_OUTBOX: outbox blocks per bank (at least two rounds' worth, up to the remote
                                 // destinations); 0: two rounds' worth on one-phase calls, a full bank on pipelined ones
  int rsStageSlots = 2;          // TIDE_RS_STAGE_SLOTS: local-staging slots (2..kRsMaxStageSlots); more let the pushes run
                                 // ahead of the sums (4 measured within noise of 2 at 8 nodes in both FSDP2 modes, 8 S more)
  int rsKernelPriority = 2;      // TIDE_RS_KERNEL_PRIORITY: levels above the default stream priority for the sum / landing
                                 // streams (persistent GEMMs give a higher-priority kernel the SMs at tile boundaries)
  int rsPipeline = 1;            // TIDE_RS_PIPELINE: 2 = multicast reduce-scatter in a read phase and a network phase, the
                                 // read of call e+1 overlapping the network phase of call e (two outbox / own-spare banks)
  int rsWaveNodes = 8;           // TIDE_RS_WAVE_NODES: nodes per landing wave; below the node count on a small fleet
  int rsDebug = 0;               // TIDE_RS_DEBUG: 1 serialize phases, 2 dump flags, 3 host time per call
  int rsDevctlDebug = 0;         // TIDE_RS_NET_DEBUG=1: the monitor prints where the wave kernel is every 5 s
  int rsDevctlTrace = 0;         // TIDE_RS_NET_TRACE=N: wave kernel stamps, each rank logs a summary every N calls
  bool traceAg = false;          // TIDE_TRACE_AG=1: GPU timestamps of the all-gather phases

  enum : unsigned { kEnvRsCtas = 1, kEnvRsMcCtas = 2, kEnvRsKway = 4, kEnvRsChunk = 8, kEnvRsParts = 16, kEnvRsOutbox = 32,
                    kEnvRsPipeline = 64 };
  static constexpr int kRsMaxParts = 64;
  static constexpr int kRsMaxStageSlots = 8;
  unsigned envSet = 0;           // the sizing knobs above that came from the environment

  static Config fromEnv();
};

const Config& envConfig();

}  // namespace tide
