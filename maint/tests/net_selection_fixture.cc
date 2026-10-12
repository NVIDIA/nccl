// SPDX-License-Identifier: Apache-2.0
// Compile the actual ncclTopoGetNetDev body with deterministic topology dependencies.
#include <cstdint>
#include <iostream>
#include <string>

using ncclResult_t = int;
constexpr int ncclSuccess = 0, ncclInternalError = 3;
constexpr int GPU = 0, NET = 1;
constexpr int PATH_NVL = 1, PATH_PXB = 4, PATH_P2C = 5, PATH_PXN = 6, PATH_SYS = 9;
constexpr int NCCL_TOPO_PATTERN_NVLS = 7;
enum ncclTopoGdrMode { ncclTopoGdrModeDisable, ncclTopoGdrModeDefault, ncclTopoGdrModePci };
#define NCCLCHECK(call) do { ncclResult_t result = (call); if (result != ncclSuccess) return result; } while (0)

struct ncclTopoLinkList { int type = PATH_SYS; };
struct ncclTopoNode {
  struct { int rank; } gpu;
  ncclTopoLinkList paths[2][2];
};
struct ncclTopoSystem {
  struct { int count; ncclTopoNode* nodes; } nodes[2];
  int systemId = 0;
};
struct ncclTopoGraph { int nChannels = 1, pattern = 0, intra[2] = {0, 1}; int64_t inter[2] = {101, 101}; };
struct ncclPeerInfo { int nvmlDev; };
struct ncclComm { ncclTopoSystem* topo; ncclPeerInfo* peerInfo; int rank = 0; };

static int pxnLevel = 2, crossNic = 1, pxnDisabled = 0;
static ncclTopoGdrMode sendGdr = ncclTopoGdrModeDefault;
static bool nicHasGpu = true, failReadCheck = false;
int ncclPxnDisable(ncclComm*) { return pxnDisabled; }
int ncclParamP2pPxnLevel() { return pxnLevel; }
int ncclParamCrossNic() { return crossNic; }
int ncclParamPxnC2c() { return 1; }
ncclResult_t getNvlsNetDev(ncclComm*, ncclTopoGraph*, int, int64_t* id) { *id = 101; return 0; }
ncclResult_t ncclTopoIdToNetDev(ncclTopoSystem*, int64_t id, int* dev) { *dev = int(id - 100); return 0; }
ncclResult_t ncclTopoGetLocalNet(ncclTopoSystem*, int rank, int channel, int64_t* id, int* dev) {
  *dev = (rank + channel) % 2; *id = 100 + *dev; return 0;
}
ncclResult_t ncclTopoDevToRank(ncclTopoSystem*, int, int dev, bool, int* rank) {
  if (dev < 0 || dev > 1) return ncclInternalError;
  *rank = dev; return 0;
}
ncclResult_t ncclTopoRankToIndex(ncclTopoSystem*, int rank, int* index, bool) { *index = rank; return 0; }
ncclResult_t ncclTopoIdToIndex(ncclTopoSystem*, int, int64_t id, int* index) { *index = int(id - 100); return 0; }
ncclResult_t ncclTopoGetLocalGpu(ncclTopoSystem*, int64_t id, int* gpu) {
  *gpu = nicHasGpu ? int(id - 100) : -1; return 0;
}
ncclResult_t ncclTopoGetIntermediateRank(ncclTopoSystem*, int rank, int64_t id, int* proxy) {
  *proxy = (id == 101 && rank == 0) ? 1 : rank; return 0;
}
ncclResult_t ncclTopoCheckGdr(ncclTopoSystem*, int, int64_t, int read, ncclTopoGdrMode* mode) {
  if (read && failReadCheck) return ncclInternalError;
  *mode = read ? sendGdr : ncclTopoGdrModeDefault; return 0;
}

#include "net_selection_under_test.inc"

int main() {
  ncclTopoNode gpuNodes[2] = {};
  gpuNodes[0].gpu.rank = 0; gpuNodes[1].gpu.rank = 1;
  gpuNodes[0].paths[NET][1].type = PATH_PXN;
  gpuNodes[1].paths[GPU][0].type = PATH_NVL;
  gpuNodes[1].paths[NET][1].type = PATH_PXB;
  ncclTopoSystem topo;
  topo.nodes[GPU] = {2, gpuNodes};
  ncclPeerInfo peers[2] = {{0}, {1}};
  ncclComm comm = {&topo, peers, 0};
  int failures = 0, cases = 0;
  auto check = [&](const char* name, int expectedProxy, int64_t expectedId, int expectedResult = 0,
                   ncclTopoGraph* graph = nullptr, bool nullOutputs = false, int peer = 1) {
    int proxy = -1, dev = -1; int64_t id = -1;
    auto result = ncclTopoGetNetDev(&comm, 0, graph, 0, peer,
                                    nullOutputs ? nullptr : &id, nullOutputs ? nullptr : &dev, &proxy);
    bool ok = result == expectedResult;
    if (result == 0) ok = ok && proxy == expectedProxy &&
        (nullOutputs || (id == expectedId && dev == expectedId - 100));
    std::cout << name << ": " << (ok ? "PASS" : "FAIL") << " result=" << result
              << " proxy=" << proxy << " net=" << id << "\n";
    ++cases; failures += !ok;
  };
  for (int level : {1, 2}) {
    pxnLevel = level;
    sendGdr = ncclTopoGdrModeDefault; crossNic = 1;
    check(level == 1 ? "level1_valid_pxn" : "level2_valid_pxn", 1, 101);
    sendGdr = ncclTopoGdrModePci;
    check(level == 1 ? "level1_pci_gdr" : "level2_pci_gdr", 1, 101);
    sendGdr = ncclTopoGdrModeDisable;
    check(level == 1 ? "level1_no_read_cross_nic" : "level2_no_read_cross_nic", 0, 100);
    crossNic = 0;
    check(level == 1 ? "level1_no_read_same_rail" : "level2_no_read_same_rail", 0, 101);
  }
  crossNic = 1; pxnLevel = 2; sendGdr = ncclTopoGdrModeDefault;
  pxnDisabled = 1; check("pxn_disabled", 0, 100); pxnDisabled = 0;
  pxnLevel = 0; check("level_zero", 0, 100); pxnLevel = 2;
  nicHasGpu = false; check("nic_without_proxy_gpu", 0, 100); nicHasGpu = true;
  peers[1].nvmlDev = 9; check("asymmetric_peer_mapping", 0, 100); peers[1].nvmlDev = 1;
  peers[1].nvmlDev = 0; check("local_proxy", 0, 100); peers[1].nvmlDev = 1;
  check("nullable_net_outputs", 1, 101, 0, nullptr, true);
  failReadCheck = true; check("gdr_error_propagation", -1, -1, ncclInternalError); failReadCheck = false;
  check("missing_peer", -1, -1, ncclInternalError, nullptr, false, -1);
  ncclTopoGraph graph;
  sendGdr = ncclTopoGdrModeDisable;
  check("graph_selection_unchanged", 1, 101, 0, &graph);
  graph.pattern = NCCL_TOPO_PATTERN_NVLS;
  check("nvls_selection_unchanged", 1, 101, 0, &graph);
  std::cout << "CASES=" << cases << " FAILURES=" << failures << "\n";
  return failures == 0 ? 0 : 1;
}
