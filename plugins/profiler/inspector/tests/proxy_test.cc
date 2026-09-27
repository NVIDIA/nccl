/* SPDX-License-Identifier: Apache-2.0 */
// Compile real callbacks, pools, ring and serializer; supply only platform
// logging/time/lock adapters. No CUDA or network emulation is involved.
#include <cstdlib>
#include "../inspector.h"
static void* testRealloc(void*, size_t);
#define realloc testRealloc
#include "../inspector_plugin.cc"
#undef realloc
#include "../inspector_json.cc"

static void testLog(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...) {}
ncclDebugLogger_t logFn = testLog;
bool enableNcclInspectorProxyNested = true;
bool enableNcclInspectorProxyStepDump = true;
bool enableNcclInspectorProxyEventTraceSnDump = true;
pid_t ncclInspectorPid = 123;
uint64_t ncclInspectorProxyPxnSkipped = 0;
// ASan retains the plugin interface table. Unused CUDA/collective entry points
// fail loudly if reached; they are not part of this host-only Proxy test.
bool enableNcclInspectorP2p = false;
bool enableNcclInspectorProxy = true;
bool requireKernelTiming = true;
size_t ncclInspectorDumpMinSizeBytes = 0;
inspectorResult_t inspectorGlobalInit(int) { abort(); }
inspectorResult_t inspectorGlobalFinalize() { abort(); }
inspectorResult_t inspectorAddComm(inspectorCommInfo**, const char*, uint64_t, int, int, int) { abort(); }
inspectorResult_t inspectorDelComm(inspectorCommInfo*) { abort(); }
void inspectorUpdateCollPerf(inspectorCompletedOpInfo*, inspectorCollInfo*) { abort(); }
void inspectorUpdateP2pPerf(inspectorCompletedOpInfo*, inspectorP2pInfo*) { abort(); }
void inspectorComputeOpBw(inspectorCommInfo*, inspectorCompletedOpInfo*) { abort(); }
ncclDataType_t inspectorStringToDatatype(const char*) { abort(); }
const char* inspectorErrorString(inspectorResult_t) { return "fixture error"; }
const char* get_git_version_info() { return "fixture"; }
bool inspectorIsDumpVerboseEnabled() { return false; }
const char* ncclFuncToString(ncclFunc_t) { abort(); }
const char* inspectorTimingSourceToString(inspectorTimingSource_t) { abort(); }
uint64_t inspectorGetTime() { static uint64_t t = 100; return __atomic_add_fetch(&t, 1, __ATOMIC_RELAXED); }
inspectorResult_t inspectorLockInit(pthread_rwlock_t* p) { return pthread_rwlock_init(p, nullptr) ? inspectorLockError : inspectorSuccess; }
inspectorResult_t inspectorLockDestroy(pthread_rwlock_t* p) { return pthread_rwlock_destroy(p) ? inspectorLockError : inspectorSuccess; }
inspectorResult_t inspectorLockRd(pthread_rwlock_t* p) { return pthread_rwlock_rdlock(p) ? inspectorLockError : inspectorSuccess; }
inspectorResult_t inspectorLockWr(pthread_rwlock_t* p) { return pthread_rwlock_wrlock(p) ? inspectorLockError : inspectorSuccess; }
inspectorResult_t inspectorUnlockRWLock(pthread_rwlock_t* p) { return pthread_rwlock_unlock(p) ? inspectorLockError : inspectorSuccess; }

#define check(ok) do { if (!(ok)) { fprintf(stderr, "proxy fixture failed at line %d: %s\n", __LINE__, #ok); abort(); } } while (0)
static bool failAllocation = false;
static size_t allocationCalls = 0;
static void* testRealloc(void* old, size_t bytes) {
  allocationCalls++;
  if (failAllocation) return nullptr;
  return std::realloc(old, bytes);
}
static inspectorCommInfo comm = {};
static inspectorCollInfo parent = {};
static jsonFileOutput* output;

static inspectorProxyOpInfo* startOp(int n) {
  ncclProfilerEventDescr_t d = {};
  d.parentObj = &parent;
  d.proxyOp.pid = ncclInspectorPid;
  d.proxyOp.nSteps = n;
  d.proxyOp.chunkSize = 100;
  d.proxyOp.isSend = 1;
  d.rank = 7;
  inspectorProxyOpInfo* op = nullptr;
  inspectorPluginProxyOpInfoInit(&op, &d);
  return op;
}

static inspectorProxyStepInfo* startStep(inspectorProxyOpInfo* op, int i) {
  ncclProfilerEventDescr_t d = {};
  d.parentObj = op;
  d.proxyStep.step = i;
  inspectorProxyStepInfo* step = nullptr;
  inspectorPluginProxyStepInfoInit(&step, &d);
  if (step) {
    ncclProfilerEventStateArgs_t a = {};
    a.proxyStep.transSize = 100;
    inspectorPluginRecordEventStateProxyStep(step, ncclProfilerProxyStepSendGPUWait, &a);
    inspectorPluginRecordEventStateProxyStep(step, ncclProfilerProxyStepSendPeerWait_v4, &a);
    inspectorPluginRecordEventStateProxyStep(step, ncclProfilerProxyStepSendWait, &a);
  }
  return step;
}

static std::vector<inspectorCompletedProxyRecord> drain() {
  std::vector<inspectorCompletedProxyRecord> records;
  check(inspectorLockWr(&comm.guard) == inspectorSuccess);
  check(inspectorRingDrain(&comm.completedProxyRing, records) == inspectorSuccess);
  check(inspectorUnlockRWLock(&comm.guard) == inspectorSuccess);
  std::vector<inspectorProxyDumpGroup> groups;
  check(inspectorGroupProxyRecords(records, groups) == inspectorSuccess);
  // Every retained nested Step must have metadata later in this SAME drain.
  for (size_t i = 0; i < records.size(); i++) {
    if (!records[i].nested || records[i].recordType != NCCL_INSP_PROXY_RECORD_STEP) continue;
    bool parentFound = false;
    for (size_t j = i + 1; j < records.size(); j++) {
      if (records[j].metadata.proxyOpSn == records[i].metadata.proxyOpSn
          && (records[j].recordType == NCCL_INSP_PROXY_RECORD_OP_SEGMENT
              || records[j].recordType == NCCL_INSP_PROXY_RECORD_OP)) {
        parentFound = true;
        break;
      }
    }
    check(parentFound);
  }
  return records;
}

static void emit(const char* name, const std::vector<inspectorCompletedProxyRecord>& records) {
  check(jsonStartObject(output) == jsonSuccess);
  check(jsonKey(output, "case") == jsonSuccess); check(jsonStr(output, name) == jsonSuccess);
  check(jsonKey(output, "records") == jsonSuccess); check(jsonStartList(output) == jsonSuccess);
  std::vector<inspectorProxyDumpGroup> groups;
  check(inspectorGroupProxyRecords(records, groups) == inspectorSuccess);
  for (const auto& group : groups) {
    check(inspectorCompletedProxy(output, group.parent, &group.steps) == inspectorSuccess);
  }
  check(jsonFinishList(output) == jsonSuccess); check(jsonFinishObject(output) == jsonSuccess);
  check(jsonNewline(output) == jsonSuccess);
}

static void emitTiming(const char* name, const inspectorCompletedProxyRecord& record) {
  char key[96];
  snprintf(key, sizeof(key), "timing_%s", name);
  emit(key, {record});
}

static inspectorCompletedProxyRecord step(bool send, unsigned mask) {
  inspectorCompletedProxyRecord record = {};
  record.recordType = NCCL_INSP_PROXY_RECORD_STEP;
  record.metadata.isSend = send;
  record.proxyStep.transSizeBytes = 200000;
  const int phases[] = {
    NCCL_INSP_EVT_TRK_PROXY_STEP_START,
    send ? NCCL_INSP_EVT_TRK_PROXY_STEP_SEND_GPU_WAIT : NCCL_INSP_EVT_TRK_PROXY_STEP_RECV_WAIT,
    send ? NCCL_INSP_EVT_TRK_PROXY_STEP_SEND_PEER_WAIT : NCCL_INSP_EVT_TRK_PROXY_STEP_RECV_FLUSH_WAIT,
    send ? NCCL_INSP_EVT_TRK_PROXY_STEP_SEND_WAIT : NCCL_INSP_EVT_TRK_PROXY_STEP_RECV_GPU_WAIT,
    NCCL_INSP_EVT_TRK_PROXY_STEP_STOP
  };
  const uint64_t times[] = {100, 110, 130, 160, 200};
  for (int i = 0; i < 5; i++) {
    record.proxyStep.evntTrace[phases[i]].sn = mask & (1u << i) ? i + 1 : 0;
    // Keep a nonzero timestamp even for missing states to test sn as the marker.
    record.proxyStep.evntTrace[phases[i]].ts = times[i];
  }
  return record;
}

static inspectorCompletedProxyRecord op(unsigned mask) {
  inspectorCompletedProxyRecord record = {};
  record.recordType = NCCL_INSP_PROXY_RECORD_OP;
  record.metadata.isSend = true;
  record.proxyOp.transSizeBytes = 200000;
  const uint64_t times[] = {100, 150, 300};
  for (int i = 0; i < NCCL_INSP_EVT_TRK_PROXY_OP_NEVT; i++) {
    record.proxyOp.evntTrace[i].sn = mask & (1u << i) ? i + 1 : 0;
    record.proxyOp.evntTrace[i].ts = times[i];
  }
  return record;
}

static void emitTimingCases() {
  char name[64];
  for (int send = 0; send < 2; send++) {
    for (unsigned mask = 0; mask < 32; mask++) {
      snprintf(name, sizeof(name), "%s_mask_%u", send ? "send" : "recv", mask);
      emitTiming(name, step(send, mask));
    }
    auto record = step(send, 31);
    for (auto& event : record.proxyStep.evntTrace) event.ts = 0;
    emitTiming(send ? "send_zero_time" : "recv_zero_time", record);
    record = step(send, 31);
    record.proxyStep.evntTrace[NCCL_INSP_EVT_TRK_PROXY_STEP_STOP].ts = 90;
    emitTiming(send ? "send_reversed" : "recv_reversed", record);
    record = step(send, 31);
    int middle = send ? NCCL_INSP_EVT_TRK_PROXY_STEP_SEND_PEER_WAIT
                      : NCCL_INSP_EVT_TRK_PROXY_STEP_RECV_FLUSH_WAIT;
    record.proxyStep.evntTrace[middle].ts = 105;
    emitTiming(send ? "send_phase_reversed" : "recv_phase_reversed", record);
  }
  for (unsigned mask = 0; mask < 8; mask++) {
    snprintf(name, sizeof(name), "op_mask_%u", mask);
    emitTiming(name, op(mask));
  }
  auto record = op(7);
  record.proxyOp.evntTrace[NCCL_INSP_EVT_TRK_PROXY_OP_STOP].ts = 100;
  emitTiming("op_zero_duration", record);
  record.proxyOp.evntTrace[NCCL_INSP_EVT_TRK_PROXY_OP_STOP].ts = 99;
  emitTiming("op_reversed", record);
  record = op(7);
  record.proxyOp.evntTrace[NCCL_INSP_EVT_TRK_PROXY_OP_START].ts = 0;
  emitTiming("op_zero_start", record);
  record = op(7);
  record.proxyOp.transSizeBytes = 0;
  emitTiming("op_zero_bytes", record);
}

static void finishSteps(inspectorProxyOpInfo* op, int n) {
  for (int i = 0; i < n; i++) {
    auto* step = startStep(op, i);
    check(step != nullptr);
    check(inspectorPluginStopEventProxyStep(step) == ncclSuccess);
    check(g_eventPool.proxyStepAllocCount == 0);
  }
}

static void initRing(int size) {
  inspectorRingFinalize(&comm.completedProxyRing);
  check(inspectorRingInit(&comm.completedProxyRing, size, sizeof(inspectorCompletedProxyRecord),
                          inspectorProxyRecordStepCount) == inspectorSuccess);
}

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  enableNcclInspectorProxyEventTraceSnDump = atoi(argv[3]) != 0;
  check(jsonInitFileOutput(&output, argv[1]) == jsonSuccess);
  emitTimingCases();
  inspectorEventPoolConfig config = {};
  config.collPoolSize = config.p2pPoolSize = config.commPoolSize = 2;
  config.proxyOpPoolSize = config.proxyStepPoolSize = 2;
  config.enableProxy = true;
  check(inspectorEventPoolInit(config) == inspectorSuccess);
  check(inspectorLockInit(&comm.guard) == inspectorSuccess);
  check(inspectorLockInit(&parent.guard) == inspectorSuccess);
  parent.type = ncclProfileColl; parent.commInfo = &comm; parent.sn = 9;
  initRing(1024);
  // A full eight-Step batch must leave no tail, and the ring must contain
  // independent small records rather than one embedded Step array.
  auto* boundary = startOp(8);
  finishSteps(boundary, 8);
  check(boundary->nBufferedSteps == 0);
  auto batch = drain();
  check(batch.size() == 10); // start + eight Steps + segment metadata
  check(batch.back().recordType == NCCL_INSP_PROXY_RECORD_OP_SEGMENT);
  for (size_t i = 1; i < 9; i++) check(batch[i].recordType == NCCL_INSP_PROXY_RECORD_STEP);
  inspectorPluginStopEventProxyOp(boundary);
  drain();
  for (int n : {0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 32, 33, 512}) {
    auto* op = startOp(n); check(op != nullptr);
    finishSteps(op, n);
    if (n == 0) check(op->bufferedSteps == nullptr && op->bufferedStepCapacity == 0);
    uint32_t capacity = n == 0 ? 0 : n == 1 ? 1 : n == 2 ? 2 : n <= 4 ? 4 : 8;
    check(op->bufferedStepCapacity == capacity);
    check(inspectorPluginStopEventProxyOp(op) == ncclSuccess);
    check(g_eventPool.proxyOpAllocCount == 0);
    char name[32]; snprintf(name, sizeof(name), "nested_%d", n);
    emit(name, drain());
  }
  // Start is visible before any Step; a stalled Op does not flush its tail.
  auto* op = startOp(33); emit("start_visible", drain());
  finishSteps(op, 17); emit("stalled_tail", drain());
  check(op->nBufferedSteps == 1);
  inspectorPluginStopEventProxyOp(op); emit("stalled_final", drain());

  // Stop may precede the last child's release. Finalize exactly once.
  op = startOp(16); finishSteps(op, 15);
  auto* last = startStep(op, 15); check(last != nullptr);
  inspectorPluginStopEventProxyOp(op);
  check(g_eventPool.proxyOpAllocCount == 1);
  inspectorPluginStopEventProxyStep(last);
  check(g_eventPool.proxyOpAllocCount == 0);
  emit("stop_before_child", drain());

  enableNcclInspectorProxyStepDump = false;
  allocationCalls = 0;
  op = startOp(33); finishSteps(op, 33);
  check(op->nBufferedSteps == 0 && op->bufferedSteps == nullptr);
  inspectorPluginStopEventProxyOp(op); emit("op_only", drain());
  enableNcclInspectorProxyStepDump = true;
  enableNcclInspectorProxyNested = false;
  op = startOp(33); finishSteps(op, 33);
  check(op->bufferedSteps == nullptr && allocationCalls == 0);
  inspectorPluginStopEventProxyOp(op); emit("flat", drain());
  enableNcclInspectorProxyNested = true;

  initRing(1);
  op = startOp(17); finishSteps(op, 17); inspectorPluginStopEventProxyOp(op);
  check(comm.completedProxyRing.dropped == 20 && comm.completedProxyRing.droppedItems == 17);
  emit("overwrite", drain());
  // Empty final summaries carry no Step weight; reset clears all counters.
  initRing(1); op = startOp(16); finishSteps(op, 16); inspectorPluginStopEventProxyOp(op);
  check(comm.completedProxyRing.droppedItems == 16);
  op = startOp(0); inspectorPluginStopEventProxyOp(op);
  check(comm.completedProxyRing.droppedItems == 16); drain();
  initRing(1); enableNcclInspectorProxyNested = false;
  op = startOp(2); finishSteps(op, 2); inspectorPluginStopEventProxyOp(op);
  check(comm.completedProxyRing.dropped == 2 && comm.completedProxyRing.droppedItems == 2); drain();
  enableNcclInspectorProxyNested = true;

  // Every ring capacity, including less than a batch, retains parent metadata.
  for (int capacity = 1; capacity <= 20; capacity++) {
    initRing(capacity);
    auto* op1 = startOp(9); auto* op2 = startOp(17);
    finishSteps(op1, 9); finishSteps(op2, 17);
    inspectorPluginStopEventProxyOp(op1); inspectorPluginStopEventProxyOp(op2);
    auto retained = drain();
    uint64_t steps = 0;
    for (const auto& r : retained) steps += r.recordType == NCCL_INSP_PROXY_RECORD_STEP;
    check(steps + comm.completedProxyRing.droppedItems == 26);
  }

  // First-allocation failure loses detail, not completed bytes or references.
  initRing(1024);
  op = startOp(3);
  failAllocation = true;
  auto* lost = startStep(op, 0); inspectorPluginStopEventProxyStep(lost);
  check(op->nStepsDropped == 1 && op->nStepsCompleted == 1 && op->transSizeBytes == 100);
  failAllocation = false;
  auto* kept = startStep(op, 1); inspectorPluginStopEventProxyStep(kept);
  // Expansion failure flushes the existing one-Step buffer, then reuses it.
  failAllocation = true;
  kept = startStep(op, 2); inspectorPluginStopEventProxyStep(kept);
  check(op->bufferedStepCapacity == 1 && op->nBufferedSteps == 1 && op->nStepsDropped == 1);
  failAllocation = false;
  inspectorPluginStopEventProxyOp(op); emit("allocation_loss", drain());
  op = startOp(0);
  check(op->bufferedSteps == nullptr && op->bufferedStepCapacity == 0);
  inspectorPluginStopEventProxyOp(op); drain();

  check(g_eventPool.proxyOpAllocCount == 0 && g_eventPool.proxyStepAllocCount == 0);

  // Real dump markers must count grouped JSON objects, not drained ring slots.
  initRing(128);
  comm.rank = 7;
  jsonFileOutput* stream = nullptr;
  check(jsonInitFileOutput(&stream, argv[2]) == jsonSuccess);
  bool wrote = false;
  op = startOp(33); finishSteps(op, 33); inspectorPluginStopEventProxyOp(op);
  check(inspectorCommInfoDump(stream, &comm, &wrote) == inspectorSuccess && wrote);
  check(!inspectorRingNonEmpty(&comm.completedProxyRing));
  check(inspectorCommInfoDump(stream, &comm, &wrote) == inspectorSuccess && !wrote);
  op = startOp(17); finishSteps(op, 17);
  check(inspectorCommInfoDump(stream, &comm, &wrote) == inspectorSuccess && wrote);
  inspectorPluginStopEventProxyOp(op);
  check(inspectorCommInfoDump(stream, &comm, &wrote) == inspectorSuccess && wrote);
  enableNcclInspectorProxyNested = false;
  op = startOp(3); finishSteps(op, 3); inspectorPluginStopEventProxyOp(op);
  check(inspectorCommInfoDump(stream, &comm, &wrote) == inspectorSuccess && wrote);
  enableNcclInspectorProxyNested = true;
  check(jsonFinalizeFileOutput(stream) == jsonSuccess);
  inspectorRingFinalize(&comm.completedProxyRing);
  check(inspectorLockDestroy(&comm.guard) == inspectorSuccess);
  check(inspectorLockDestroy(&parent.guard) == inspectorSuccess);
  // Quiescent pool teardown also releases an unfinished Op's lazy buffer.
  auto* unfinished = inspectorEventPoolAllocProxyOp();
  check(unfinished != nullptr);
  unfinished->bufferedSteps = static_cast<inspectorProxyStepSnapshot*>(malloc(sizeof(inspectorProxyStepSnapshot)));
  inspectorEventPoolFinalize();
  check(jsonFinalizeFileOutput(output) == jsonSuccess);
}
