#include "inspector_json.h"
#include "inspector_ring.h"
#include "profiler.h"

#include <unistd.h>
#include <algorithm>
#include <unordered_map>
#include <vector>

#define JSON_CHK(expr)                                          \
  do {                                                          \
    const jsonResult_t res = (expr);                            \
    if (res != jsonSuccess) {                                   \
      INFO_INSPECTOR("jsonError: %s\n", jsonErrorString(res));  \
      return inspectorJsonError;                                \
    }                                                           \
  } while (0)

#define JSON_CHK_GOTO(expr, res, label)                                 \
  do {                                                                  \
    const jsonResult_t macro_res = (expr);                              \
    if (macro_res != jsonSuccess) {                                     \
      INFO_INSPECTOR("jsonError: %s\n", jsonErrorString(macro_res));    \
      res = inspectorJsonError;                                         \
      goto label;                                                       \
    }                                                                   \
  } while (0)

static inspectorResult_t inspectorCommInfoHeader(jsonFileOutput* jfo,
                                                 struct inspectorCommInfo* commInfo) {
  JSON_CHK(jsonStartObject(jfo));
  JSON_CHK(jsonKey(jfo, "id")); JSON_CHK(jsonStr(jfo, commInfo->commHashStr));
  const char* commName
    = (commInfo->commName && commInfo->commName[0]) ? commInfo->commName : "unknown";
  JSON_CHK(jsonKey(jfo, "comm_name")); JSON_CHK(jsonStr(jfo, commName));
  JSON_CHK(jsonKey(jfo, "rank")); JSON_CHK(jsonInt(jfo, commInfo->rank));
  JSON_CHK(jsonKey(jfo, "n_ranks")); JSON_CHK(jsonInt(jfo, commInfo->nranks));
  JSON_CHK(jsonKey(jfo, "nnodes")); JSON_CHK(jsonUint64(jfo, commInfo->nnodes));
  JSON_CHK(jsonFinishObject(jfo));
  return inspectorSuccess;
}

/*
 * Description:
 *
 *   Writes metadata header information to the JSON output.
 *
 * Thread Safety:
 *   Not thread-safe (should be called with proper locking).
 *
 * Input:
 *   jsonFileOutput* jfo - JSON output handle.
 *
 * Output:
 *   Metadata header is written to JSON output.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 *
 */
static inspectorResult_t inspectorCommInfoMetaHeader(jsonFileOutput* jfo) {
  JSON_CHK(jsonStartObject(jfo));
  {
    JSON_CHK(jsonKey(jfo, "inspector_output_format_version")); JSON_CHK(jsonStr(jfo, "v4.4"));
    JSON_CHK(jsonKey(jfo, "git_rev")); JSON_CHK(jsonStr(jfo, get_git_version_info()));
    JSON_CHK(jsonKey(jfo, "rec_mechanism")); JSON_CHK(jsonStr(jfo, "nccl_profiler_interface"));
    JSON_CHK(jsonKey(jfo, "dump_timestamp_us")); JSON_CHK(jsonUint64(jfo, inspectorGetTime()));
    char hostname[256];
    gethostname(hostname, 255);
    JSON_CHK(jsonKey(jfo, "hostname")); JSON_CHK(jsonStr(jfo, hostname));
    JSON_CHK(jsonKey(jfo, "pid")); JSON_CHK(jsonUint64(jfo, getpid()));
  }
  JSON_CHK(jsonFinishObject(jfo));
  return inspectorSuccess;
}

/*
 * Description:
 *
 *   Writes verbose information (event_trace) for a completed
 *   collective operation to the JSON output.
 *
 * Thread Safety:
 *   Not thread-safe (should be called with proper locking).
 *
 * Input:
 *   jsonFileOutput* jfo - JSON output handle.
 *   const struct inspectorCompletedOpInfo* op - completed collective info.
 *
 * Output:
 *   Verbose collective info is written to JSON output.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 *
 */
static inline inspectorResult_t inspectorCompletedCollVerbose(jsonFileOutput* jfo,
                                                              const struct inspectorCompletedOpInfo* op) {
  JSON_CHK(jsonKey(jfo, "event_trace_sn"));
  JSON_CHK(jsonStartObject(jfo));
  {
    JSON_CHK(jsonKey(jfo, "coll_start_sn")); JSON_CHK(jsonUint64(jfo, op->evtTrk.evntTrace[NCCL_INSP_EVT_TRK_OP_START].sn));
    JSON_CHK(jsonKey(jfo, "coll_stop_sn")); JSON_CHK(jsonUint64(jfo, op->evtTrk.evntTrace[NCCL_INSP_EVT_TRK_OP_STOP].sn));

    JSON_CHK(jsonKey(jfo, "kernel_events"));
    JSON_CHK(jsonStartList(jfo));
    for (uint32_t ch = 0; ch < op->evtTrk.nChannels; ch++) {
      JSON_CHK(jsonStartObject(jfo));
      JSON_CHK(jsonKey(jfo, "channel_id")); JSON_CHK(jsonInt(jfo, ch));
      JSON_CHK(jsonKey(jfo, "kernel_start_sn")); JSON_CHK(jsonUint64(jfo, op->evtTrk.kernelCh[ch].evntTrace[NCCL_INSP_EVT_TRK_KERNEL_START].sn));
      JSON_CHK(jsonKey(jfo, "kernel_stop_sn")); JSON_CHK(jsonUint64(jfo, op->evtTrk.kernelCh[ch].evntTrace[NCCL_INSP_EVT_TRK_KERNEL_STOP].sn));
      JSON_CHK(jsonKey(jfo, "kernel_record_sn")); JSON_CHK(jsonUint64(jfo, op->evtTrk.kernelCh[ch].evntTrace[NCCL_INSP_EVT_TRK_KERNEL_RECORD].sn));
      JSON_CHK(jsonFinishObject(jfo));
    }
    JSON_CHK(jsonFinishList(jfo));
  }
  JSON_CHK(jsonFinishObject(jfo));

  JSON_CHK(jsonKey(jfo, "event_trace_ts"));
  JSON_CHK(jsonStartObject(jfo));
  {
    JSON_CHK(jsonKey(jfo, "coll_start_ts")); JSON_CHK(jsonUint64(jfo, op->evtTrk.evntTrace[NCCL_INSP_EVT_TRK_OP_START].ts));
    JSON_CHK(jsonKey(jfo, "coll_stop_ts")); JSON_CHK(jsonUint64(jfo, op->evtTrk.evntTrace[NCCL_INSP_EVT_TRK_OP_STOP].ts));

    JSON_CHK(jsonKey(jfo, "kernel_events"));
    JSON_CHK(jsonStartList(jfo));
    for (uint32_t ch = 0; ch < op->evtTrk.nChannels; ch++) {
      JSON_CHK(jsonStartObject(jfo));
      JSON_CHK(jsonKey(jfo, "channel_id")); JSON_CHK(jsonInt(jfo, ch));
      JSON_CHK(jsonKey(jfo, "kernel_start_ts")); JSON_CHK(jsonUint64(jfo, op->evtTrk.kernelCh[ch].evntTrace[NCCL_INSP_EVT_TRK_KERNEL_START].ts));
      JSON_CHK(jsonKey(jfo, "kernel_stop_ts")); JSON_CHK(jsonUint64(jfo, op->evtTrk.kernelCh[ch].evntTrace[NCCL_INSP_EVT_TRK_KERNEL_STOP].ts));
      JSON_CHK(jsonKey(jfo, "kernel_record_ts")); JSON_CHK(jsonUint64(jfo, op->evtTrk.kernelCh[ch].evntTrace[NCCL_INSP_EVT_TRK_KERNEL_RECORD].ts));
      JSON_CHK(jsonFinishObject(jfo));
    }
    JSON_CHK(jsonFinishList(jfo));
  }
  JSON_CHK(jsonFinishObject(jfo));

  return inspectorSuccess;
}

static inline inspectorResult_t inspectorCompletedP2pVerbose(jsonFileOutput* jfo,
                                                             const struct inspectorCompletedOpInfo* op) {
  JSON_CHK(jsonKey(jfo, "event_trace_sn"));
  JSON_CHK(jsonStartObject(jfo));
  {
    JSON_CHK(jsonKey(jfo, "p2p_start_sn")); JSON_CHK(jsonUint64(jfo, op->evtTrk.evntTrace[NCCL_INSP_EVT_TRK_OP_START].sn));
    JSON_CHK(jsonKey(jfo, "p2p_stop_sn")); JSON_CHK(jsonUint64(jfo, op->evtTrk.evntTrace[NCCL_INSP_EVT_TRK_OP_STOP].sn));

    JSON_CHK(jsonKey(jfo, "kernel_events"));
    JSON_CHK(jsonStartList(jfo));
    for (uint32_t ch = 0; ch < op->evtTrk.nChannels; ch++) {
      JSON_CHK(jsonStartObject(jfo));
      JSON_CHK(jsonKey(jfo, "channel_id")); JSON_CHK(jsonInt(jfo, ch));
      JSON_CHK(jsonKey(jfo, "kernel_start_sn")); JSON_CHK(jsonUint64(jfo, op->evtTrk.kernelCh[ch].evntTrace[NCCL_INSP_EVT_TRK_KERNEL_START].sn));
      JSON_CHK(jsonKey(jfo, "kernel_stop_sn")); JSON_CHK(jsonUint64(jfo, op->evtTrk.kernelCh[ch].evntTrace[NCCL_INSP_EVT_TRK_KERNEL_STOP].sn));
      JSON_CHK(jsonKey(jfo, "kernel_record_sn")); JSON_CHK(jsonUint64(jfo, op->evtTrk.kernelCh[ch].evntTrace[NCCL_INSP_EVT_TRK_KERNEL_RECORD].sn));
      JSON_CHK(jsonFinishObject(jfo));
    }
    JSON_CHK(jsonFinishList(jfo));
  }
  JSON_CHK(jsonFinishObject(jfo));

  JSON_CHK(jsonKey(jfo, "event_trace_ts"));
  JSON_CHK(jsonStartObject(jfo));
  {
    JSON_CHK(jsonKey(jfo, "p2p_start_ts")); JSON_CHK(jsonUint64(jfo, op->evtTrk.evntTrace[NCCL_INSP_EVT_TRK_OP_START].ts));
    JSON_CHK(jsonKey(jfo, "p2p_stop_ts")); JSON_CHK(jsonUint64(jfo, op->evtTrk.evntTrace[NCCL_INSP_EVT_TRK_OP_STOP].ts));

    JSON_CHK(jsonKey(jfo, "kernel_events"));
    JSON_CHK(jsonStartList(jfo));
    for (uint32_t ch = 0; ch < op->evtTrk.nChannels; ch++) {
      JSON_CHK(jsonStartObject(jfo));
      JSON_CHK(jsonKey(jfo, "channel_id")); JSON_CHK(jsonInt(jfo, ch));
      JSON_CHK(jsonKey(jfo, "kernel_start_ts")); JSON_CHK(jsonUint64(jfo, op->evtTrk.kernelCh[ch].evntTrace[NCCL_INSP_EVT_TRK_KERNEL_START].ts));
      JSON_CHK(jsonKey(jfo, "kernel_stop_ts")); JSON_CHK(jsonUint64(jfo, op->evtTrk.kernelCh[ch].evntTrace[NCCL_INSP_EVT_TRK_KERNEL_STOP].ts));
      JSON_CHK(jsonKey(jfo, "kernel_record_ts")); JSON_CHK(jsonUint64(jfo, op->evtTrk.kernelCh[ch].evntTrace[NCCL_INSP_EVT_TRK_KERNEL_RECORD].ts));
      JSON_CHK(jsonFinishObject(jfo));
    }
    JSON_CHK(jsonFinishList(jfo));
  }
  JSON_CHK(jsonFinishObject(jfo));

  return inspectorSuccess;
}

/*
 * Description:
 *
 *   Writes completed collective operation information to the JSON
 *   output.
 *
 * Thread Safety:
 *   Not thread-safe (should be called with proper locking).
 *
 * Input:
 *   jsonFileOutput* jfo - JSON output handle.
 *   const struct inspectorCompletedOpInfo* op - completed collective info.
 *
 * Output:
 *   Collective info is written to JSON output.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 *
 */
static inline inspectorResult_t inspectorCompletedColl(jsonFileOutput* jfo,
                                                       const struct inspectorCompletedOpInfo* op) {
  JSON_CHK(jsonStartObject(jfo));
  {
    JSON_CHK(jsonKey(jfo, "coll")); JSON_CHK(jsonStr(jfo, ncclFuncToString(op->func)));

    JSON_CHK(jsonKey(jfo, "coll_algo")); JSON_CHK(jsonStr(jfo, op->algo ? op->algo : "unknown"));

    JSON_CHK(jsonKey(jfo, "coll_proto")); JSON_CHK(jsonStr(jfo, op->proto ? op->proto : "unknown"));

    JSON_CHK(jsonKey(jfo, "coll_sn")); JSON_CHK(jsonUint64(jfo, op->sn));

    JSON_CHK(jsonKey(jfo, "coll_msg_size_bytes")); JSON_CHK(jsonUint64(jfo, op->msgSizeBytes));

    JSON_CHK(jsonKey(jfo, "coll_exec_time_us")); JSON_CHK(jsonUint64(jfo, op->execTimeUsecs));

    JSON_CHK(jsonKey(jfo, "coll_timing_source")); JSON_CHK(jsonStr(jfo, inspectorTimingSourceToString(op->timingSource)));

    JSON_CHK(jsonKey(jfo, "coll_algobw_gbs")); JSON_CHK(jsonDouble(jfo, op->algoBwGbs));

    JSON_CHK(jsonKey(jfo, "coll_busbw_gbs")); JSON_CHK(jsonDouble(jfo, op->busBwGbs));

    if (inspectorIsDumpVerboseEnabled()) {
      INS_CHK(inspectorCompletedCollVerbose(jfo, op));
    }
  }
  JSON_CHK(jsonFinishObject(jfo));

  return inspectorSuccess;
}

static inline inspectorResult_t inspectorCompletedP2p(jsonFileOutput* jfo,
                                                      const struct inspectorCompletedOpInfo* op) {
  JSON_CHK(jsonStartObject(jfo));
  {
    JSON_CHK(jsonKey(jfo, "p2p")); JSON_CHK(jsonStr(jfo, ncclFuncToString(op->func)));

    JSON_CHK(jsonKey(jfo, "p2p_sn")); JSON_CHK(jsonUint64(jfo, op->sn));

    JSON_CHK(jsonKey(jfo, "p2p_peer")); JSON_CHK(jsonInt(jfo, op->peer));

    JSON_CHK(jsonKey(jfo, "p2p_msg_size_bytes")); JSON_CHK(jsonUint64(jfo, op->msgSizeBytes));

    JSON_CHK(jsonKey(jfo, "p2p_exec_time_us")); JSON_CHK(jsonUint64(jfo, op->execTimeUsecs));

    JSON_CHK(jsonKey(jfo, "p2p_timing_source")); JSON_CHK(jsonStr(jfo, inspectorTimingSourceToString(op->timingSource)));

    JSON_CHK(jsonKey(jfo, "p2p_algobw_gbs")); JSON_CHK(jsonDouble(jfo, op->algoBwGbs));

    JSON_CHK(jsonKey(jfo, "p2p_busbw_gbs")); JSON_CHK(jsonDouble(jfo, op->busBwGbs));

    if (inspectorIsDumpVerboseEnabled()) {
      INS_CHK(inspectorCompletedP2pVerbose(jfo, op));
    }
  }
  JSON_CHK(jsonFinishObject(jfo));

  return inspectorSuccess;
}

static const char* inspectorProxyParentTypeToString(uint64_t parentType) {
  if (parentType == ncclProfileColl) return "coll";
  if (parentType == ncclProfileP2p) return "p2p";
  return "unknown";
}

static bool inspectorProxyElapsedTime(const struct inspectorEventTraceInfo* trace,
                                       int startIndex, int stopIndex,
                                       uint64_t* elapsedUsecs) {
  const struct inspectorEventTraceInfo& start = trace[startIndex];
  const struct inspectorEventTraceInfo& stop = trace[stopIndex];
  // sn, not ts, distinguishes an observed callback from an unfired state.
  // Check ordering before subtracting unsigned timestamps (the clock may move).
  if (start.sn == 0 || stop.sn == 0 || stop.ts < start.ts) return false;
  *elapsedUsecs = stop.ts - start.ts;
  return true;
}

static inspectorResult_t inspectorProxyTraceTimestamp(jsonFileOutput* jfo,
                                                       const char* key,
                                                       const struct inspectorEventTraceInfo& event) {
  if (event.sn == 0) return inspectorSuccess;
  JSON_CHK(jsonKey(jfo, key));
  JSON_CHK(jsonUint64(jfo, event.ts));
  return inspectorSuccess;
}

static inspectorResult_t inspectorProxyDuration(jsonFileOutput* jfo, const char* key,
                                                 const struct inspectorEventTraceInfo* trace,
                                                 int startIndex, int stopIndex) {
  uint64_t elapsedUsecs;
  if (inspectorProxyElapsedTime(trace, startIndex, stopIndex, &elapsedUsecs)) {
    JSON_CHK(jsonKey(jfo, key));
    JSON_CHK(jsonUint64(jfo, elapsedUsecs));
  }
  return inspectorSuccess;
}

static inline inspectorResult_t inspectorCompletedProxyOpTrace(jsonFileOutput* jfo,
                                                               const struct inspectorCompletedProxyRecord* proxy) {
  const struct inspectorEventTraceInfo* trace = proxy->proxyOp.evntTrace;

  uint64_t totalUsecs;
  if (inspectorProxyElapsedTime(trace, NCCL_INSP_EVT_TRK_PROXY_OP_START,
                                 NCCL_INSP_EVT_TRK_PROXY_OP_STOP, &totalUsecs)) {
    JSON_CHK(jsonKey(jfo, "total_us"));
    JSON_CHK(jsonUint64(jfo, totalUsecs));
    if (totalUsecs > 0) {
      // Decimal GB/s from observed bytes and CPU callback elapsed microseconds.
      // Convert before multiplying to avoid integer overflow for long intervals.
      double bandwidth = static_cast<double>(proxy->proxyOp.transSizeBytes)
                         / (static_cast<double>(totalUsecs) * 1000.0);
      JSON_CHK(jsonKey(jfo, "op_bw_gbs"));
      JSON_CHK(jsonDouble(jfo, bandwidth));
    }
  }

  if (enableNcclInspectorProxyEventTraceSnDump) {
    JSON_CHK(jsonKey(jfo, "event_trace_sn"));
    JSON_CHK(jsonStartObject(jfo));
    JSON_CHK(jsonKey(jfo, "proxy_op_start_sn"));
    JSON_CHK(jsonUint64(jfo, trace[NCCL_INSP_EVT_TRK_PROXY_OP_START].sn));
    JSON_CHK(jsonKey(jfo, "proxy_op_in_progress_sn"));
    JSON_CHK(jsonUint64(jfo, trace[NCCL_INSP_EVT_TRK_PROXY_OP_IN_PROGRESS].sn));
    JSON_CHK(jsonKey(jfo, "proxy_op_stop_sn"));
    JSON_CHK(jsonUint64(jfo, trace[NCCL_INSP_EVT_TRK_PROXY_OP_STOP].sn));
    JSON_CHK(jsonFinishObject(jfo));
  }

  JSON_CHK(jsonKey(jfo, "event_trace_ts"));
  JSON_CHK(jsonStartObject(jfo));
  {
    INS_CHK(inspectorProxyTraceTimestamp(jfo, "proxy_op_start_ts",
                                          trace[NCCL_INSP_EVT_TRK_PROXY_OP_START]));
    INS_CHK(inspectorProxyTraceTimestamp(jfo, "proxy_op_in_progress_ts",
                                          trace[NCCL_INSP_EVT_TRK_PROXY_OP_IN_PROGRESS]));
    INS_CHK(inspectorProxyTraceTimestamp(jfo, "proxy_op_stop_ts",
                                          trace[NCCL_INSP_EVT_TRK_PROXY_OP_STOP]));
  }
  JSON_CHK(jsonFinishObject(jfo));
  return inspectorSuccess;
}

static inline inspectorResult_t inspectorCompletedProxyStepTrace(jsonFileOutput* jfo,
                                                                 const struct inspectorProxyStepSnapshot* step,
                                                                 bool isSend) {
  const struct inspectorEventTraceInfo* trace = step->evntTrace;

  INS_CHK(inspectorProxyDuration(jfo, "total_us", trace,
                                  NCCL_INSP_EVT_TRK_PROXY_STEP_START,
                                  NCCL_INSP_EVT_TRK_PROXY_STEP_STOP));
  if (isSend) {
    INS_CHK(inspectorProxyDuration(jfo, "gpu_produce_us", trace,
                                    NCCL_INSP_EVT_TRK_PROXY_STEP_SEND_GPU_WAIT,
                                    NCCL_INSP_EVT_TRK_PROXY_STEP_SEND_PEER_WAIT));
    INS_CHK(inspectorProxyDuration(jfo, "peer_credit_us", trace,
                                    NCCL_INSP_EVT_TRK_PROXY_STEP_SEND_PEER_WAIT,
                                    NCCL_INSP_EVT_TRK_PROXY_STEP_SEND_WAIT));
    INS_CHK(inspectorProxyDuration(jfo, "wire_us", trace,
                                    NCCL_INSP_EVT_TRK_PROXY_STEP_SEND_WAIT,
                                    NCCL_INSP_EVT_TRK_PROXY_STEP_STOP));
  } else {
    INS_CHK(inspectorProxyDuration(jfo, "wire_us", trace,
                                    NCCL_INSP_EVT_TRK_PROXY_STEP_RECV_WAIT,
                                    NCCL_INSP_EVT_TRK_PROXY_STEP_RECV_FLUSH_WAIT));
    INS_CHK(inspectorProxyDuration(jfo, "flush_us", trace,
                                    NCCL_INSP_EVT_TRK_PROXY_STEP_RECV_FLUSH_WAIT,
                                    NCCL_INSP_EVT_TRK_PROXY_STEP_RECV_GPU_WAIT));
    INS_CHK(inspectorProxyDuration(jfo, "gpu_consume_us", trace,
                                    NCCL_INSP_EVT_TRK_PROXY_STEP_RECV_GPU_WAIT,
                                    NCCL_INSP_EVT_TRK_PROXY_STEP_STOP));
  }

  if (enableNcclInspectorProxyEventTraceSnDump) {
    JSON_CHK(jsonKey(jfo, "event_trace_sn"));
    JSON_CHK(jsonStartObject(jfo));
    JSON_CHK(jsonKey(jfo, "proxy_step_start_sn"));
    JSON_CHK(jsonUint64(jfo, trace[NCCL_INSP_EVT_TRK_PROXY_STEP_START].sn));
    if (isSend) {
      JSON_CHK(jsonKey(jfo, "send_gpu_wait_sn"));
      JSON_CHK(jsonUint64(jfo, trace[NCCL_INSP_EVT_TRK_PROXY_STEP_SEND_GPU_WAIT].sn));
      JSON_CHK(jsonKey(jfo, "send_peer_wait_sn"));
      JSON_CHK(jsonUint64(jfo, trace[NCCL_INSP_EVT_TRK_PROXY_STEP_SEND_PEER_WAIT].sn));
      JSON_CHK(jsonKey(jfo, "send_wait_sn"));
      JSON_CHK(jsonUint64(jfo, trace[NCCL_INSP_EVT_TRK_PROXY_STEP_SEND_WAIT].sn));
    } else {
      JSON_CHK(jsonKey(jfo, "recv_wait_sn"));
      JSON_CHK(jsonUint64(jfo, trace[NCCL_INSP_EVT_TRK_PROXY_STEP_RECV_WAIT].sn));
      JSON_CHK(jsonKey(jfo, "recv_flush_wait_sn"));
      JSON_CHK(jsonUint64(jfo, trace[NCCL_INSP_EVT_TRK_PROXY_STEP_RECV_FLUSH_WAIT].sn));
      JSON_CHK(jsonKey(jfo, "recv_gpu_wait_sn"));
      JSON_CHK(jsonUint64(jfo, trace[NCCL_INSP_EVT_TRK_PROXY_STEP_RECV_GPU_WAIT].sn));
    }
    JSON_CHK(jsonKey(jfo, "proxy_step_stop_sn"));
    JSON_CHK(jsonUint64(jfo, trace[NCCL_INSP_EVT_TRK_PROXY_STEP_STOP].sn));
    JSON_CHK(jsonFinishObject(jfo));
  }

  JSON_CHK(jsonKey(jfo, "event_trace_ts"));
  JSON_CHK(jsonStartObject(jfo));
  {
    INS_CHK(inspectorProxyTraceTimestamp(jfo, "proxy_step_start_ts",
                                          trace[NCCL_INSP_EVT_TRK_PROXY_STEP_START]));
    if (isSend) {
      INS_CHK(inspectorProxyTraceTimestamp(jfo, "send_gpu_wait_ts",
                                            trace[NCCL_INSP_EVT_TRK_PROXY_STEP_SEND_GPU_WAIT]));
      INS_CHK(inspectorProxyTraceTimestamp(jfo, "send_peer_wait_ts",
                                            trace[NCCL_INSP_EVT_TRK_PROXY_STEP_SEND_PEER_WAIT]));
      INS_CHK(inspectorProxyTraceTimestamp(jfo, "send_wait_ts",
                                            trace[NCCL_INSP_EVT_TRK_PROXY_STEP_SEND_WAIT]));
    } else {
      INS_CHK(inspectorProxyTraceTimestamp(jfo, "recv_wait_ts",
                                            trace[NCCL_INSP_EVT_TRK_PROXY_STEP_RECV_WAIT]));
      INS_CHK(inspectorProxyTraceTimestamp(jfo, "recv_flush_wait_ts",
                                            trace[NCCL_INSP_EVT_TRK_PROXY_STEP_RECV_FLUSH_WAIT]));
      INS_CHK(inspectorProxyTraceTimestamp(jfo, "recv_gpu_wait_ts",
                                            trace[NCCL_INSP_EVT_TRK_PROXY_STEP_RECV_GPU_WAIT]));
    }
    INS_CHK(inspectorProxyTraceTimestamp(jfo, "proxy_step_stop_ts",
                                          trace[NCCL_INSP_EVT_TRK_PROXY_STEP_STOP]));
  }
  JSON_CHK(jsonFinishObject(jfo));
  return inspectorSuccess;
}

static inspectorResult_t inspectorCompletedProxyStepFields(jsonFileOutput* jfo,
                                                            const inspectorProxyStepSnapshot* step,
                                                            bool isSend) {
  JSON_CHK(jsonKey(jfo, "proxy_step_sn")); JSON_CHK(jsonUint64(jfo, step->proxyStepSn));
  JSON_CHK(jsonKey(jfo, "step")); JSON_CHK(jsonInt(jfo, step->step));
  JSON_CHK(jsonKey(jfo, "trans_size_bytes")); JSON_CHK(jsonSize_t(jfo, step->transSizeBytes));
  return inspectorCompletedProxyStepTrace(jfo, step, isSend);
}

static inspectorResult_t inspectorCompletedProxySteps(jsonFileOutput* jfo,
                                                       const inspectorCompletedProxyRecord* proxy,
                                                       const std::vector<const inspectorProxyStepSnapshot*>* steps) {
  JSON_CHK(jsonKey(jfo, "steps"));
  JSON_CHK(jsonStartList(jfo));
  if (steps != nullptr) {
    for (const auto* step : *steps) {
      JSON_CHK(jsonStartObject(jfo));
      INS_CHK(inspectorCompletedProxyStepFields(jfo, step, proxy->metadata.isSend));
      JSON_CHK(jsonFinishObject(jfo));
    }
  }
  JSON_CHK(jsonFinishList(jfo));
  return inspectorSuccess;
}

static inline inspectorResult_t inspectorCompletedProxy(jsonFileOutput* jfo,
                                                         const struct inspectorCompletedProxyRecord* proxy,
                                                         const std::vector<const inspectorProxyStepSnapshot*>* steps = nullptr) {
  if (proxy->recordType != NCCL_INSP_PROXY_RECORD_OP
      && proxy->recordType != NCCL_INSP_PROXY_RECORD_STEP
      && proxy->recordType != NCCL_INSP_PROXY_RECORD_OP_START
      && proxy->recordType != NCCL_INSP_PROXY_RECORD_OP_SEGMENT) {
    return inspectorJsonError;
  }

  JSON_CHK(jsonStartObject(jfo));
  {
    JSON_CHK(jsonKey(jfo, "record_type"));
    const char* recordType = proxy->recordType == NCCL_INSP_PROXY_RECORD_OP ? "proxy_op"
                            : proxy->recordType == NCCL_INSP_PROXY_RECORD_STEP ? "proxy_step"
                            : proxy->recordType == NCCL_INSP_PROXY_RECORD_OP_START ? "proxy_op_start"
                            : "proxy_op_segment";
    JSON_CHK(jsonStr(jfo, recordType));
    JSON_CHK(jsonKey(jfo, "record_sn"));
    JSON_CHK(jsonUint64(jfo, proxy->recordSn));

    JSON_CHK(jsonKey(jfo, "parent_type"));
    JSON_CHK(jsonStr(jfo, inspectorProxyParentTypeToString(
                           proxy->metadata.parentType)));
    JSON_CHK(jsonKey(jfo, "parent_sn"));
    JSON_CHK(jsonUint64(jfo, proxy->metadata.parentSn));
    JSON_CHK(jsonKey(jfo, "proxy_op_sn"));
    JSON_CHK(jsonUint64(jfo, proxy->metadata.proxyOpSn));
    // Step records inherit rank from the enclosing dump marker's header.
    if (proxy->recordType == NCCL_INSP_PROXY_RECORD_OP) {
      JSON_CHK(jsonKey(jfo, "rank"));
      JSON_CHK(jsonInt(jfo, proxy->metadata.rank));
    }
    JSON_CHK(jsonKey(jfo, "channel_id"));
    JSON_CHK(jsonInt(jfo, proxy->metadata.channelId));
    JSON_CHK(jsonKey(jfo, "peer"));
    JSON_CHK(jsonInt(jfo, proxy->metadata.peer));
    JSON_CHK(jsonKey(jfo, "direction"));
    JSON_CHK(jsonStr(jfo, proxy->metadata.isSend ? "send" : "recv"));
    JSON_CHK(jsonKey(jfo, "timing_source"));
    JSON_CHK(jsonStr(jfo, "proxy_cpu"));
    if (proxy->nested) {
      JSON_CHK(jsonKey(jfo, "is_final"));
      // jsonBool emits a string; use an actual JSON boolean for this marker.
      JSON_CHK(jsonBoolean(jfo, proxy->recordType == NCCL_INSP_PROXY_RECORD_OP));
    }

    if (proxy->recordType == NCCL_INSP_PROXY_RECORD_OP_START) {
      JSON_CHK(jsonKey(jfo, "n_steps")); JSON_CHK(jsonInt(jfo, proxy->proxyOp.nSteps));
      JSON_CHK(jsonKey(jfo, "chunk_size_bytes")); JSON_CHK(jsonInt(jfo, proxy->proxyOp.chunkSize));
      INS_CHK(inspectorCompletedProxyOpTrace(jfo, proxy));
    } else if (proxy->recordType == NCCL_INSP_PROXY_RECORD_OP_SEGMENT) {
      INS_CHK(inspectorCompletedProxySteps(jfo, proxy, steps));
    } else if (proxy->recordType == NCCL_INSP_PROXY_RECORD_OP) {
      JSON_CHK(jsonKey(jfo, "n_steps"));
      JSON_CHK(jsonInt(jfo, proxy->proxyOp.nSteps));
      JSON_CHK(jsonKey(jfo, "chunk_size_bytes"));
      JSON_CHK(jsonInt(jfo, proxy->proxyOp.chunkSize));
      JSON_CHK(jsonKey(jfo, "n_steps_started"));
      JSON_CHK(jsonUint32(jfo, proxy->proxyOp.nStepsStarted));
      JSON_CHK(jsonKey(jfo, "n_steps_completed"));
      JSON_CHK(jsonUint32(jfo, proxy->proxyOp.nStepsCompleted));
      JSON_CHK(jsonKey(jfo, "n_steps_dropped"));
      JSON_CHK(jsonUint32(jfo, proxy->proxyOp.nStepsDropped));
      JSON_CHK(jsonKey(jfo, "trans_size_bytes"));
      JSON_CHK(jsonSize_t(jfo, proxy->proxyOp.transSizeBytes));
      INS_CHK(inspectorCompletedProxyOpTrace(jfo, proxy));
      if (proxy->nested && enableNcclInspectorProxyStepDump) {
        INS_CHK(inspectorCompletedProxySteps(jfo, proxy, steps));
      }
    } else {
      INS_CHK(inspectorCompletedProxyStepFields(jfo, &proxy->proxyStep, proxy->metadata.isSend));
    }
  }
  JSON_CHK(jsonFinishObject(jfo));
  return inspectorSuccess;
}

// Views into the drained batch only: no active-event pointers and no state
// retained across dumps. Starts stay independent; other nested records merge
// by the communicator-local Op sequence. Only a final record supplies totals.
struct inspectorProxyDumpGroup {
  const inspectorCompletedProxyRecord* parent = nullptr;
  std::vector<const inspectorProxyStepSnapshot*> steps;
};

static inspectorResult_t inspectorGroupProxyRecords(
    const std::vector<inspectorCompletedProxyRecord>& records,
    std::vector<inspectorProxyDumpGroup>& groups) {
  groups.clear();
  std::unordered_map<uint64_t, size_t> byOp;
  for (const auto& record : records) {
    if (!record.nested || record.recordType == NCCL_INSP_PROXY_RECORD_OP_START) {
      groups.emplace_back();
      groups.back().parent = &record;
      continue;
    }
    auto found = byOp.emplace(record.metadata.proxyOpSn, groups.size());
    if (found.second) groups.emplace_back();
    auto& group = groups[found.first->second];
    if (record.recordType == NCCL_INSP_PROXY_RECORD_STEP) {
      group.steps.push_back(&record.proxyStep);
    } else if (group.parent == nullptr || record.recordType == NCCL_INSP_PROXY_RECORD_OP) {
      group.parent = &record;
    }
  }
  // Atomic Steps-first/parent-last enqueue plus FIFO eviction guarantees this.
  for (const auto& group : groups) if (group.parent == nullptr) return inspectorJsonError;
  // Merging consumes several ring sequence numbers; output uses the selected
  // parent's sequence, in increasing order. Child order remains callback order.
  std::sort(groups.begin(), groups.end(), [](const inspectorProxyDumpGroup& a,
                                            const inspectorProxyDumpGroup& b) {
    return a.parent->recordSn < b.parent->recordSn;
  });
  return inspectorSuccess;
}


/*
 * Description:
 *
 *   Dumps the state of a communicator to the JSON output if needed.
 *
 * Thread Safety:
 *   Not thread-safe (should be called with proper locking).
 *
 * Input:
 *   jsonFileOutput* jfo - JSON output handle.
 *   inspectorCommInfo* commInfo - communicator info.
 *   bool* needs_writing - set to true if output was written.
 *
 * Output:
 *   State is dumped to JSON output if needed.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 *
 */
// Per-dump statistics for one communicator: records written this dump and
// cumulative / since-last-dump ring drop counts for each op type.
struct inspectorCommDumpStats {
  uint64_t collRecords = 0;
  uint64_t collDroppedTotal = 0;
  uint64_t collDroppedSinceLastDump = 0;
  uint64_t p2pRecords = 0;
  uint64_t p2pDroppedTotal = 0;
  uint64_t p2pDroppedSinceLastDump = 0;
  uint64_t proxyRecords = 0;
  uint64_t proxyDroppedTotal = 0;
  uint64_t proxyDroppedSinceLastDump = 0;
  uint64_t proxyOpsDroppedTotal = 0;
  uint64_t proxyOpsDroppedSinceLastDump = 0;
  uint64_t proxyStepsOverwrittenTotal = 0;
  uint64_t proxyStepsOverwrittenSinceLastDump = 0;
  uint64_t proxyPxnSkippedProcessTotal = 0;
  bool proxyPxnSkippedChanged = false;
};

/*
 * Description:
 *   Snapshots a ring's drop counters and advances its reported watermark so the
 *   next dump reports only newly dropped entries.
 *
 * Thread Safety:
 *   Not thread-safe (caller must hold commInfo->guard).
 */
static inline void inspectorCommInfoUpdateDropStats(struct inspectorCompletedRing* ring,
                                                    uint64_t* total,
                                                    uint64_t* sinceLastDump) {
  *total = ring->dropped;
  *sinceLastDump = ring->dropped - ring->droppedReported;
  ring->droppedReported = ring->dropped;
}

// Drains the collective ring under the comm guard (one lock hold for this ring)
// and snapshots its drop stats. Records are appended to drained.
static inspectorResult_t inspectorCommInfoDrainColl(inspectorCommInfo* commInfo,
                                                    std::vector<inspectorCompletedOpInfo>& drained,
                                                    inspectorCommDumpStats* stats) {
  inspectorLockWr(&commInfo->guard);
  if (commInfo->dump_coll) {
    // Make sure we won't allocate while draining (steady-state: no-op).
    if (commInfo->completedCollRing.size > 0
        && drained.capacity() < commInfo->completedCollRing.size) {
      drained.reserve(commInfo->completedCollRing.size);
    }
    INS_CHK(inspectorRingDrain<inspectorCompletedOpInfo>(&commInfo->completedCollRing,
                                                         drained));
    commInfo->dump_coll = inspectorRingNonEmpty(&commInfo->completedCollRing);
  }
  inspectorCommInfoUpdateDropStats(&commInfo->completedCollRing,
                                   &stats->collDroppedTotal,
                                   &stats->collDroppedSinceLastDump);
  inspectorUnlockRWLock(&commInfo->guard);
  stats->collRecords = drained.size();
  return inspectorSuccess;
}

// Drains the P2P ring under the comm guard (one lock hold for this ring)
// and snapshots its drop stats.
static inspectorResult_t inspectorCommInfoDrainP2p(inspectorCommInfo* commInfo,
                                                   std::vector<inspectorCompletedOpInfo>& drained,
                                                   inspectorCommDumpStats* stats) {
  inspectorLockWr(&commInfo->guard);
  if (commInfo->dump_p2p) {
    if (commInfo->completedP2pRing.size > 0
        && drained.capacity() < commInfo->completedP2pRing.size) {
      drained.reserve(commInfo->completedP2pRing.size);
    }
    INS_CHK(inspectorRingDrain<inspectorCompletedOpInfo>(&commInfo->completedP2pRing,
                                                         drained));
    commInfo->dump_p2p = inspectorRingNonEmpty(&commInfo->completedP2pRing);
  }
  inspectorCommInfoUpdateDropStats(&commInfo->completedP2pRing,
                                   &stats->p2pDroppedTotal,
                                   &stats->p2pDroppedSinceLastDump);
  inspectorUnlockRWLock(&commInfo->guard);
  stats->p2pRecords = drained.size();
  return inspectorSuccess;
}

/*
 * Description:
 *   Writes a single per-dump stats record for a communicator: a marker at the
 *   start of the comm's dump cycle carrying records-written and drop counts.
 *   Emitted once per dump per comm, so consumers must not assume every record
 *   carries coll_perf/p2p_perf — the "dump_stats" key identifies this record.
 *
 * Thread Safety:
 *   Not thread-safe (caller must hold the JSON output lock).
 */
static inspectorResult_t inspectorCommInfoDumpStats(jsonFileOutput* jfo,
                                                    inspectorCommInfo* commInfo,
                                                    const inspectorCommDumpStats* stats) {
  JSON_CHK(jsonStartObject(jfo));
  {
    JSON_CHK(jsonKey(jfo, "header"));
    INS_CHK(inspectorCommInfoHeader(jfo, commInfo));

    JSON_CHK(jsonKey(jfo, "metadata"));
    INS_CHK(inspectorCommInfoMetaHeader(jfo));

    JSON_CHK(jsonKey(jfo, "dump_stats"));
    JSON_CHK(jsonStartObject(jfo));
    {
      JSON_CHK(jsonKey(jfo, "coll_records")); JSON_CHK(jsonUint64(jfo, stats->collRecords));
      JSON_CHK(jsonKey(jfo, "coll_dropped_total")); JSON_CHK(jsonUint64(jfo, stats->collDroppedTotal));
      JSON_CHK(jsonKey(jfo, "coll_dropped_since_last_dump"));
      JSON_CHK(jsonUint64(jfo, stats->collDroppedSinceLastDump));
      JSON_CHK(jsonKey(jfo, "p2p_records")); JSON_CHK(jsonUint64(jfo, stats->p2pRecords));
      JSON_CHK(jsonKey(jfo, "p2p_dropped_total")); JSON_CHK(jsonUint64(jfo, stats->p2pDroppedTotal));
      JSON_CHK(jsonKey(jfo, "p2p_dropped_since_last_dump"));
      JSON_CHK(jsonUint64(jfo, stats->p2pDroppedSinceLastDump));
      if (enableNcclInspectorProxy) {
        JSON_CHK(jsonKey(jfo, "proxy_records")); JSON_CHK(jsonUint64(jfo, stats->proxyRecords));
        JSON_CHK(jsonKey(jfo, "proxy_dropped_total")); JSON_CHK(jsonUint64(jfo, stats->proxyDroppedTotal));
        JSON_CHK(jsonKey(jfo, "proxy_dropped_since_last_dump"));
        JSON_CHK(jsonUint64(jfo, stats->proxyDroppedSinceLastDump));
        JSON_CHK(jsonKey(jfo, "proxy_ops_dropped_total"));
        JSON_CHK(jsonUint64(jfo, stats->proxyOpsDroppedTotal));
        JSON_CHK(jsonKey(jfo, "proxy_ops_dropped_since_last_dump"));
        JSON_CHK(jsonUint64(jfo, stats->proxyOpsDroppedSinceLastDump));
        JSON_CHK(jsonKey(jfo, "proxy_steps_overwritten_total"));
        JSON_CHK(jsonUint64(jfo, stats->proxyStepsOverwrittenTotal));
        JSON_CHK(jsonKey(jfo, "proxy_steps_overwritten_since_last_dump"));
        JSON_CHK(jsonUint64(jfo, stats->proxyStepsOverwrittenSinceLastDump));
        JSON_CHK(jsonKey(jfo, "proxy_output_mode"));
        JSON_CHK(jsonStr(jfo, enableNcclInspectorProxyNested ? "nested" : "flat"));
        JSON_CHK(jsonKey(jfo, "proxy_segment_step_capacity"));
        JSON_CHK(jsonInt(jfo, NCCL_INSP_PROXY_SEGMENT_STEPS));
        JSON_CHK(jsonKey(jfo, "proxy_pxn_skipped_process_total"));
        JSON_CHK(jsonUint64(jfo, stats->proxyPxnSkippedProcessTotal));
        JSON_CHK(jsonKey(jfo, "proxy_steps_enabled"));
        JSON_CHK(jsonInt(jfo, enableNcclInspectorProxyStepDump ? 1 : 0));
      }
    }
    JSON_CHK(jsonFinishObject(jfo));
  }
  JSON_CHK(jsonFinishObject(jfo));
  JSON_CHK(jsonNewline(jfo));
  return inspectorSuccess;
}

static inspectorResult_t inspectorCommInfoDrainProxy(inspectorCommInfo* commInfo,
                                                     std::vector<inspectorCompletedProxyRecord>& drained,
                                                     inspectorCommDumpStats* stats) {
  INS_CHK(inspectorLockWr(&commInfo->guard));
  inspectorResult_t res = inspectorSuccess;
  if (commInfo->dump_proxy) {
    if (commInfo->completedProxyRing.size > 0
        && drained.capacity() < commInfo->completedProxyRing.size) {
      drained.reserve(commInfo->completedProxyRing.size);
    }
    INS_CHK_GOTO(inspectorRingDrain<inspectorCompletedProxyRecord>(&commInfo->completedProxyRing, drained), res, exit);
    commInfo->dump_proxy = inspectorRingNonEmpty(&commInfo->completedProxyRing);
  }
  // Snapshot even an empty ring: pool failures and PXN skips may be the only
  // activity in this window. Their counters have different scopes from drops.
  inspectorCommInfoUpdateDropStats(&commInfo->completedProxyRing,
                                   &stats->proxyDroppedTotal,
                                   &stats->proxyDroppedSinceLastDump);
  stats->proxyRecords = drained.size();
  stats->proxyStepsOverwrittenTotal = commInfo->completedProxyRing.droppedItems;
  stats->proxyStepsOverwrittenSinceLastDump = stats->proxyStepsOverwrittenTotal
                                            - commInfo->proxyStepsOverwrittenReported;
  commInfo->proxyStepsOverwrittenReported = stats->proxyStepsOverwrittenTotal;
  stats->proxyOpsDroppedTotal = commInfo->proxyOpsDropped;
  stats->proxyOpsDroppedSinceLastDump = commInfo->proxyOpsDropped - commInfo->proxyOpsDroppedReported;
  commInfo->proxyOpsDroppedReported = commInfo->proxyOpsDropped;
  stats->proxyPxnSkippedProcessTotal = __atomic_load_n(&ncclInspectorProxyPxnSkipped, __ATOMIC_RELAXED);
  stats->proxyPxnSkippedChanged = stats->proxyPxnSkippedProcessTotal != commInfo->proxyPxnSkippedReported;
  commInfo->proxyPxnSkippedReported = stats->proxyPxnSkippedProcessTotal;
exit:
  inspectorUnlockRWLock(&commInfo->guard);
  return res;
}

static inspectorResult_t inspectorCommInfoDump(jsonFileOutput* jfo,
                                               inspectorCommInfo* commInfo,
                                               bool* needs_writing) {
  *needs_writing = false;
  if (commInfo == nullptr) {
    return inspectorSuccess;
  }

  thread_local std::vector<inspectorCompletedOpInfo> drainedColl;
  thread_local std::vector<inspectorCompletedOpInfo> drainedP2p;
  thread_local std::vector<inspectorCompletedProxyRecord> drainedProxy;
  drainedColl.clear();
  drainedP2p.clear();
  drainedProxy.clear();

  inspectorCommDumpStats stats;
  // One guard hold per ring, matching the original per-ring locking.
  INS_CHK(inspectorCommInfoDrainColl(commInfo, drainedColl, &stats));
  INS_CHK(inspectorCommInfoDrainP2p(commInfo, drainedP2p, &stats));
  if (enableNcclInspectorProxy) {
    INS_CHK(inspectorCommInfoDrainProxy(commInfo, drainedProxy, &stats));
  }
  std::vector<inspectorProxyDumpGroup> proxyGroups;
  if (enableNcclInspectorProxyNested) {
    INS_CHK(inspectorGroupProxyRecords(drainedProxy, proxyGroups));
    // dump_stats counts JSON payloads, not the small slots consumed in the ring.
    stats.proxyRecords = proxyGroups.size();
  }

  // Emit only when this comm produced records or has newly dropped entries to
  // report, so idle communicators don't generate empty stats records.
  bool haveDrops = stats.collDroppedSinceLastDump != 0 || stats.p2pDroppedSinceLastDump != 0
                   || stats.proxyDroppedSinceLastDump != 0 || stats.proxyOpsDroppedSinceLastDump != 0
                   || stats.proxyPxnSkippedChanged;
  if (drainedColl.empty() && drainedP2p.empty() && drainedProxy.empty() && !haveDrops) {
    return inspectorSuccess;
  }

  *needs_writing = true;
  inspectorResult_t res = inspectorSuccess;
  // Hold the output lock for the whole batch. All following records inherit
  // the header/metadata of this dump_stats marker until the next marker.
  JSON_CHK(jsonLockOutput(jfo));
  INS_CHK_GOTO(inspectorCommInfoDumpStats(jfo, commInfo, &stats), res, exit);
  for (size_t i = 0; i < drainedColl.size(); i++) {
    JSON_CHK_GOTO(jsonStartObject(jfo), res, exit);
    JSON_CHK_GOTO(jsonKey(jfo, "coll_perf"), res, exit);
    INS_CHK_GOTO(inspectorCompletedColl(jfo, &drainedColl[i]), res, exit);
    JSON_CHK_GOTO(jsonFinishObject(jfo), res, exit);
    JSON_CHK_GOTO(jsonNewline(jfo), res, exit);
  }
  for (size_t i = 0; i < drainedP2p.size(); i++) {
    JSON_CHK_GOTO(jsonStartObject(jfo), res, exit);
    JSON_CHK_GOTO(jsonKey(jfo, "p2p_perf"), res, exit);
    INS_CHK_GOTO(inspectorCompletedP2p(jfo, &drainedP2p[i]), res, exit);
    JSON_CHK_GOTO(jsonFinishObject(jfo), res, exit);
    JSON_CHK_GOTO(jsonNewline(jfo), res, exit);
  }
  for (size_t i = 0; i < stats.proxyRecords; i++) {
    const auto* parent = enableNcclInspectorProxyNested ? proxyGroups[i].parent : &drainedProxy[i];
    const auto* steps = enableNcclInspectorProxyNested ? &proxyGroups[i].steps : nullptr;
    JSON_CHK_GOTO(jsonStartObject(jfo), res, exit);
    JSON_CHK_GOTO(jsonKey(jfo, "proxy_trace"), res, exit);
    INS_CHK_GOTO(inspectorCompletedProxy(jfo, parent, steps), res, exit);
    JSON_CHK_GOTO(jsonFinishObject(jfo), res, exit);
    JSON_CHK_GOTO(jsonNewline(jfo), res, exit);
  }
exit:
  JSON_CHK(jsonUnlockOutput(jfo));
  return res;
}


/*
 * Description:
 *
 *   Dumps the state of all communicators in a commList to the JSON
 *   output.
 *
 * Thread Safety:
 *   Thread-safe - assumes no locks are taken and acquires all
 *   necessary locks to iterate through all communicator objects and
 *   dump their state.
 *
 * Input:
 *   jsonFileOutput* jfo - JSON output handle (must not be NULL).
 *   struct inspectorCommInfoList* commList - list of communicators
 *   (must not be NULL).
 *
 * Output:
 *   State of all communicators is dumped to JSON output.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 *
 */
inspectorResult_t inspectorCommInfoListDump(jsonFileOutput* jfo,
                                            struct inspectorCommInfoList* commList) {
  bool flush = false;
  INS_CHK(inspectorLockRd(&commList->guard));
  inspectorResult_t res = inspectorSuccess;
  if (commList->ncomms > 0) {
    for (struct inspectorCommInfo* itr = commList->comms;
         itr != nullptr;
         itr = itr->next) {
      bool needs_writing;
      INS_CHK_GOTO(inspectorCommInfoDump(jfo, itr, &needs_writing),
                   res, exit);
      if (needs_writing) {
        flush = true;
      }
    }
    if (flush) {
      JSON_CHK_GOTO(jsonLockOutput(jfo), res, exit);
      JSON_CHK_GOTO(jsonFlushOutput(jfo), res, exit);
      JSON_CHK_GOTO(jsonUnlockOutput(jfo), res, exit);
    }
  }
exit:
  INS_CHK(inspectorUnlockRWLock(&commList->guard));
  return res;
}
