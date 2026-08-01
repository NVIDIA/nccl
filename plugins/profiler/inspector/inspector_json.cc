#include "inspector_json.h"
#include "inspector_ring.h"

#include <unistd.h>
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
    JSON_CHK(jsonKey(jfo, "inspector_output_format_version")); JSON_CHK(jsonStr(jfo, "v4.1"));
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
    inspectorCommInfoHeader(jfo, commInfo);

    JSON_CHK(jsonKey(jfo, "metadata"));
    inspectorCommInfoMetaHeader(jfo);

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
    }
    JSON_CHK(jsonFinishObject(jfo));
  }
  JSON_CHK(jsonFinishObject(jfo));
  JSON_CHK(jsonNewline(jfo));
  return inspectorSuccess;
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
  drainedColl.clear();
  drainedP2p.clear();

  inspectorCommDumpStats stats;
  // One guard hold per ring, matching the original per-ring locking.
  INS_CHK(inspectorCommInfoDrainColl(commInfo, drainedColl, &stats));
  INS_CHK(inspectorCommInfoDrainP2p(commInfo, drainedP2p, &stats));

  // Emit only when this comm produced records or has newly dropped entries to
  // report, so idle communicators don't generate empty stats records.
  bool haveDrops = stats.collDroppedSinceLastDump != 0 || stats.p2pDroppedSinceLastDump != 0;
  if (drainedColl.empty() && drainedP2p.empty() && !haveDrops) {
    return inspectorSuccess;
  }

  *needs_writing = true;
  JSON_CHK(jsonLockOutput(jfo));
  INS_CHK(inspectorCommInfoDumpStats(jfo, commInfo, &stats));
  for (size_t i = 0; i < drainedColl.size(); i++) {
    JSON_CHK(jsonStartObject(jfo));
    {
      JSON_CHK(jsonKey(jfo, "header"));
      inspectorCommInfoHeader(jfo, commInfo);

      JSON_CHK(jsonKey(jfo, "metadata"));
      inspectorCommInfoMetaHeader(jfo);

      JSON_CHK(jsonKey(jfo, "coll_perf"));
      INS_CHK(inspectorCompletedColl(jfo, &drainedColl[i]));
    }
    JSON_CHK(jsonFinishObject(jfo));
    JSON_CHK(jsonNewline(jfo));
  }
  for (size_t i = 0; i < drainedP2p.size(); i++) {
    JSON_CHK(jsonStartObject(jfo));
    {
      JSON_CHK(jsonKey(jfo, "header"));
      inspectorCommInfoHeader(jfo, commInfo);

      JSON_CHK(jsonKey(jfo, "metadata"));
      inspectorCommInfoMetaHeader(jfo);

      JSON_CHK(jsonKey(jfo, "p2p_perf"));
      INS_CHK(inspectorCompletedP2p(jfo, &drainedP2p[i]));
    }
    JSON_CHK(jsonFinishObject(jfo));
    JSON_CHK(jsonNewline(jfo));
  }
  JSON_CHK(jsonUnlockOutput(jfo));
  return inspectorSuccess;
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
