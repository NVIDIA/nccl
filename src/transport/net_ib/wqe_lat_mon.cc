/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "wqe_lat_mon.h"
#include "common.h"

#include <math.h>
#include <mutex>
#include <stdint.h>
#include <string.h>
#include <time.h>

NCCL_PARAM(IbWqeLatencyThresholdNs, "IB_WQE_LATENCY_THRESHOLD_NS", 0);
NCCL_PARAM(IbWqeLatencyReport, "IB_WQE_LATENCY_REPORT", 1);

bool ncclIbWqeLatEnabled = false;
uint64_t ncclIbWqeLatThresholdNs = 0;
bool ncclIbWqeLatReportEnabled = false;

static const uint64_t kReportIntervalNs = 1000000000ULL;
static const double kPctlTargets[NCCL_IB_WQE_LAT_NUM_PCTL] = {0.50, 0.90, 0.99, 0.999};

static void ensureInitialized(void) {
  static std::once_flag once;
  std::call_once(once, []() {
    int64_t thr = ncclParamIbWqeLatencyThresholdNs();
    if (thr <= 0) return;
    ncclIbWqeLatThresholdNs = (uint64_t)thr;
    ncclIbWqeLatReportEnabled = ncclParamIbWqeLatencyReport() != 0;
    ncclIbWqeLatEnabled = true;
  });
}

uint64_t ncclIbWqeLatMonNowNs(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

// Jain & Chlamtac P^2: 5 markers, no stored samples, O(1) per observation.

static void p2Init(struct ncclIbP2Quantile* e, double p) {
  e->p = p;
  for (int i = 0; i < 5; i++) {
    e->q[i] = 0.0;
    e->n[i] = 0;
  }
}

static void p2SortAsc(double* a, int n) {
  for (int i = 1; i < n; i++) {
    double key = a[i];
    int j = i - 1;
    while (j >= 0 && a[j] > key) {
      a[j + 1] = a[j];
      j--;
    }
    a[j + 1] = key;
  }
}

static void p2Update(struct ncclIbP2Quantile* e, double x, uint64_t count) {
  if (count <= 5) {
    e->q[count - 1] = x;
    if (count == 5) {
      p2SortAsc(e->q, 5);
      for (int i = 0; i < 5; i++) e->n[i] = (uint64_t)(i + 1);
    }
    return;
  }

  const double p = e->p;
  const double f[5] = {0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0};

  int k;
  if (x < e->q[0]) {
    e->q[0] = x;
    k = 0;
  } else if (x >= e->q[4]) {
    e->q[4] = x;
    k = 3;
  } else {
    k = 0;
    for (int i = 1; i < 5; i++) {
      if (x < e->q[i]) {
        k = i - 1;
        break;
      }
    }
  }

  for (int i = k + 1; i < 5; i++) e->n[i] += 1;

  for (int i = 1; i <= 3; i++) {
    double ni = (double)e->n[i];
    double nim1 = (double)e->n[i - 1];
    double nip1 = (double)e->n[i + 1];
    double npi = 1.0 + (double)(count - 1) * f[i];
    double d = npi - ni;
    if ((d >= 1.0 && (nip1 - ni) > 1.0) || (d <= -1.0 && (nim1 - ni) < -1.0)) {
      int s = (d >= 0.0) ? 1 : -1;
      double qim1 = e->q[i - 1], qi = e->q[i], qip1 = e->q[i + 1];
      double qNew = qi + ((double)s / (nip1 - nim1)) * ((ni - nim1 + (double)s) * (qip1 - qi) / (nip1 - ni) +
                                                        (nip1 - ni - (double)s) * (qi - qim1) / (ni - nim1));
      if (qim1 < qNew && qNew < qip1) {
        e->q[i] = qNew;
      } else if (s == 1) {
        e->q[i] = qi + (qip1 - qi) / (nip1 - ni);
      } else {
        e->q[i] = qi - (qi - qim1) / (ni - nim1);
      }
      e->n[i] = (uint64_t)((int64_t)e->n[i] + s);
    }
  }
}

static uint64_t p2Estimate(const struct ncclIbP2Quantile* e, uint64_t count) {
  if (count == 0) return 0;
  double v;
  if (count >= 5) {
    v = e->q[2];
  } else {
    double tmp[5];
    int n = (int)count;
    for (int i = 0; i < n; i++) tmp[i] = e->q[i];
    p2SortAsc(tmp, n);
    int idx = (int)ceil(e->p * (double)n) - 1;
    if (idx < 0) idx = 0;
    if (idx >= n) idx = n - 1;
    v = tmp[idx];
  }
  return v < 0.0 ? 0 : (uint64_t)(v + 0.5);
}

void ncclIbWqeLatMonInit(struct ncclIbWqeLatMon* m) {
  ensureInitialized();
  memset(m, 0, sizeof(*m));
  for (int i = 0; i < NCCL_IB_WQE_LAT_NUM_PCTL; i++) p2Init(&m->p2[i], kPctlTargets[i]);
}

void ncclIbWqeLatMonStampSend(struct ncclIbQp* qp, struct ibv_send_wr* head) {
  struct ncclIbWqeLatMon* m = &qp->latMon;
  uint64_t now = 0;
  bool nowValid = false;
  for (struct ibv_send_wr* w = head; w != NULL; w = w->next) {
    if ((w->send_flags & IBV_SEND_SIGNALED) == 0) continue;
    if (!nowValid) {
      now = ncclIbWqeLatMonNowNs();
      nowValid = true;
    }
    if (!m->tracking) {
      m->tracking = true;
      m->trackedPostNs = now;
      m->pendingBefore = m->inflight;
    }
    m->inflight++;
  }
}

static void welfordUpdate(struct ncclIbWqeLatMon* m, uint64_t deltaNs) {
  m->count++;
  double x = (double)deltaNs;
  double delta = x - m->meanNs;
  m->meanNs += delta / (double)m->count;
  m->m2Ns += delta * (x - m->meanNs);
  if (deltaNs > m->maxNs) m->maxNs = deltaNs;
}

static void snapshotStats(const struct ncclIbWqeLatMon* m, struct ncclIbWqeLatStats* out) {
  if (!out) return;
  out->count = m->count;
  out->slowCount = m->slowCount;
  out->meanNs = m->meanNs;
  out->maxNs = m->maxNs;
  if (m->count > 1) {
    double var = m->m2Ns / (double)(m->count - 1);
    out->stddevNs = sqrt(var > 0.0 ? var : 0.0);
  } else {
    out->stddevNs = 0.0;
  }
  out->p50Ns = p2Estimate(&m->p2[0], m->count);
  out->p90Ns = p2Estimate(&m->p2[1], m->count);
  out->p99Ns = p2Estimate(&m->p2[2], m->count);
  out->p999Ns = p2Estimate(&m->p2[3], m->count);
}

void ncclIbWqeLatMonSnapshot(const struct ncclIbWqeLatMon* m, struct ncclIbWqeLatStats* out) {
  if (!out) return;
  memset(out, 0, sizeof(*out));
  snapshotStats(m, out);
}

bool ncclIbWqeLatMonOnComplete(struct ncclIbWqeLatMon* m, uint64_t tPollNs, uint64_t* outDeltaNs, uint64_t* outPostNs,
                               struct ncclIbWqeLatStats* outStats) {
  if (outDeltaNs) *outDeltaNs = 0;
  if (outPostNs) *outPostNs = 0;
  if (outStats) memset(outStats, 0, sizeof(*outStats));

  if (m->inflight > 0) m->inflight--;
  if (!m->tracking) return false;
  if (m->pendingBefore > 0) {
    m->pendingBefore--;
    return false;
  }

  uint64_t tPostNs = m->trackedPostNs;
  m->tracking = false;
  m->trackedPostNs = 0;

  uint64_t delta = (tPollNs > tPostNs) ? tPollNs - tPostNs : 0;
  welfordUpdate(m, delta);
  for (int i = 0; i < NCCL_IB_WQE_LAT_NUM_PCTL; i++) p2Update(&m->p2[i], (double)delta, m->count);
  if (outDeltaNs) *outDeltaNs = delta;
  if (outPostNs) *outPostNs = tPostNs;
  snapshotStats(m, outStats);

  if (delta <= ncclIbWqeLatThresholdNs) return false;
  m->slowCount++;
  if (m->lastWarnNs && (tPollNs - m->lastWarnNs) < kReportIntervalNs) return false;
  m->lastWarnNs = tPollNs;
  return true;
}

bool ncclIbWqeLatMonCheckStall(struct ncclIbWqeLatMon* m, uint64_t nowNs, uint64_t* outAgeNs, uint32_t* outInflight) {
  if (outAgeNs) *outAgeNs = 0;
  if (outInflight) *outInflight = 0;
  if (!m->tracking) return false;
  if (nowNs <= m->trackedPostNs) return false;

  uint64_t age = nowNs - m->trackedPostNs;
  if (outAgeNs) *outAgeNs = age;
  if (outInflight) *outInflight = m->inflight;

  if (age <= ncclIbWqeLatThresholdNs) return false;
  if (m->lastStallWarnNs && (nowNs - m->lastStallWarnNs) < kReportIntervalNs) return false;
  m->lastStallWarnNs = nowNs;
  return true;
}

namespace {

struct peerInfo {
  char sock[SOCKET_NAME_MAXLEN + 1];
  char localGid[INET6_ADDRSTRLEN];
  char remoteGid[INET6_ADDRSTRLEN];
  const char* hca;
  uint16_t localLid;
  uint16_t remoteLid;
  uint8_t peerPort;
};

static void getPeerInfo(struct ncclIbNetCommBase* base, int devIndex, struct ncclIbQp* qp, struct peerInfo* info) {
  memset(info, 0, sizeof(*info));
  info->hca = "?";
  union ncclSocketAddress addr;
  if (ncclSocketGetAddr(&base->sock, &addr) == ncclSuccess) {
    ncclSocketToString(&addr, info->sock);
  }
  struct ncclIbNetCommDevBase* devBase = ncclIbGetNetCommDevBase(base, devIndex);
  if (devBase->gidInfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
    ibvGetGidStr(&devBase->gidInfo.localGid, info->localGid, sizeof(info->localGid));
    if (qp->remDevIdx >= 0 && qp->remDevIdx < base->nRemDevs) {
      ibvGetGidStr(&base->remDevs[qp->remDevIdx].remoteGid, info->remoteGid, sizeof(info->remoteGid));
    }
  }
  if (devBase->pd && devBase->pd->context && devBase->pd->context->device) {
    info->hca = devBase->pd->context->device->name;
  }
  info->localLid = (uint16_t)ncclIbDevs[devBase->ibDevN].portAttr.lid;
  info->remoteLid = (uint16_t)qp->rtrAttr.remoteLid;
  if (qp->remDevIdx >= 0 && qp->remDevIdx < base->nRemDevs) {
    info->peerPort = base->remDevs[qp->remDevIdx].ib_port;
  }
}

static void reportSlow(struct ncclIbNetCommBase* base, int devIndex, struct ncclIbQp* qp, uint32_t qpn,
                       uint64_t deltaNs, uint64_t tPostNs, uint64_t tPollNs, const struct ncclIbWqeLatStats* s) {
  struct peerInfo p;
  getPeerInfo(base, devIndex, qp, &p);
  INFO(NCCL_NET,
       "NET/IB: WQE slow: peer=%s | local: qpn=%u dev=%s port=%u lid=%u localGid=%s | "
       "remote: qpn=%u port=%u lid=%u remoteGid=%s | "
       "thr=%luns delta=%luns mean=%.0fns stddev=%.0fns "
       "p50=%luns p90=%luns p99=%luns p99.9=%luns max=%luns count=%lu | "
       "t_post=%lu t_poll=%lu",
       p.sock, qpn, p.hca, qp->rtrAttr.localIbPort, (unsigned)p.localLid, p.localGid, qp->rtrAttr.remoteQpNum,
       (unsigned)p.peerPort, (unsigned)p.remoteLid, p.remoteGid, (unsigned long)ncclIbWqeLatThresholdNs,
       (unsigned long)deltaNs, s->meanNs, s->stddevNs, (unsigned long)s->p50Ns, (unsigned long)s->p90Ns,
       (unsigned long)s->p99Ns, (unsigned long)s->p999Ns, (unsigned long)s->maxNs, (unsigned long)s->count,
       (unsigned long)tPostNs, (unsigned long)tPollNs);
}

static void reportStall(struct ncclIbNetCommBase* base, int devIndex, struct ncclIbQp* qp, uint64_t ageNs,
                        uint32_t inflight) {
  struct peerInfo p;
  getPeerInfo(base, devIndex, qp, &p);
  INFO(NCCL_NET,
       "NET/IB: WQE stall (no CQE): peer=%s | local: qpn=%u dev=%s port=%u lid=%u localGid=%s | "
       "remote: qpn=%u port=%u lid=%u remoteGid=%s | "
       "stall_thr=%luns age=%luns inflight=%u",
       p.sock, qp->qp->qp_num, p.hca, qp->rtrAttr.localIbPort, (unsigned)p.localLid, p.localGid,
       qp->rtrAttr.remoteQpNum, (unsigned)p.peerPort, (unsigned)p.remoteLid, p.remoteGid,
       (unsigned long)ncclIbWqeLatThresholdNs, (unsigned long)ageNs, inflight);
}

}  // namespace

void ncclIbWqeLatHandleCompletion(struct ncclIbNetCommBase* base, int devIndex, struct ibv_wc* wc, uint64_t* tPollNs,
                                  bool* tPollNsValid) {
  const bool isSendOpcode =
    (wc->opcode == IBV_WC_SEND) || (wc->opcode == IBV_WC_RDMA_WRITE) || (wc->opcode == IBV_WC_RDMA_READ);
  if (!isSendOpcode) return;
  struct ncclIbQp* qp = NULL;
  int qpIndex = -1;
  if (ncclIbCommBaseGetQpByQpNum(base, devIndex, wc->qp_num, &qp, &qpIndex) != ncclSuccess) return;
  if (qp == NULL) return;
  if (!*tPollNsValid) {
    *tPollNs = ncclIbWqeLatMonNowNs();
    *tPollNsValid = true;
  }
  uint64_t deltaNs = 0, tPostNs = 0;
  struct ncclIbWqeLatStats stats;
  if (ncclIbWqeLatMonOnComplete(&qp->latMon, *tPollNs, &deltaNs, &tPostNs, &stats)) {
    reportSlow(base, devIndex, qp, wc->qp_num, deltaNs, tPostNs, *tPollNs, &stats);
  }
}

void ncclIbWqeLatScanStalls(struct ncclIbNetCommBase* base, int devIndex) {
  if (base->nqps <= 0 || base->vProps.ndevs <= 0) return;
  int nqpsPerDev = base->nqps / base->vProps.ndevs;
  uint64_t now = ncclIbWqeLatMonNowNs();
  for (int k = 0; k < nqpsPerDev; k++) {
    struct ncclIbQp* qp = &base->qps[base->vProps.ndevs * k + devIndex];
    uint64_t age = 0;
    uint32_t inflight = 0;
    if (ncclIbWqeLatMonCheckStall(&qp->latMon, now, &age, &inflight)) {
      reportStall(base, devIndex, qp, age, inflight);
    }
  }
}

void ncclIbWqeLatReportQpSummary(struct ncclIbNetCommBase* base, int devIndex, struct ncclIbQp* qp) {
  if (!ncclIbWqeLatEnabled || !ncclIbWqeLatReportEnabled) return;
  if (qp == NULL || qp->qp == NULL || qp->latMon.count == 0) return;
  struct ncclIbWqeLatStats s;
  ncclIbWqeLatMonSnapshot(&qp->latMon, &s);
  struct peerInfo p;
  getPeerInfo(base, devIndex, qp, &p);
  INFO(NCCL_NET,
       "NET/IB: WQE latency summary [%s peer=%s dev=%s qpn=%u port=%u]: "
       "n=%lu slow=%lu thr=%luns | post-to-poll ns: mean=%.0f std=%.0f "
       "p50=%lu p90=%lu p99=%lu p99.9=%lu max=%lu",
       base->isSend ? "send" : "recv", p.sock, p.hca, qp->qp->qp_num, qp->rtrAttr.localIbPort, (unsigned long)s.count,
       (unsigned long)s.slowCount, (unsigned long)ncclIbWqeLatThresholdNs, s.meanNs, s.stddevNs, (unsigned long)s.p50Ns,
       (unsigned long)s.p90Ns, (unsigned long)s.p99Ns, (unsigned long)s.p999Ns, (unsigned long)s.maxNs);
}
