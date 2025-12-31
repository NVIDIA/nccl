/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "connect.h"
#include "common.h"
#include "p2p_resiliency.h"

NCCL_PARAM(IbGidIndex, "IB_GID_INDEX", -1);
NCCL_PARAM(IbRoutableFlidIbGidIndex, "IB_ROUTABLE_FLID_GID_INDEX", 1);
NCCL_PARAM(IbRoceVersionNum, "IB_ROCE_VERSION_NUM", 2);
NCCL_PARAM(IbTimeout, "IB_TIMEOUT", 20);
NCCL_PARAM(IbRetryCnt, "IB_RETRY_CNT", 7);
NCCL_PARAM(IbPkey, "IB_PKEY", 0);
NCCL_PARAM(IbUseInline, "IB_USE_INLINE", 0);
NCCL_PARAM(IbSl, "IB_SL", -1);
NCCL_PARAM(IbTc, "IB_TC", -1);
NCCL_PARAM(IbFifoTc, "IB_FIFO_TC", -1);
NCCL_PARAM(IbEceEnable,"IB_ECE_ENABLE",1);

extern int64_t ncclParamIbOooRq();

struct ncclIbDevExtraProps {
  bool oooRq;
};

enum ncclIbCommState {
  ncclIbCommStateStart = 0,
  ncclIbCommStateConnect = 1,
  ncclIbCommStateAccept = 3,
  ncclIbCommStateSend = 4,
  ncclIbCommStateRecv = 5,
  ncclIbCommStateConnecting = 6,
  ncclIbCommStateConnected = 7,
  ncclIbCommStatePendingReady = 8,
  ncclIbCommStateSendDevList = 9,
  ncclIbCommStateRecvDevList = 10,
};

struct ncclIbCommStage {
  enum ncclIbCommState state;
  int offset;
  void* buffer;
  void* comm;
};

struct ncclIbHandle {
  union ncclSocketAddress connectAddr; // Filled by the target
  uint64_t magic; // random number to help debugging
  struct ncclIbCommStage stage; // Used by the other side when connecting
};

NCCL_PARAM(IbQpsPerConn, "IB_QPS_PER_CONNECTION", 1);

ncclResult_t ncclIbInitCommDevBase(int ibDevN, struct ncclIbNetCommDevBase* base, void* cq_context, int cqSize) {
  base->ibDevN = ibDevN;
  ncclIbDev* ibDev = ncclIbDevs + ibDevN;
  {
    std::lock_guard<std::mutex> lock(ibDev->mutex);
    if (0 == ibDev->pdRefs++) {
      NCCLCHECK(wrap_ibv_alloc_pd(&ibDev->pd, ibDev->context));
    }
    base->pd = ibDev->pd;
  }

  NCCLCHECK(wrap_ibv_create_cq(&base->cq, ibDev->context, cqSize, cq_context, NULL, 0));

  return ncclSuccess;
}

ncclResult_t ncclIbDestroyBase(struct ncclIbNetCommDevBase* base) {
  NCCLCHECK(wrap_ibv_destroy_cq(base->cq));

  std::lock_guard<std::mutex> lock(ncclIbDevs[base->ibDevN].mutex);
  if (0 == --ncclIbDevs[base->ibDevN].pdRefs) {
    NCCLCHECK(wrap_ibv_dealloc_pd(ncclIbDevs[base->ibDevN].pd));
  }
  return ncclSuccess;
}

// GID Format
// global:  |              64b  - subnet-prefix                |                 64b - EUI                          |
// raw   :  | 10b fixed | 22b 0 | 16b FLID | 16b subnet-prefix |                 64b - EUI                          |
static uint16_t ncclIbExtractLocalSubnetPrefix(uint64_t subnet_prefix)
{
  return (be64toh(subnet_prefix) & 0xffff);
}

static int ncclIbExtractFlid (union ibv_gid *gid)
{
  return ntohs(*((uint16_t*)((uintptr_t)(gid->raw) + 4)));
}

static sa_family_t envIbAddrFamily(void) {
  sa_family_t family = AF_INET;
  const char* env = ncclGetEnv("NCCL_IB_ADDR_FAMILY");
  if (env == NULL || strlen(env) == 0) {
    return family;
  }

  INFO(NCCL_ENV, "NCCL_IB_ADDR_FAMILY set by environment to %s", env);

  if (strcmp(env, "AF_INET") == 0) {
    family = AF_INET;
  } else if (strcmp(env, "AF_INET6") == 0) {
    family = AF_INET6;
  }

  return family;
}

static void* envIbAddrRange(sa_family_t af, int* mask) {
  *mask = 0;
  static struct in_addr addr;
  static struct in6_addr addr6;
  void *ret = (af == AF_INET) ? (void *)&addr : (void *)&addr6;

  const char* env = ncclGetEnv("NCCL_IB_ADDR_RANGE");
  if (NULL == env || strlen(env) == 0) {
    return NULL;
  }

  INFO(NCCL_ENV, "NCCL_IB_ADDR_RANGE set by environment to %s", env);

  char addrString[128] = { 0 };
  snprintf(addrString, 128, "%s", env);
  char *addrStrPtr = addrString;
  char *maskStrPtr = strstr(addrString, "/");
  if (NULL == maskStrPtr) {
    return NULL;
  }
  *(maskStrPtr++) = '\0';

  if (inet_pton(af, addrStrPtr, ret) == 0) {
    INFO(NCCL_INIT|NCCL_NET, "NET/IB: Ip address '%s' is invalid for family %s, ignoring address", addrStrPtr, (af == AF_INET) ? "AF_INET" : "AF_INET6");
    return NULL;
  }

  *mask = (int)strtol(maskStrPtr, NULL, 10);
  if (af == AF_INET && *mask > 32) {
    INFO(NCCL_INIT|NCCL_NET, "NET/IB: Ip address mask '%d' is invalid for family %s, ignoring mask", *mask, (af == AF_INET) ? "AF_INET" : "AF_INET6");
    *mask = 0;
    ret = NULL;
  } else if (af == AF_INET6 && *mask > 128) {
    INFO(NCCL_INIT|NCCL_NET, "NET/IB: Ip address mask '%d' is invalid for family %s, ignoring mask", *mask, (af == AF_INET) ? "AF_INET" : "AF_INET6");
    *mask = 0;
    ret = NULL;
  }

  return ret;
}

static sa_family_t getGidAddrFamily(union ibv_gid* gid) {
  const struct in6_addr *a = (struct in6_addr *)gid->raw;
  bool isIpV4Mapped = ((a->s6_addr32[0] | a->s6_addr32[1]) | (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL;
  bool isIpV4MappedMulticast = (a->s6_addr32[0] == htonl(0xff0e0000) && ((a->s6_addr32[1] | (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL));
  return (isIpV4Mapped || isIpV4MappedMulticast) ? AF_INET : AF_INET6;
}

static bool matchGidAddrPrefix(sa_family_t af, void* prefix, int prefixlen, union ibv_gid* gid) {
  struct in_addr *base = NULL;
  struct in6_addr *base6 = NULL;
  struct in6_addr *addr6 = NULL;;
  if (af == AF_INET) {
    base = (struct in_addr *)prefix;
  } else {
    base6 = (struct in6_addr *)prefix;
  }
  addr6 = (struct in6_addr *)gid->raw;

#define NETMASK(bits) (htonl(0xffffffff ^ ((1 << (32 - bits)) - 1)))

  int i = 0;
  while (prefixlen > 0 && i < 4) {
    if (af == AF_INET) {
      int mask = NETMASK(prefixlen);
      if ((base->s_addr & mask) ^ (addr6->s6_addr32[3] & mask)) {
        break;
      }
      prefixlen = 0;
      break;
    } else {
      if (prefixlen >= 32) {
        if (base6->s6_addr32[i] ^ addr6->s6_addr32[i]) {
          break;
        }
        prefixlen -= 32;
        ++i;
      } else {
        int mask = NETMASK(prefixlen);
        if ((base6->s6_addr32[i] & mask) ^ (addr6->s6_addr32[i] & mask)) {
          break;
        }
        prefixlen = 0;
      }
    }
  }

  return (prefixlen == 0) ? true : false;
}

static bool configuredGid(union ibv_gid* gid) {
  const struct in6_addr *a = (struct in6_addr *)gid->raw;
  int trailer = (a->s6_addr32[1] | a->s6_addr32[2] | a->s6_addr32[3]);
  if (((a->s6_addr32[0] | trailer) == 0UL) || ((a->s6_addr32[0] == htonl(0xfe800000)) && (trailer == 0UL))) {
    return false;
  }
  return true;
}

static bool linkLocalGid(union ibv_gid* gid) {
  const struct in6_addr *a = (struct in6_addr *)gid->raw;
  if (a->s6_addr32[0] == htonl(0xfe800000) && a->s6_addr32[1] == 0UL) {
    return true;
  }
  return false;
}

static bool validGid(union ibv_gid* gid) {
  return (configuredGid(gid) && !linkLocalGid(gid));
}

static ncclResult_t ncclIbRoceGetVersionNum(const char* deviceName, int portNum, int gidIndex, int* version) {
  char gidRoceVerStr[16] = { 0 };
  char roceTypePath[PATH_MAX] = { 0 };
  snprintf(roceTypePath, sizeof(roceTypePath), "/sys/class/infiniband/%s/ports/%d/gid_attrs/types/%d", deviceName, portNum, gidIndex);

  int fd = open(roceTypePath, O_RDONLY);
  if (fd == -1) {
    WARN("NET/IB: open failed in ncclIbRoceGetVersionNum: %s", strerror(errno));
    return ncclSystemError;
  }
  int ret = read(fd, gidRoceVerStr, 15);
  close(fd);

  if (ret == -1) {
    // In containerized environments, read could return EINVAL if the GID index is not mapped to the
    // container sysfs. In this case return ncclSuccess and let the caller move to next GID index.
    if (errno == EINVAL) return ncclSuccess;
    WARN("NET/IB: read failed in ncclIbRoceGetVersionNum: %s", strerror(errno));
    return ncclSystemError;
  }

  if (strlen(gidRoceVerStr)) {
    if (strncmp(gidRoceVerStr, "IB/RoCE v1", strlen("IB/RoCE v1")) == 0 || strncmp(gidRoceVerStr, "RoCE v1", strlen("RoCE v1")) == 0) {
      *version = 1;
    } else if (strncmp(gidRoceVerStr, "RoCE v2", strlen("RoCE v2")) == 0) {
      *version = 2;
    }
  }

  return ncclSuccess;
}

static ncclResult_t ncclUpdateGidIndex(struct ibv_context* context, uint8_t portNum, sa_family_t af, void* prefix, int prefixlen, int roceVer, int gidIndexCandidate, int* gidIndex) {
  union ibv_gid gid, gidCandidate;
  NCCLCHECK(wrap_ibv_query_gid(context, portNum, *gidIndex, &gid));
  NCCLCHECK(wrap_ibv_query_gid(context, portNum, gidIndexCandidate, &gidCandidate));

  sa_family_t usrFam = af;
  sa_family_t gidFam = getGidAddrFamily(&gid);
  sa_family_t gidCandidateFam = getGidAddrFamily(&gidCandidate);
  bool gidCandidateMatchSubnet = matchGidAddrPrefix(usrFam, prefix, prefixlen, &gidCandidate);

  if (gidCandidateFam != gidFam && gidCandidateFam == usrFam && gidCandidateMatchSubnet) {
    *gidIndex = gidIndexCandidate;
  } else {
    if (gidCandidateFam != usrFam || !validGid(&gidCandidate) || !gidCandidateMatchSubnet) {
      return ncclSuccess;
    }
    int usrRoceVer = roceVer;
    int gidRoceVerNum, gidRoceVerNumCandidate = -1;
    const char* deviceName = wrap_ibv_get_device_name(context->device);
    NCCLCHECK(ncclIbRoceGetVersionNum(deviceName, portNum, *gidIndex, &gidRoceVerNum));
    NCCLCHECK(ncclIbRoceGetVersionNum(deviceName, portNum, gidIndexCandidate, &gidRoceVerNumCandidate));
    if ((gidRoceVerNum != gidRoceVerNumCandidate || !validGid(&gid)) && gidRoceVerNumCandidate == usrRoceVer) {
      *gidIndex = gidIndexCandidate;
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclIbGetGidIndex(struct ibv_context *context, uint8_t portNum, struct ibv_port_attr* portAttr, int *gidIndex) {
  int gidTblLen = portAttr->gid_tbl_len;

  //for IB, choose GID Index that will have routable FLID if present
  if (portAttr->link_layer == IBV_LINK_LAYER_INFINIBAND) {
    union ibv_gid gid;
    int routableGidIndex = ncclParamIbRoutableFlidIbGidIndex();
    if (routableGidIndex < gidTblLen) {
      NCCLCHECK(wrap_ibv_query_gid(context, portNum, routableGidIndex, &gid));
      if (ncclIbExtractFlid(&gid) != 0) {
        *gidIndex = routableGidIndex;
        return ncclSuccess;
      }
    }
    *gidIndex = 0;
    return ncclSuccess;
  }

  //for ROCE
  *gidIndex = ncclParamIbGidIndex();
  if (*gidIndex >= 0) {
    return ncclSuccess;
  }

  sa_family_t userAddrFamily = envIbAddrFamily();
  int userRoceVersion = ncclParamIbRoceVersionNum();
  int prefixlen;
  void *prefix = envIbAddrRange(userAddrFamily, &prefixlen);

  *gidIndex = 0;
  for (int gidIndexNext = 1; gidIndexNext < gidTblLen; ++gidIndexNext) {
    NCCLCHECK(ncclUpdateGidIndex(context, portNum, userAddrFamily, prefix, prefixlen, userRoceVersion, gidIndexNext, gidIndex));
  }

  return ncclSuccess;
}
ncclResult_t ncclIbQpInit(struct ncclIbQp* qp) {
  struct ncclIbQpInitAttr* initAttr = &qp->initAttr;
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = initAttr->state;
  qpAttr.pkey_index = initAttr->pkeyIndex;
  qpAttr.port_num = initAttr->portNum;
  qpAttr.qp_access_flags = initAttr->qpAccessFlags;
  NCCLCHECK(wrap_ibv_modify_qp(qp->qp, &qpAttr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));
  return ncclSuccess;
}

static ncclResult_t ncclIbCreateQpMlx5(struct ncclIbQpCreateAttr* createQpAttrs, struct ncclIbQp* qp) {
  struct ibv_qp_init_attr_ex qpInitAttr;
  struct mlx5dv_qp_init_attr dvAttr;
  memset(&qpInitAttr, 0, sizeof(struct ibv_qp_init_attr_ex));
  memset(&dvAttr, 0 , sizeof(struct mlx5dv_qp_init_attr));
  qpInitAttr.qp_context = createQpAttrs->qpContext;
  qpInitAttr.send_cq = createQpAttrs->cq;
  qpInitAttr.recv_cq = createQpAttrs->cq;
  qpInitAttr.qp_type = createQpAttrs->type;
  qpInitAttr.cap.max_recv_wr = createQpAttrs->maxRecvWorkRequest;
  qpInitAttr.cap.max_send_wr = createQpAttrs->maxSendWorkRequest;
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.cap.max_inline_data = ncclParamIbUseInline() ? sizeof(struct ncclIbSendFifo) : 0;

  qpInitAttr.comp_mask = IBV_QP_INIT_ATTR_PD;
  qpInitAttr.pd = createQpAttrs->pd;

  if (createQpAttrs->oooRq) {
    dvAttr.create_flags |= MLX5DV_QP_CREATE_OOO_DP;
    dvAttr.comp_mask |= MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS;
  }
  qp->qp = wrap_mlx5dv_create_qp(createQpAttrs->pd->context, &qpInitAttr, &dvAttr);
  if (qp->qp == NULL) { WARN("NET/IB: %s: mlx5dv_create_qp failed to create QP: %m", __func__);  return ncclInternalError; }
  return ncclSuccess;
}

ncclResult_t ncclIbQpCreate(struct ncclIbQp* qp, struct ncclIbQpCreateAttr* createQpAttrs) {
  if (createQpAttrs->oooRq) {
     NCCLCHECK(ncclIbCreateQpMlx5(createQpAttrs, qp));
     return ncclSuccess;
  }
  struct ibv_qp_init_attr qpInitAttr;
  memset(&qpInitAttr, 0, sizeof(struct ibv_qp_init_attr));
  qpInitAttr.qp_context = createQpAttrs->qpContext;
  qpInitAttr.send_cq = createQpAttrs->cq;
  qpInitAttr.recv_cq = createQpAttrs->cq;
  qpInitAttr.qp_type = createQpAttrs->type;
  qpInitAttr.cap.max_recv_wr = createQpAttrs->maxRecvWorkRequest;
  qpInitAttr.cap.max_send_wr = createQpAttrs->maxSendWorkRequest;
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.cap.max_inline_data = ncclParamIbUseInline() ? sizeof(struct ncclIbSendFifo) : 0;
  NCCLCHECK(wrap_ibv_create_qp(&qp->qp, createQpAttrs->pd, &qpInitAttr));
  return ncclSuccess;
}

ncclResult_t ncclIbQpRtr(struct ncclIbQp* qp) {
  struct ncclIbQpRtrAttr* rtrAttr = &qp->rtrAttr;
  struct ibv_qp_attr qpAttr;
  int attrMask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = rtrAttr->mtu;
  qpAttr.dest_qp_num = rtrAttr->remoteQpNum;
  qpAttr.rq_psn = 0;
  if (qp->qp->qp_type != IBV_QPT_UC) {
    qpAttr.max_dest_rd_atomic = 1;
    qpAttr.min_rnr_timer = 12;
    attrMask |= IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
  }
  if (rtrAttr->linkLayer == IBV_LINK_LAYER_ETHERNET) {
    qpAttr.ah_attr.is_global = 1;
    qpAttr.ah_attr.grh.dgid.global.subnet_prefix = rtrAttr->remoteGid.global.subnet_prefix;
    qpAttr.ah_attr.grh.dgid.global.interface_id = rtrAttr->remoteGid.global.interface_id;
    qpAttr.ah_attr.grh.flow_label = 0;
    qpAttr.ah_attr.grh.sgid_index = rtrAttr->localGidIndex;
    qpAttr.ah_attr.grh.hop_limit = 255;
    qpAttr.ah_attr.grh.traffic_class = rtrAttr->tc;
  } else {
    //pick lid if subnet prefixs are same, FLID if they are not
    if (ncclIbExtractLocalSubnetPrefix(rtrAttr->localGid.global.subnet_prefix) ==
        ncclIbExtractLocalSubnetPrefix(rtrAttr->remoteGid.global.subnet_prefix)) {
      qpAttr.ah_attr.is_global = 0;
      qpAttr.ah_attr.dlid = rtrAttr->remoteLid;
    } else {
      uint16_t flid = ncclIbExtractFlid(&rtrAttr->remoteGid);
      if (flid == 0) {
        WARN("Warning: remote FLID configured as zero even when endpoints are on different subnets, using dlid as fallback");
        qpAttr.ah_attr.dlid = rtrAttr->remoteLid;
      } else {
        qpAttr.ah_attr.dlid = ncclIbExtractFlid(&rtrAttr->remoteGid);
      }
      qpAttr.ah_attr.is_global = 1;
      qpAttr.ah_attr.grh.dgid.global.subnet_prefix = rtrAttr->remoteGid.global.subnet_prefix;
      qpAttr.ah_attr.grh.dgid.global.interface_id = rtrAttr->remoteGid.global.interface_id;
      qpAttr.ah_attr.grh.sgid_index = rtrAttr->localGidIndex;
      qpAttr.ah_attr.grh.hop_limit = 255;
    }
  }
  qpAttr.ah_attr.sl = rtrAttr->sl;
  qpAttr.ah_attr.src_path_bits = 0;
  qpAttr.ah_attr.port_num = rtrAttr->localIbPort;
  TRACE(NCCL_NET, "NET/IB: %s: qpn=%u mtu=%d dst=%u ll=%u port=%u sl: %d tc: %d", __func__, qp->qp->qp_num, qpAttr.path_mtu, qpAttr.dest_qp_num, rtrAttr->linkLayer, qpAttr.ah_attr.port_num, qpAttr.ah_attr.sl, qpAttr.ah_attr.grh.traffic_class);
  NCCLCHECK(wrap_ibv_modify_qp(qp->qp, &qpAttr, attrMask));
  return ncclSuccess;
}

ncclResult_t ncclIbQpRts(struct ncclIbQp* qp) {
  struct ncclIbQpRtsAttr* rtsAttr = &qp->rtsAttr;
  struct ibv_qp_attr qpAttr;
  int attrMask = IBV_QP_STATE | IBV_QP_SQ_PSN;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTS;
  if (qp->qp->qp_type != IBV_QPT_UC) {
    qpAttr.timeout = rtsAttr->timeout;
    qpAttr.retry_cnt = rtsAttr->retryCnt;
    qpAttr.rnr_retry = 7;
    qpAttr.max_rd_atomic = 1;
    attrMask |= IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC;
  }
  qpAttr.sq_psn = 0;
  NCCLCHECK(wrap_ibv_modify_qp(qp->qp, &qpAttr, attrMask));
  return ncclSuccess;
}

ncclResult_t ncclIbQpReset(struct ncclIbQp* qp) {
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RESET;
  NCCLCHECK(wrap_ibv_modify_qp(qp->qp, &attr, IBV_QP_STATE));
  return ncclSuccess;
}

ncclResult_t ncclIbQpError(struct ncclIbQp* qp) {
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_ERR;
  NCCLCHECK(wrap_ibv_modify_qp(qp->qp, &attr, IBV_QP_STATE));
  return ncclSuccess;
}

ncclResult_t ncclIbListen(void* ctx, int dev, void* opaqueHandle, void** listenComm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclIbListenComm* comm;
  NCCLCHECK(ncclCalloc(&comm, 1));
  struct ncclIbHandle* handle = (struct ncclIbHandle*) opaqueHandle;
  static_assert(sizeof(struct ncclIbHandle) < NCCL_NET_HANDLE_MAXSIZE, "ncclIbHandle size too large");
  memset(handle, 0, sizeof(struct ncclIbHandle));
  comm->dev = dev;
  handle->magic = NCCL_SOCKET_MAGIC;
  NCCLCHECKGOTO(ncclSocketInit(&comm->sock, &ncclIbIfAddr, handle->magic, ncclSocketTypeNetIb, NULL, 1), ret, fail);
  NCCLCHECKGOTO(ncclSocketListen(&comm->sock), ret, fail);
  NCCLCHECKGOTO(ncclSocketGetAddr(&comm->sock, &handle->connectAddr), ret, fail);
  *listenComm = comm;
exit:
  return ret;
fail:
  (void)ncclSocketClose(&comm->sock);
  free(comm);
  goto exit;
}

#define NCCL_IB_SL_DEFAULT 0
#define NCCL_IB_TC_DEFAULT 0

// The function creates and initializes QPs (modifies the QPs to INIT) on the
// sender side. Afterwards it populates the metadata structure, provided to the
// function (meta), with the QPs' information. Note that after the QPs'
// creation, the QPs are also queried for ECE support and the metadata structure
// is updated accordingly. The meta data structure is then expected to be
// delivered to the remote side (receiver) as part of the connection
// establishment process.
static ncclResult_t ncclIbSenderQpsCreate(ncclIbSendComm* comm, struct ncclIbConnectionMetadata* meta) {
  uint nqps = comm->base.nqps;
  struct ncclIbQpCreateAttr qpCreateAttrs;
  memset(&qpCreateAttrs, 0, sizeof(struct ncclIbQpCreateAttr));
  qpCreateAttrs.type = IBV_QPT_RC;
  qpCreateAttrs.maxRecvWorkRequest = 0;
  // Send requests are sent using at most 2 messages (RDMA Write and RDMA Write with Immediate)
  qpCreateAttrs.maxSendWorkRequest = 2*NET_IB_MAX_REQUESTS;
  for (int qpIndex = 0; qpIndex < nqps; qpIndex++) {
    // The QPs are created in a "striped" manner across the available devices.
    // For example, if there are 2 devices and 4 QPs, the QPs will be created
    // on the devices as follows:
    // Dev0 -> QP0, QP2
    // Dev1 -> QP1, QP3
    uint devIndex = qpIndex % comm->base.vProps.ndevs;
    ncclIbSendCommDev* commDev = &comm->devs[devIndex];
    ncclIbDev* ibDev = &ncclIbDevs[commDev->base.ibDevN];
    ncclIbQp* localQp = &comm->base.qps[qpIndex];
    ncclIbQpInfo* localQpInfo = &meta->qpInfo[qpIndex];

    qpCreateAttrs.cq = commDev->base.cq;
    qpCreateAttrs.pd = commDev->base.pd;
    qpCreateAttrs.qpContext = &comm->base.stats;

    if (ibDev->ibProvider == IB_PROVIDER_MLX5 && ncclParamIbOooRq()) {
      if (ibDev->ar == 0) {
        WARN("NET/IB: %s: OOO RQ is force enabled but AR is not enabled, which is required for OOO RQ (device=%s)", __func__, ibDev->devName);
        return ncclInternalError;
      }
      qpCreateAttrs.oooRq = (comm->base.remOooRq && comm->base.localOooRq);
      if (!qpCreateAttrs.oooRq) {
        WARN("NET/IB: %s: OOO RQ is force enabled but not supported on both sides of the connection (device=%s, localOooRq=%d, remOooRq=%d)",
          __func__, ibDev->devName, comm->base.localOooRq, comm->base.remOooRq);
        return ncclInternalError;
      }
    }

    NCCLCHECK(ncclIbQpCreate(localQp, &qpCreateAttrs));
    INFO(NCCL_NET, "NET/IB: %s: QP created: port=%d dev=%d devName=%s ndevs=%d nmdevs=%d qp_num=%u pkey=%u pd=%p oooRq=%d",
        __func__,
        ibDev->portNum,
        commDev->base.ibDevN,
        ncclIbDevs[commDev->base.ibDevN].devName,
        ncclNIbDevs,
        ncclNMergedIbDevs,
        localQp->qp->qp_num,
        (uint16_t)ncclParamIbPkey(),
        commDev->base.pd,
        qpCreateAttrs.oooRq);
    localQp->devIndex = devIndex;

    // Populate the metadata that will be delivered to the remote peer
    localQpInfo->qpn      = localQp->qp->qp_num;
    localQpInfo->devIndex = localQp->devIndex;

    // Transition the QP to INIT state
    struct ncclIbQpInitAttr* initAttr = &localQp->initAttr;
    initAttr->state = IBV_QPS_INIT;
    initAttr->pkeyIndex = ncclParamIbPkey();
    initAttr->portNum = ibDev->portNum;
    initAttr->qpAccessFlags = IBV_ACCESS_REMOTE_WRITE;
    NCCLCHECK(ncclIbQpInit(localQp));

    if (ncclParamIbEceEnable()) {
      // Query ECE (Enhanced Connection Establishment) capabilities and
      // populate the initial ECE into the metadata structure that is sent to
      // the remote (receiver) side.
      NCCLCHECK(wrap_ibv_query_ece(localQp->qp, &localQpInfo->ece, &localQp->eceSupported));
      localQpInfo->ece_supported = localQp->eceSupported;
    } else {
      // Declare to the remote side that ECE is not supported
      localQpInfo->ece_supported = 0;
      // Store locally that ECE is not supported
      localQp->ece = {0};
      localQp->eceSupported = 0;
    }
  }

  if (comm->base.resiliency) {
    ncclIbResiliencySenderCreateQps(comm->base.resiliency, &meta->resiliencyInfo);
  }

  return ncclSuccess;
}

// The function modifies the QPs on the sender side to RTR and RTS states. It
// uses the remote metadata (remMeta) provided to the function to get the remote
// QPs' information. The remote metadata is expected to be obtained from the
// remote side (receiver) as part of the connection establishment process.
// Note that if ECE is supported, the function sets up the reduced ECE (which
// was delivered from the receiver side) on the QPs before modifying the QPs
// to RTR.
static ncclResult_t ncclIbSenderQpsToRts(ncclIbSendComm* comm, struct ncclIbConnectionMetadata* remMeta) {
  uint nqps = comm->base.nqps;
  for (int qpIndex = 0; qpIndex < nqps; qpIndex++) {
    ncclIbQp* localQp = &comm->base.qps[qpIndex];
    ncclIbSendCommDev* commDev = &comm->devs[localQp->devIndex];
    ncclIbDev* ibDev = &ncclIbDevs[commDev->base.ibDevN];
    ncclIbQpInfo* remQpInfo   = &remMeta->qpInfo[qpIndex];
    ncclIbDevInfo* remDevInfo = &remMeta->devs[remQpInfo->devIndex];

    localQp->remDevIdx = remQpInfo->devIndex;

    if (localQp->eceSupported && remQpInfo->ece_supported) {
      INFO(NCCL_NET,"NET/IB: %s: Set ECE: IbDev %d Port %d qp_num %d set_ece={supported=%d, vendor_id=0x%x, options=0x%x, comp_mask=0x%x}", __func__, commDev->base.ibDevN, ibDev->portNum, localQp->qp->qp_num, remQpInfo->ece_supported, remQpInfo->ece.vendor_id, remQpInfo->ece.options, remQpInfo->ece.comp_mask);
      // Set the reduced ECE received from the receiver side
      NCCLCHECK(wrap_ibv_set_ece(localQp->qp, &remQpInfo->ece, &localQp->eceSupported));
      // Store the reduced ECE locally as well
      localQp->ece = remQpInfo->ece;
    } else {
      // If remote does not support ECE, disable it locally as well
      localQp->eceSupported = 0;
      localQp->ece = {0};
    }

    struct ncclIbQpRtrAttr *rtrAttr = &localQp->rtrAttr;
    rtrAttr->mtu = std::min(remDevInfo->mtu, ibDev->portAttr.active_mtu);
    rtrAttr->linkLayer = remDevInfo->link_layer;
    rtrAttr->tc = remDevInfo->link_layer == IBV_LINK_LAYER_ETHERNET ? remMeta->tc : -1;
    rtrAttr->sl = remMeta->sl;
    rtrAttr->remoteQpNum = remQpInfo->qpn;
    rtrAttr->remoteLid = remDevInfo->lid;
    rtrAttr->remoteGid = remDevInfo->gid;
    rtrAttr->localIbPort = remDevInfo->ib_port;
    rtrAttr->localGid = commDev->base.gidInfo.localGid;
    rtrAttr->localGidIndex = commDev->base.gidInfo.localGidIndex;
    NCCLCHECK(ncclIbQpRtr(localQp));
    struct ncclIbQpRtsAttr* rtsAttr = &localQp->rtsAttr;
    rtsAttr->timeout = ncclParamIbTimeout();
    rtsAttr->retryCnt = ncclParamIbRetryCnt();
    NCCLCHECK(ncclIbQpRts(localQp));
  }

  if (comm->base.resiliency) {
    NCCLCHECK(ncclIbResiliencySenderQpsToRts(comm->base.resiliency, remMeta));
  }

  return ncclSuccess;
}

ncclResult_t ncclIbConnect(void* ctx, int dev, void* opaqueHandle, void** sendComm, ncclNetDeviceHandle_t** /*sendDevComm*/) {
  ncclResult_t ret = ncclSuccess;
  struct ncclIbHandle* handle = (struct ncclIbHandle*) opaqueHandle;
  struct ncclIbCommStage* stage = &handle->stage;
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)stage->comm;
  int ready;

  uint8_t link_layer = IBV_LINK_LAYER_UNSPECIFIED;
  *sendComm = NULL;

  if (stage->state == ncclIbCommStateConnect)      goto ib_connect_check;
  if (stage->state == ncclIbCommStateSendDevList)  goto ib_send_dev_list;
  if (stage->state == ncclIbCommStateRecvDevList)  goto ib_recv_dev_list;
  if (stage->state == ncclIbCommStateSend)         goto ib_send;
  if (stage->state == ncclIbCommStateConnecting)   goto ib_connect;
  if (stage->state == ncclIbCommStateConnected)    goto ib_send_ready;
  if (stage->state != ncclIbCommStateStart) {
    WARN("Error: trying to connect already connected sendComm");
    return ncclInternalError;
  }
  stage->buffer = NULL;

  NCCLCHECK(ncclIbMalloc((void**)&comm, sizeof(struct ncclIbSendComm)));
  NCCLCHECKGOTO(ncclIbSendCommInit(comm), ret, fail);
  NCCLCHECKGOTO(ncclIbStatsInit(&comm->base.stats), ret, fail);
  NCCLCHECKGOTO(ncclSocketInit(&comm->base.sock, &handle->connectAddr, handle->magic, ncclSocketTypeNetIb, NULL, 1), ret, fail);
  stage->comm = comm;
  stage->state = ncclIbCommStateConnect;
  NCCLCHECKGOTO(ncclSocketConnect(&comm->base.sock), ret, fail);

ib_connect_check:
  /* since ncclSocketConnect is async, we must check if connection is complete */
  NCCLCHECKGOTO(ncclSocketReady(&comm->base.sock, &ready), ret, fail);
  if (!ready) return ncclSuccess;

  // IB Setup
  struct ncclIbMergedDev* mergedDev;
  if (dev >= ncclNMergedIbDevs) {
    WARN("NET/IB : Trying to use non-existent virtual device %d", dev);
    return ncclInternalError;
  }

  mergedDev = ncclIbMergedDevs + dev;
  comm->base.vProps = mergedDev->vProps;
  stage->state = ncclIbCommStateSendDevList;
  stage->offset = 0;
  struct ncclIbConnectionMetadata meta;
  NCCLCHECKGOTO(ncclIbMalloc((void**)&stage->buffer, sizeof(meta)), ret, fail);
  memcpy(stage->buffer, &mergedDev->vProps, sizeof(ncclNetVDeviceProps_t));

  struct ncclIbDevExtraProps exProps;
  exProps.oooRq = true;
  for (int i = 0; i < mergedDev->vProps.ndevs; i++) {
    int ibDevN = mergedDev->vProps.devs[i];
    exProps.oooRq = exProps.oooRq && ncclIbDevs[ibDevN].oooRqSize;
  }
  comm->base.localOooRq = exProps.oooRq;
  memcpy((char *)stage->buffer + sizeof(ncclNetVDeviceProps_t), &exProps, sizeof(struct ncclIbDevExtraProps));

// In the case of mismatched nDevs, we will make sure that both sides of a logical connection have the same number of RC qps
ib_send_dev_list:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, &comm->base.sock, stage->buffer, sizeof(ncclNetVDeviceProps_t) + sizeof(struct ncclIbDevExtraProps), &stage->offset));
  if (stage->offset != (sizeof(ncclNetVDeviceProps_t) + sizeof(struct ncclIbDevExtraProps))) return ncclSuccess;

  stage->state = ncclIbCommStateRecvDevList;
  stage->offset = 0;

ib_recv_dev_list:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->base.sock, stage->buffer, sizeof(ncclNetVDeviceProps_t) + sizeof(struct ncclIbDevExtraProps), &stage->offset));
  if (stage->offset != (sizeof(ncclNetVDeviceProps_t) + sizeof(struct ncclIbDevExtraProps))) return ncclSuccess;
  stage->offset = 0;
  ncclNetVDeviceProps_t remoteVProps;
  ncclNetCommConfig_t* config;
  memcpy(&remoteVProps, stage->buffer, sizeof(ncclNetVDeviceProps_t));
  memcpy(&exProps, (char *)stage->buffer + sizeof(ncclNetVDeviceProps_t), sizeof(exProps));
  comm->base.remOooRq = exProps.oooRq;

  mergedDev = ncclIbMergedDevs + dev;
  comm->base.vProps = mergedDev->vProps;
  int localNqps, remoteNqps;
  localNqps  = ncclParamIbQpsPerConn() * comm->base.vProps.ndevs; // We must have at least 1 qp per-device
  remoteNqps = ncclParamIbQpsPerConn() * remoteVProps.ndevs;
  comm->base.nqps = remoteNqps > localNqps ? remoteNqps : localNqps; // Select max nqps (local or remote)

  comm->base.nDataQps = std::max(comm->base.vProps.ndevs, remoteVProps.ndevs);

  if (comm->base.resiliency) {
    NCCLCHECK(ncclIbResiliencyDeviceNumSet(comm->base.resiliency, comm->base.vProps.ndevs, remoteVProps.ndevs));
  }

  // Init PD, Ctx for each IB device
  comm->ar = 1; // Set to 1 for logic
  // Sender's CQ size needs to accomodate the upper bound of number of send
  // requests multiplied by the number of QPs used per request.
  int cqSize;
  cqSize = NET_IB_MAX_REQUESTS*ncclParamIbQpsPerConn();
  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    int ibDevN = comm->base.vProps.devs[i];
    if (comm->base.resiliency) {
      ncclIbResiliencyDataCqSizeGet(comm->base.resiliency, i, &cqSize);
    }
    NCCLCHECKGOTO(ncclIbInitCommDevBase(ibDevN, &comm->devs[i].base, &comm->base.stats, cqSize), ret, fail);
    comm->ar = comm->ar && ncclIbDevs[ibDevN].ar; // ADAPTIVE_ROUTING - if all merged devs have it enabled
    if (comm->base.resiliency) {
      NCCLCHECKGOTO(ncclIbResiliencyDevInit(comm->base.resiliency, i, &ncclIbDevs[ibDevN]), ret, fail);
    }
  }

  memset(&meta, 0, sizeof(meta));
  meta.ndevs = comm->base.vProps.ndevs;

  // Create QPs on the sender side
  NCCLCHECKGOTO(ncclIbSenderQpsCreate(comm, &meta), ret, fail);

  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
    ncclIbSendCommDev* commDev = comm->devs + i;
    ncclIbDev* ibDev = ncclIbDevs + commDev->base.ibDevN;

    // Write to the metadata struct via this pointer
    ncclIbDevInfo* devInfo = meta.devs + i;
    devInfo->ib_port       = ibDev->portNum;
    devInfo->mtu           = ibDev->portAttr.active_mtu;
    devInfo->lid           = ibDev->portAttr.lid;

    // Prepare GIN Put Signal scratchpad (for RDMA Atomic result)
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&commDev->putSignalScratchpadMr, commDev->base.pd, &comm->putSignalScratchpad, sizeof(comm->putSignalScratchpad), IBV_ACCESS_LOCAL_WRITE), ret, fail);

    // Prepare my CTS FIFO
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&commDev->ctsFifoMr, commDev->base.pd, comm->ctsFifo, sizeof(comm->ctsFifo), IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ), ret, fail);
    devInfo->rkey = commDev->ctsFifoMr->rkey;

    // Pack local GID info
    devInfo->link_layer = commDev->base.gidInfo.link_layer = ibDev->portAttr.link_layer;
    NCCLCHECKGOTO(ncclIbGetGidIndex(ibDev->context, ibDev->portNum, &ibDev->portAttr, &commDev->base.gidInfo.localGidIndex), ret, fail);
    NCCLCHECKGOTO(wrap_ibv_query_gid(ibDev->context, ibDev->portNum, commDev->base.gidInfo.localGidIndex, &commDev->base.gidInfo.localGid), ret, fail);
    devInfo->gid.global.subnet_prefix = commDev->base.gidInfo.localGid.global.subnet_prefix;
    devInfo->gid.global.interface_id = commDev->base.gidInfo.localGid.global.interface_id;

    // info logging
    for (int q = 0; q < comm->base.nqps; q++) {
      // Print just the QPs for this dev
      if (comm->base.qps[q].devIndex == i) {
        if (devInfo->link_layer == IBV_LINK_LAYER_INFINIBAND) { // IB
          INFO(NCCL_NET,"NET/IB: %s: %s %d IbDev %d Port %d qp_num %d mtu %d LID %d subnet-prefix %lu  FLID %d ctsFifoRkey=0x%x ctsFifoLkey=0x%x", __func__,
               comm->base.vProps.ndevs > 2 ? "NCCL MergedDev" : "NCCL Dev",
               dev, commDev->base.ibDevN, ibDev->portNum, meta.qpInfo[q].qpn, devInfo->mtu, devInfo->lid,
               (uint64_t)devInfo->gid.global.subnet_prefix, ncclIbExtractFlid(&devInfo->gid), commDev->ctsFifoMr->rkey, commDev->ctsFifoMr->lkey);
        } else { // RoCE
          INFO(NCCL_NET,"NET/IB: %s: %s %d IbDev %d Port %d qp_num %d mtu %d GID %ld (%lX/%lX) ctsFifoRkey=0x%x ctsFifoLkey=0x%x", __func__,
               comm->base.vProps.ndevs > 2 ? "NCCL MergedDev" : "NCCL Dev", dev,
               commDev->base.ibDevN, ibDev->portNum, meta.qpInfo[q].qpn, devInfo->mtu,
               (int64_t)commDev->base.gidInfo.localGidIndex,
               (uint64_t)devInfo->gid.global.subnet_prefix, devInfo->gid.global.interface_id, commDev->ctsFifoMr->rkey, commDev->ctsFifoMr->lkey);
        }
        // Log ECE info
        if (meta.qpInfo[q].ece_supported) {
          INFO(NCCL_NET,"NET/IB: %s: IbDev %d Port %d qp_num %d query_ece={supported=%d, vendor_id=0x%x, options=0x%x, comp_mask=0x%x}", __func__,
               commDev->base.ibDevN, ibDev->portNum, meta.qpInfo[q].qpn,
               meta.qpInfo[q].ece_supported, meta.qpInfo[q].ece.vendor_id, meta.qpInfo[q].ece.options, meta.qpInfo[q].ece.comp_mask);
        }
      }
    }
    if (link_layer == IBV_LINK_LAYER_UNSPECIFIED) link_layer = devInfo->link_layer;
    if (link_layer != devInfo->link_layer) {
      int ibDev0 = comm->devs[0].base.ibDevN;
      WARN("NET/IB : Attempted to connect incompatible devices: [%d]%s:%d/%s and [%d]%s:%d/%s. Try selecting NICs of only one link type using NCCL_IB_HCA",
           commDev->base.ibDevN, ibDev->devName, ibDev->portNum, NCCL_IB_LLSTR(ibDev->portAttr.link_layer), ibDev0, ncclIbDevs[ibDev0].devName, ncclIbDevs[ibDev0].portNum, NCCL_IB_LLSTR(link_layer));
      return ncclInternalError;
    }
  }
  config = (ncclNetCommConfig_t*)ctx;
  meta.addr = (uint64_t)comm->ctsFifo;
  meta.sl = (ncclParamIbSl() != -1) ? ncclParamIbSl() : (config && config->trafficClass != NCCL_NET_TRAFFIC_CLASS_UNDEF) ? config->trafficClass : NCCL_IB_SL_DEFAULT;
  meta.tc = (ncclParamIbTc() != -1) ? ncclParamIbTc() : (config && config->trafficClass != NCCL_NET_TRAFFIC_CLASS_UNDEF) ? config->trafficClass : NCCL_IB_TC_DEFAULT;
  strncpy(meta.devName, mergedDev->devName, MAX_MERGED_DEV_NAME);

  stage->state = ncclIbCommStateSend;
  stage->offset = 0;

  memcpy(stage->buffer, &meta, sizeof(meta));

ib_send:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &comm->base.sock, stage->buffer, sizeof(meta), &stage->offset), ret, fail);
  if (stage->offset != sizeof(meta)) return ncclSuccess;

  stage->state = ncclIbCommStateConnecting;
  stage->offset = 0;
  // Clear the staging buffer for re-use
  memset(stage->buffer, 0, sizeof(meta));

ib_connect:
  struct ncclIbConnectionMetadata remMeta;
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->base.sock, stage->buffer, sizeof(ncclIbConnectionMetadata), &stage->offset), ret, fail);
  if (stage->offset != sizeof(remMeta)) return ncclSuccess;

  memcpy(&remMeta, stage->buffer, sizeof(ncclIbConnectionMetadata));

  // ensure that the remote devices have the same link layer than the local devices used in the connection.
  if (comm->base.vProps.ndevs > 0) {
    int ibDev0 = comm->devs[0].base.ibDevN;
    link_layer = ncclIbDevs[ibDev0].portAttr.link_layer;
    for (int i = 0; i < remMeta.ndevs; i++) {
      if (remMeta.devs[i].link_layer != link_layer) {
        WARN("NET/IB : Remote %s device is incompatible with the local [%d]%s:%d/%s. Try selecting NICs of only one link type using NCCL_IB_HCA",
             NCCL_IB_LLSTR(remMeta.devs[i].link_layer), ibDev0, ncclIbDevs[ibDev0].devName, ncclIbDevs[ibDev0].portNum, NCCL_IB_LLSTR(link_layer));
        return ncclInternalError;
      }
    }
  }

  // Store the number of remote devices
  comm->base.nRemDevs = remMeta.ndevs;

  // Store the remote GID information per-device provided by the remote peer
  for (int i = 0; i < comm->base.nRemDevs; i++) {
    comm->base.remDevs[i] = remMeta.devs[i];
    comm->base.remDevs[i].remoteGid.global.interface_id = comm->base.remDevs[i].gid.global.interface_id;
    comm->base.remDevs[i].remoteGid.global.subnet_prefix = comm->base.remDevs[i].gid.global.subnet_prefix;
  }

  // Store the completion records info provided by the remote
  comm->remCmplsRecords.addr = remMeta.addr;
  for (int i = 0; i < comm->base.nRemDevs; i++) {
    comm->remCmplsRecords.rkeys[i] = remMeta.devs[i].rkey;
    if (comm->base.resiliency) {
      NCCLCHECKGOTO(ncclIbResiliencyRemoteCompletionRecordsSet(comm->base.resiliency, comm->remCmplsRecords.rkeys[i], comm->remCmplsRecords.addr, i), ret, fail);
    }
  }

  for (int i=0; i < comm->base.vProps.ndevs; i++) {
    ncclIbSendCommDev* commDev = comm->devs + i;
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&commDev->cmplsRecordsMr, comm->devs[i].base.pd, &comm->remCmplsRecords.elems, sizeof(comm->remCmplsRecords.elems), IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_READ), ret, fail);
    comm->devs[i].sge.lkey = comm->devs[i].cmplsRecordsMr->lkey;
  }

  NCCLCHECKGOTO(ncclIbSenderQpsToRts(comm, &remMeta), ret, fail);

  comm->base.ready = 1;
  stage->state = ncclIbCommStateConnected;
  stage->offset = 0;

ib_send_ready:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &comm->base.sock, &comm->base.ready, sizeof(int), &stage->offset), ret, fail);
  if (stage->offset != sizeof(int)) return ncclSuccess;

  *sendComm = comm;
exit:
  if (stage->buffer) free(stage->buffer);
  stage->state = ncclIbCommStateStart;
  return ret;
fail:
  free(comm);
  goto exit;
}

NCCL_PARAM(IbWarnRailLocal, "IB_WARN_RAIL_LOCAL", 0);

ncclResult_t ncclIbCheckVProps(ncclNetVDeviceProps_t* vProps1, ncclNetVDeviceProps_t* vProps2) {
  ncclNetVDeviceProps_t  outVProps = {0};
  ncclNetVDeviceProps_t* minVProps = vProps2;
  ncclNetVDeviceProps_t* maxVProps = vProps1;
  if (vProps2->ndevs > vProps1->ndevs) {
    minVProps = vProps1;
    maxVProps = vProps2;
  }

  // Find the intersection of devices
  for (int i = 0; i < minVProps->ndevs; i++) {
    int dev = minVProps->devs[i];
    for (int j = 0; j < maxVProps->ndevs; j++) {
      // Found
      if (maxVProps->devs[j] == dev) {
        outVProps.devs[outVProps.ndevs++] = dev;
      }
    }
  }

  // In the case that at least one side has a fused NIC but there are no matching physical NICs, we should check if the user wants this
  if (ncclParamIbWarnRailLocal() && outVProps.ndevs < maxVProps->ndevs) {
    char local[128];
    int cursor = 1;
    snprintf(local, sizeof(local), "%d", vProps1->devs[0]);
    for (int i = 1; i < vProps1->ndevs; i++) {
      snprintf(local+cursor, sizeof(local)-cursor, ",%d", vProps1->devs[i]);
      cursor += 2;
    }
    char remote[128];
    snprintf(remote, sizeof(remote), "%d", vProps2->devs[0]);
    cursor = 1;
    for (int i = 1; i < vProps2->ndevs; i++) {
      snprintf(remote+cursor, sizeof(remote)-cursor, ",%d", vProps2->devs[i]);
      cursor += 2;
    }
    INFO(NCCL_NET, "NET/IB : There are mismatched physical devices between local (%s) and remote (%s). To disable this warning, set NCCL_IB_WARN_RAIL_LOCAL=0", local, remote);
  }

  return ncclSuccess;
}

// The function creates and modifies QPs to RTS state on the receiver side
// using remote information from the sender side (remMeta). It also populates
// the remote metadata structure, provided to the function (remMeta), with the
// QPs' information so that data structure could be delivered to the remote
// side (sender) as part of the connection establishment process.
static ncclResult_t ncclIbReceiverQpsCreateToRts(ncclIbRecvComm* rComm, struct ncclIbConnectionMetadata* remMeta, struct ncclIbConnectionMetadata* meta) {
  uint nqps = rComm->base.nqps;
  struct ncclIbQpCreateAttr qpCreateAttrs;
  memset(&qpCreateAttrs, 0, sizeof(struct ncclIbQpCreateAttr));
  qpCreateAttrs.type = IBV_QPT_RC;
  qpCreateAttrs.maxRecvWorkRequest = NET_IB_MAX_REQUESTS;
  // CTS messages are posted using send work requests.
  // Note that because only specific CTS messages are signaled, the send queue
  // size needs to be double the number of max requests.
  // When resiliency is enabled, the number of send work requests is as the
  // number of max requests because every CTS message is signaled.
  qpCreateAttrs.maxSendWorkRequest = NET_IB_MAX_REQUESTS * (rComm->base.resiliency ? 1 : 2);
  for (int qpIndex = 0; qpIndex < nqps; qpIndex++) {
    // The QPs are created in a "striped" manner across the available devices.
    // For example, if there are 2 devices and 4 QPs, the QPs will be created
    // on the devices as follows:
    // Dev0 -> QP0, QP2
    // Dev1 -> QP1, QP3
    uint devIndex = qpIndex % rComm->base.vProps.ndevs;
    ncclIbRecvCommDev* rCommDev = &rComm->devs[devIndex];
    ncclIbDev* ibDev = &ncclIbDevs[rCommDev->base.ibDevN];
    ncclIbQpInfo* remQpInfo = &remMeta->qpInfo[qpIndex];
    ncclIbQpInfo* localQpInfo = &meta->qpInfo[qpIndex];
    int remDevIndex = remQpInfo->devIndex;
    ncclIbDevInfo* remDevInfo = &remMeta->devs[remDevIndex];
    ncclIbQp* localQp = &rComm->base.qps[qpIndex];

    localQp->remDevIdx = remDevIndex;
    localQp->devIndex = devIndex;

    qpCreateAttrs.cq = rCommDev->base.cq;
    qpCreateAttrs.pd = rCommDev->base.pd;
    qpCreateAttrs.qpContext = &rComm->base.stats;
    if (rComm->base.resiliency) {
      ncclIbResiliencyDataRqSizeGet(rComm->base.resiliency, devIndex, &qpCreateAttrs.maxRecvWorkRequest);
    }
    if (ibDev->ibProvider == IB_PROVIDER_MLX5 && ncclParamIbOooRq()) {
      if (ibDev->ar == 0) {
        WARN("NET/IB: %s: OOO RQ is force enabled but AR is not enabled, which is required for OOO RQ (device=%s)", __func__, ibDev->devName);
        return ncclInternalError;
      }
      qpCreateAttrs.oooRq = (rComm->base.remOooRq && rComm->base.localOooRq);
      // out-of-order recv prerequisite: oooRq is supported on both sides
      if (!qpCreateAttrs.oooRq) {
        WARN("NET/IB: %s: OOO RQ is force enabled but not supported on both sides of the connection (device=%s, localOooRq=%d, remOooRq=%d)",
          __func__, ibDev->devName, rComm->base.localOooRq, rComm->base.remOooRq);
        return ncclInternalError;
      }
      // out-of-order recv prerequisite: oooRq size requirements are met
      if (ibDev->oooRqSize < qpCreateAttrs.maxRecvWorkRequest) {
        WARN("NET/IB: %s: OOO RQ is force enabled but size %u is less than the required recv work request size %u on device:%s",
          __func__, ibDev->oooRqSize, qpCreateAttrs.maxRecvWorkRequest, ibDev->devName);
        return ncclInternalError;
      }
    }
    NCCLCHECK(ncclIbQpCreate(localQp, &qpCreateAttrs));
    INFO(NCCL_NET, "NET/IB: %s: QP created: port=%d dev=%d devName=%s ndevs=%d nmdevs=%d qp_num=%u pkey=%u pd=%p oooRq=%d",
        __func__,
        ibDev->portNum,
        rCommDev->base.ibDevN,
        ncclIbDevs[rCommDev->base.ibDevN].devName,
        ncclNIbDevs,
        ncclNMergedIbDevs,
        localQp->qp->qp_num,
        (uint16_t)ncclParamIbPkey(),
        rCommDev->base.pd,
        qpCreateAttrs.oooRq);

    localQpInfo->qpn      = localQp->qp->qp_num;
    localQpInfo->devIndex = localQp->devIndex;

    // Transition the QP to INIT state
    struct ncclIbQpInitAttr* initAttr = &localQp->initAttr;
    initAttr->state = IBV_QPS_INIT;
    initAttr->pkeyIndex = ncclParamIbPkey();
    initAttr->portNum = ibDev->portNum;
    // Remote Atomic operations are used for GIN! REMOTE_READ is required for GIN Get (RDMA READ).
    initAttr->qpAccessFlags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC | IBV_ACCESS_REMOTE_READ;
    NCCLCHECK(ncclIbQpInit(localQp));

    if (remQpInfo->ece_supported) {
      // Set the ECE received from the remote (sender) side.
      // coverity[copy_paste_error]
      NCCLCHECK(wrap_ibv_set_ece(localQp->qp, &remQpInfo->ece, &localQpInfo->ece_supported));
    } else {
      localQpInfo->ece_supported = 0;
      localQp->ece = {0};
      localQp->eceSupported = 0;
    }

    // Reduce the local MTU to match the remote MTU if needed
    ibDev->portAttr.active_mtu = std::min(ibDev->portAttr.active_mtu, remDevInfo->mtu);

    struct ncclIbQpRtrAttr *rtrAttr = &localQp->rtrAttr;
    rtrAttr->mtu = ibDev->portAttr.active_mtu;
    rtrAttr->linkLayer = remDevInfo->link_layer;
    rtrAttr->tc = (remDevInfo->link_layer == IBV_LINK_LAYER_ETHERNET && ncclParamIbFifoTc() != -1) ? ncclParamIbFifoTc() : remMeta->tc;
    rtrAttr->sl = remMeta->sl;
    rtrAttr->remoteQpNum = remQpInfo->qpn;
    rtrAttr->remoteLid = remDevInfo->lid;
    rtrAttr->remoteGid = remDevInfo->gid;
    rtrAttr->localIbPort = remDevInfo->ib_port;
    rtrAttr->localGid = rCommDev->base.gidInfo.localGid;
    rtrAttr->localGidIndex = rCommDev->base.gidInfo.localGidIndex;
    NCCLCHECK(ncclIbQpRtr(localQp));
    struct ncclIbQpRtsAttr* rtsAttr = &localQp->rtsAttr;
    rtsAttr->timeout = ncclParamIbTimeout();
    rtsAttr->retryCnt = ncclParamIbRetryCnt();
    NCCLCHECK(ncclIbQpRts(localQp));

    // Query the reduced ECE by the device and storing it in the local QP info
    // to return it to the requestor (sender).
    if (remQpInfo->ece_supported && localQpInfo->ece_supported) {
      NCCLCHECK(wrap_ibv_query_ece(localQp->qp, &localQpInfo->ece, &localQpInfo->ece_supported));
      // Store the reduced ECE locally as well
      localQp->ece = localQpInfo->ece;
      localQp->eceSupported = localQpInfo->ece_supported;
    } else {
      localQp->ece = {0};
      localQp->eceSupported = 0;
    }
  }

  if (rComm->flushEnabled) {
    for (int i = 0; i < rComm->base.vProps.ndevs; i++) {
      ncclIbRecvCommDev* rCommDev = &rComm->devs[i];
      ncclIbDev* ibDev = &ncclIbDevs[rCommDev->base.ibDevN];

      struct ncclIbQpCreateAttr qpCreateAttrs;
      memset(&qpCreateAttrs, 0, sizeof(struct ncclIbQpCreateAttr));
      qpCreateAttrs.type = IBV_QPT_RC;
      qpCreateAttrs.cq = rCommDev->base.cq;
      qpCreateAttrs.pd = rCommDev->base.pd;
      qpCreateAttrs.maxRecvWorkRequest = 0;
      qpCreateAttrs.maxSendWorkRequest = NET_IB_MAX_REQUESTS;
      qpCreateAttrs.qpContext = &rComm->base.stats;
      NCCLCHECK(ncclIbQpCreate(&rCommDev->gpuFlush.qp, &qpCreateAttrs));
      INFO(NCCL_NET, "NET/IB: %s: Flush QP created: port=%d dev=%d devName=%s ndevs=%d nmdevs=%d qp_num=%u pkey=%u pd=%p",
          __func__,
          ibDev->portNum,
          rCommDev->base.ibDevN,
          ncclIbDevs[rCommDev->base.ibDevN].devName,
          ncclNIbDevs,
          ncclNMergedIbDevs,
          rCommDev->gpuFlush.qp.qp->qp_num,
          (uint16_t)ncclParamIbPkey(),
          rCommDev->base.pd);

      ncclIbQp* flushQp = &rCommDev->gpuFlush.qp;

      // Transition the QP to INIT state
      struct ncclIbQpInitAttr* initAttr = &flushQp->initAttr;
      initAttr->state = IBV_QPS_INIT;
      initAttr->pkeyIndex = ncclParamIbPkey();
      initAttr->portNum = ibDev->portNum;
      initAttr->qpAccessFlags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ;
      NCCLCHECK(ncclIbQpInit(flushQp));

      struct ncclIbQpRtrAttr *rtrAttr = &flushQp->rtrAttr;
      rtrAttr->mtu = ibDev->portAttr.active_mtu;
      rtrAttr->linkLayer = ibDev->portAttr.link_layer;
      // TODO: Flush QP is a "loopback QP" (connected to itself), so it should
      // not use any information from the remote side during configuration.
      rtrAttr->tc = ibDev->portAttr.link_layer == IBV_LINK_LAYER_ETHERNET ? remMeta->tc : -1;
      rtrAttr->sl = remMeta->sl;
      rtrAttr->remoteQpNum = rCommDev->gpuFlush.qp.qp->qp_num;
      rtrAttr->remoteLid = ibDev->portAttr.lid;
      rtrAttr->remoteGid = rCommDev->base.gidInfo.localGid;
      rtrAttr->localIbPort = ibDev->portNum;
      rtrAttr->localGid = rCommDev->base.gidInfo.localGid;
      rtrAttr->localGidIndex = rCommDev->base.gidInfo.localGidIndex;
      NCCLCHECK(ncclIbQpRtr(flushQp));
      struct ncclIbQpRtsAttr* rtsAttr = &flushQp->rtsAttr;
      rtsAttr->timeout = ncclParamIbTimeout();
      rtsAttr->retryCnt = ncclParamIbRetryCnt();
      NCCLCHECK(ncclIbQpRts(flushQp));
    }
  }

  if (rComm->base.resiliency) {
    NCCLCHECK(ncclIbResiliencyReceiverQpsCreateToRts(rComm->base.resiliency, remMeta, &meta->resiliencyInfo));
  }

  return ncclSuccess;
}

ncclResult_t ncclIbPostReceiveWorkRequestsOnQp(struct ncclIbRecvComm* recvComm, ncclIbQp* dataQp) {
  uint32_t nRecvWorkRequestsPerQp = NET_IB_MAX_REQUESTS;
  if (recvComm->base.resiliency) {
    ncclIbResiliencyDataRqSizeGet(recvComm->base.resiliency, dataQp->devIndex, &nRecvWorkRequestsPerQp);
  }
  INFO(NCCL_NET, "NET/IB: %s: Pre-posting %d Receive WQEs on QP (qp_num=%d, comm=%p)", __func__, nRecvWorkRequestsPerQp, dataQp->qp->qp_num, recvComm);
  for (int j = 0; j < nRecvWorkRequestsPerQp; j++) {
    NCCLCHECK(ncclIbPostRecvWorkRequest(dataQp->qp, &recvComm->ibRecvWorkRequest));
  }
  return ncclSuccess;
}

ncclResult_t ncclIbReceiverPrePostReceiveWorkRequests(struct ncclIbRecvComm* recvComm) {
  int nqps = recvComm->base.nqps;
  for (int i = 0; i < nqps; i++) {
    NCCLCHECK(ncclIbPostReceiveWorkRequestsOnQp(recvComm, &recvComm->base.qps[i]));
  }
  return ncclSuccess;
}

NCCL_PARAM(IbGdrFlushDisable, "GDR_FLUSH_DISABLE", 0);

ncclResult_t ncclIbAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** /*recvDevComm*/) {
  ncclResult_t ret = ncclSuccess;
  struct ncclIbListenComm* lComm = (struct ncclIbListenComm*)listenComm;
  struct ncclIbCommStage* stage = lComm->stage;
  if (stage == NULL) {
    NCCLCHECK(ncclCalloc(&lComm->stage, 1));
    stage = lComm->stage;
  }
  struct ncclIbRecvComm* rComm = (struct ncclIbRecvComm*)stage->comm;
  int ready;
  int link_layer = IBV_LINK_LAYER_UNSPECIFIED;
  *recvComm = NULL;

  if (stage->state == ncclIbCommStateAccept)   goto ib_accept_check;
  if (stage->state == ncclIbCommStateRecvDevList) goto ib_recv_dev_list;
  if (stage->state == ncclIbCommStateSendDevList) goto ib_send_dev_list;
  if (stage->state == ncclIbCommStateRecv) goto ib_recv;
  if (stage->state == ncclIbCommStateSend) goto ib_send;
  if (stage->state == ncclIbCommStatePendingReady) goto ib_recv_ready;
  if (stage->state != ncclIbCommStateStart) {
    WARN("Listencomm in unknown state %d", stage->state);
    return ncclInternalError;
  }

  NCCLCHECK(ncclIbMalloc((void**)&rComm, sizeof(struct ncclIbRecvComm)));
  NCCLCHECKGOTO(ncclIbRecvCommInit(rComm), ret, fail);
  NCCLCHECKGOTO(ncclIbStatsInit(&rComm->base.stats), ret, fail);
  stage->comm = rComm;
  stage->state = ncclIbCommStateAccept;
  NCCLCHECKGOTO(ncclSocketInit(&rComm->base.sock), ret, fail);
  NCCLCHECKGOTO(ncclSocketAccept(&rComm->base.sock, &lComm->sock), ret, fail);

  // Alloc stage->buffer here to be used for all following steps
  struct ncclIbConnectionMetadata remMeta;
  struct ncclIbDevExtraProps exProps;
  stage->offset = 0;
  NCCLCHECK(ncclIbMalloc((void**)&stage->buffer, sizeof(remMeta)));

ib_accept_check:
  NCCLCHECKGOTO(ncclSocketReady(&rComm->base.sock, &ready), ret, fail);
  if (!ready) return ncclSuccess;
  stage->state = ncclIbCommStateRecvDevList;
  stage->offset = 0;

// In the case of mismatched nDevs, we will make sure that both sides of a logical connection have the same number of RC qps
ib_recv_dev_list:
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &rComm->base.sock, stage->buffer, sizeof(ncclNetVDeviceProps_t) + sizeof(struct ncclIbDevExtraProps), &stage->offset));
  if (stage->offset != (sizeof(ncclNetVDeviceProps_t) + sizeof(struct ncclIbDevExtraProps))) return ncclSuccess;
  ncclNetVDeviceProps_t remoteVProps;
  memcpy(&remoteVProps, stage->buffer, sizeof(ncclNetVDeviceProps_t));
  if (lComm->dev >= ncclNMergedIbDevs) {
    WARN("NET/IB : Trying to use non-existent virtual device %d", lComm->dev);
    return ncclInternalError;
  }

  memcpy(&exProps, (char *)stage->buffer + sizeof(ncclNetVDeviceProps_t), sizeof(exProps));
  rComm->base.remOooRq = exProps.oooRq;

  // Reduce the physical device list and store in the connection base
  struct ncclIbMergedDev* mergedDev;
  mergedDev = ncclIbMergedDevs + lComm->dev;
  NCCLCHECK(ncclIbCheckVProps(&mergedDev->vProps, &remoteVProps));
  rComm->base.vProps = mergedDev->vProps;
  memcpy(stage->buffer, &rComm->base.vProps, sizeof(ncclNetVDeviceProps_t));
  int localNqps, remoteNqps;
  localNqps  = ncclParamIbQpsPerConn() * rComm->base.vProps.ndevs; // We must have at least 1 qp per-device
  remoteNqps = ncclParamIbQpsPerConn() * remoteVProps.ndevs;
  rComm->base.nqps = remoteNqps > localNqps ? remoteNqps : localNqps; // Select max nqps (local or remote)

  rComm->base.nDataQps = std::max(rComm->base.vProps.ndevs, remoteVProps.ndevs);

  if (rComm->base.resiliency) {
    NCCLCHECK(ncclIbResiliencyDeviceNumSet(rComm->base.resiliency, rComm->base.vProps.ndevs, remoteVProps.ndevs));
  }

  stage->offset = 0;
  stage->state = ncclIbCommStateSendDevList;

  exProps.oooRq = true;
  for (int i = 0; i < mergedDev->vProps.ndevs; i++) {
    int ibDevN = mergedDev->vProps.devs[i];
    exProps.oooRq = exProps.oooRq && ncclIbDevs[ibDevN].oooRqSize;
  }
  rComm->base.localOooRq = exProps.oooRq;
  memcpy((char *)stage->buffer + sizeof(ncclNetVDeviceProps_t), &exProps, sizeof(struct ncclIbDevExtraProps));

ib_send_dev_list:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &rComm->base.sock, stage->buffer, sizeof(ncclNetVDeviceProps_t) + sizeof(struct ncclIbDevExtraProps), &stage->offset), ret, fail);
  if (stage->offset != (sizeof(ncclNetVDeviceProps_t) + sizeof(struct ncclIbDevExtraProps))) return ncclSuccess;

  stage->offset = 0;
  stage->state = ncclIbCommStateRecv;

ib_recv:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_RECV, &rComm->base.sock, stage->buffer, sizeof(remMeta), &stage->offset), ret, fail);
  if (stage->offset != sizeof(remMeta)) return ncclSuccess;

  /* copy back the received info */
  memcpy(&remMeta, stage->buffer, sizeof(struct ncclIbConnectionMetadata));

  // IB setup
  // Pre-declare variables because of goto
  struct ncclIbDev* ibDev;
  int ibDevN;
  struct ncclIbRecvCommDev* rCommDev;

  mergedDev = ncclIbMergedDevs + lComm->dev;

  if (remMeta.ndevs != rComm->base.vProps.ndevs) {
    INFO(NCCL_NET, "NET/IB : Local mergedDev %s has a different number of devices=%d as remote %s %d",
      mergedDev->devName, rComm->base.vProps.ndevs, remMeta.devName, remMeta.ndevs);
  }

  // Metadata to send back to requestor (sender)
  struct ncclIbConnectionMetadata meta;
  memset(&meta, 0, sizeof(meta));
  // Receiver's CQ size needs to accomodate receive requests that can generate
  // up to 2 completions (one for the CTS message and one for the completion
  // of a receive request) per QP, in the worst case.
  int cqSize;
  cqSize = 2*NET_IB_MAX_REQUESTS*ncclParamIbQpsPerConn();
  for (int i = 0; i < rComm->base.vProps.ndevs; i++) {
    rCommDev = rComm->devs + i;
    ibDevN = rComm->base.vProps.devs[i];
    if (rComm->base.resiliency) {
      ncclIbResiliencyDataCqSizeGet(rComm->base.resiliency, i, &cqSize);
    }
    NCCLCHECKGOTO(ncclIbInitCommDevBase(ibDevN, &rCommDev->base, &rComm->base.stats, cqSize), ret, fail);
    if (rComm->base.resiliency) {
      NCCLCHECKGOTO(ncclIbResiliencyDevInit(rComm->base.resiliency, i, &ncclIbDevs[ibDevN]), ret, fail);
    }
    ibDev = ncclIbDevs + ibDevN;
    NCCLCHECKGOTO(ncclIbGetGidIndex(ibDev->context, ibDev->portNum, &ibDev->portAttr, &rCommDev->base.gidInfo.localGidIndex), ret, fail);
    NCCLCHECKGOTO(wrap_ibv_query_gid(ibDev->context, ibDev->portNum, rCommDev->base.gidInfo.localGidIndex, &rCommDev->base.gidInfo.localGid), ret, fail);
    if (link_layer == IBV_LINK_LAYER_UNSPECIFIED) link_layer = ibDev->portAttr.link_layer;
    if (link_layer != ibDev->portAttr.link_layer) {
      int ibDev0 = rComm->devs[0].base.ibDevN;
      WARN("NET/IB : Attempted to connect incompatible devices: [%d]%s:%d/%s and [%d]%s:%d/%s. Try selecting NICs of only one link type using NCCL_IB_HCA",
           ibDevN, ibDev->devName, ibDev->portNum, NCCL_IB_LLSTR(ibDev->portAttr.link_layer), ibDev0, ncclIbDevs[ibDev0].devName, ncclIbDevs[ibDev0].portNum, NCCL_IB_LLSTR(link_layer));
      return ncclInternalError;
    }
  }

  // Before assigning information about remote devices provided by the remote,
  // ensure that they are compatible with local devices
  for (int i = 0; i < remMeta.ndevs; i++) {
    if (remMeta.devs[i].link_layer != link_layer) {
      int ibDev0 = rComm->devs[0].base.ibDevN;
      WARN("NET/IB : Remote %s device is incompatible with the local [%d]%s:%d/%s. Try selecting NICs of only one link type using NCCL_IB_HCA",
           NCCL_IB_LLSTR(remMeta.devs[i].link_layer), ibDev0, ncclIbDevs[ibDev0].devName, ncclIbDevs[ibDev0].portNum, NCCL_IB_LLSTR(link_layer));
      return ncclInternalError;
    }
  }

  // Store the number of remote devices provided by the remote peer
  rComm->base.nRemDevs = remMeta.ndevs;

  // Store the remote GID information per-device provided by the remote peer
  for (int i = 0; i < rComm->base.nRemDevs; i++) {
    rComm->base.remDevs[i] = remMeta.devs[i];
    rComm->base.remDevs[i].remoteGid.global.interface_id  = rComm->base.remDevs[i].gid.global.interface_id;
    rComm->base.remDevs[i].remoteGid.global.subnet_prefix = rComm->base.remDevs[i].gid.global.subnet_prefix;
  }

  // Determine if Flush is enabled for this Comm. Must be done before creating
  // QPs. If Flush is enabled, extra QPs will be created for Flush operations.
  rComm->flushEnabled = ((ncclIbGdrSupport() == ncclSuccess || ncclIbDmaBufSupport(lComm->dev) == ncclSuccess)
                            && (ncclParamIbGdrFlushDisable() == 0)) ? 1 : 0;

  NCCLCHECKGOTO(ncclIbReceiverQpsCreateToRts(rComm, &remMeta, &meta), ret, fail);
  if (rComm->prepostReceiveWorkRequests) {
    NCCLCHECKGOTO(ncclIbReceiverPrePostReceiveWorkRequests(rComm), ret, fail);
  }

  // Store the remote CTS FIFO info provided by the remote peer
  rComm->remCtsFifo.addr = remMeta.addr;
  for (int i = 0; i < rComm->base.nRemDevs; i++) {
    rComm->remCtsFifo.rkeys[i] = remMeta.devs[i].rkey;
  }

  for (int i = 0; i < rComm->base.vProps.ndevs; i++) {
    rCommDev = rComm->devs + i;

    NCCLCHECKGOTO(wrap_ibv_reg_mr(&rCommDev->ctsFifoMr, rCommDev->base.pd, &rComm->remCtsFifo.elems, sizeof(rComm->remCtsFifo.elems), IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_READ), ret, fail);
    rCommDev->sge.lkey = rCommDev->ctsFifoMr->lkey;

    // Register completion records
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&rCommDev->cmplsRecordsMr, rCommDev->base.pd, &rComm->cmplsRecords, sizeof(rComm->cmplsRecords), IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ), ret, fail);
    meta.devs[i].rkey = rCommDev->cmplsRecordsMr->rkey;

  }
  if (ncclParamIbUseInline()) rComm->remCtsFifo.flags = IBV_SEND_INLINE;

  for (int i = 0; i < rComm->base.vProps.ndevs; i++) {
    rCommDev = rComm->devs + i;
    ibDev = ncclIbDevs + rCommDev->base.ibDevN;

    // Allocate Flush dummy buffer for GPU Direct RDMA
    if (rComm->flushEnabled) {
      NCCLCHECKGOTO(wrap_ibv_reg_mr(&rCommDev->gpuFlush.hostMr, rCommDev->base.pd, &rComm->gpuFlushHostMem, sizeof(int), IBV_ACCESS_LOCAL_WRITE), ret, fail);
      rCommDev->gpuFlush.sge.addr = (uint64_t)&rComm->gpuFlushHostMem;
      rCommDev->gpuFlush.sge.length = 1;
      rCommDev->gpuFlush.sge.lkey = rCommDev->gpuFlush.hostMr->lkey;
    }

    // Fill Handle
    meta.devs[i].lid                            = ibDev->portAttr.lid;
    meta.devs[i].link_layer                     = rCommDev->base.gidInfo.link_layer = ibDev->portAttr.link_layer;
    meta.devs[i].ib_port                        = ibDev->portNum;
    meta.devs[i].gid.global.subnet_prefix       = rCommDev->base.gidInfo.localGid.global.subnet_prefix;
    meta.devs[i].gid.global.interface_id        = rCommDev->base.gidInfo.localGid.global.interface_id;
    meta.devs[i].mtu                            = ibDev->portAttr.active_mtu;
  }
  meta.addr = (uint64_t)rComm->cmplsRecords;
  meta.sl = remMeta.sl;
  meta.tc = remMeta.tc;

  meta.ndevs = rComm->base.vProps.ndevs;
  strncpy(meta.devName, mergedDev->devName, MAX_MERGED_DEV_NAME);

  stage->state = ncclIbCommStateSend;
  stage->offset = 0;
  if (stage->buffer) {
    free(stage->buffer);
    stage->buffer = NULL;
  }
  NCCLCHECKGOTO(ncclIbMalloc((void**)&stage->buffer, sizeof(struct ncclIbConnectionMetadata)), ret, fail);
  memcpy(stage->buffer, &meta, sizeof(struct ncclIbConnectionMetadata));

ib_send:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &rComm->base.sock, stage->buffer, sizeof(struct ncclIbConnectionMetadata), &stage->offset), ret, fail);
  if (stage->offset < sizeof(struct ncclIbConnectionMetadata)) return ncclSuccess;

  stage->offset = 0;
  stage->state = ncclIbCommStatePendingReady;

ib_recv_ready:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_RECV,  &rComm->base.sock, &rComm->base.ready, sizeof(int), &stage->offset), ret, fail);
  if (stage->offset != sizeof(int)) return ncclSuccess;

  *recvComm = rComm;
exit:
  /* reset lComm stage */
  if (stage->buffer) free(stage->buffer);
  free(stage);
  lComm->stage = NULL;
  return ret;
fail:
  free(rComm);
  goto exit;
}

ncclResult_t ncclIbCloseSend(void* sendComm) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
  if (comm) {
    NCCLCHECK(ncclSocketClose(&comm->base.sock));

    for (int q = 0; q < comm->base.nqps; q++)
      if (comm->base.qps[q].qp != NULL) NCCLCHECK(wrap_ibv_destroy_qp(comm->base.qps[q].qp));

    if (comm->base.resiliency) {
      NCCLCHECK(ncclIbResiliencyClose(comm->base.resiliency));
    }

    for (int i = 0; i < comm->base.vProps.ndevs; i++) {
      struct ncclIbSendCommDev* commDev = comm->devs + i;
      if (commDev->ctsFifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->ctsFifoMr));
      if (commDev->cmplsRecordsMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->cmplsRecordsMr));
      if (commDev->putSignalScratchpadMr != NULL)
        NCCLCHECK(wrap_ibv_dereg_mr(commDev->putSignalScratchpadMr));
      if (comm->base.resiliency) {
         NCCLCHECK(ncclIbResiliencyDevDestroy(comm->base.resiliency, i));
      }
      NCCLCHECK(ncclIbDestroyBase(&commDev->base));
    }
    if (comm->base.resiliency) {
      NCCLCHECK(ncclIbResiliencyDestroy(&comm->base.resiliency));
    }
    free(comm);
  }
  TIME_PRINT("IB");
  return ncclSuccess;
}

ncclResult_t ncclIbCloseRecv(void* recvComm) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  if (comm) {
    NCCLCHECK(ncclSocketClose(&comm->base.sock));

    for (int q = 0; q < comm->base.nqps; q++)
      if (comm->base.qps[q].qp != NULL) NCCLCHECK(wrap_ibv_destroy_qp(comm->base.qps[q].qp));

    if (comm->base.resiliency) {
      NCCLCHECK(ncclIbResiliencyClose(comm->base.resiliency));
    }

    for (int i = 0; i < comm->base.vProps.ndevs; i++) {
      struct ncclIbRecvCommDev* commDev = comm->devs + i;
      if (comm->flushEnabled) {
        if (commDev->gpuFlush.qp.qp != NULL) NCCLCHECK(wrap_ibv_destroy_qp(commDev->gpuFlush.qp.qp));
        if (commDev->gpuFlush.hostMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->gpuFlush.hostMr));
      }
      if (commDev->ctsFifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->ctsFifoMr));
      if (commDev->cmplsRecordsMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->cmplsRecordsMr));
      if (comm->base.resiliency) {
        ncclIbResiliencyDevDestroy(comm->base.resiliency, i);
      }
      NCCLCHECK(ncclIbDestroyBase(&commDev->base));
    }
    if (comm->base.resiliency) {
      NCCLCHECK(ncclIbResiliencyDestroy(&comm->base.resiliency));
    }
    free(comm);
  }
  return ncclSuccess;
}

ncclResult_t ncclIbCloseListen(void* listenComm) {
  struct ncclIbListenComm* comm = (struct ncclIbListenComm*)listenComm;
  if (comm) {
    NCCLCHECK(ncclSocketClose(&comm->sock));
    free(comm);
  }
  return ncclSuccess;
}
