/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TRANSPORT_H_
#define NCCL_TRANSPORT_H_

#include "device.h"
#include "graph.h"
#include "nvmlwrap.h"
#include "core.h"

#define NTRANSPORTS 4
#define TRANSPORT_P2P 0
#define TRANSPORT_SHM 1
#define TRANSPORT_NET 2
#define TRANSPORT_COLLNET 3

#include "proxy.h"

extern struct ncclTransport p2pTransport;
extern struct ncclTransport shmTransport;
extern struct ncclTransport netTransport;
extern struct ncclTransport collNetTransport;

extern struct ncclTransport* ncclTransports[];

// Forward declarations
struct ncclRing;
struct ncclConnector;
struct ncclComm;

struct ncclPeerInfo {
  int rank;
  int cudaDev;
  int nvmlDev;
  int gdrSupport;
  uint64_t hostHash;
  uint64_t pidHash;
  dev_t shmDev;
  int64_t busId;
  struct ncclComm* comm;
  int cudaCompCap;
};

#define CONNECT_SIZE 128
struct ncclConnect {
  char data[CONNECT_SIZE];
};

#if CUDART_VERSION >= 12010

#define NVLS_HANDLE_SIZE 64
struct ncclNvlsSharedRes {
  int refCount;
  CUmulticastObjectProp properties;
  CUmemAccessDesc accessDesc;
  int dev;
  size_t size;
  size_t granularity;
  CUmemGenericAllocationHandle mcHandle; // Multicast handle for NVLS buffer
  char* mcBuff; // Multicast NVLS buffer address
  CUmemGenericAllocationHandle ucHandle; // Unicast Handle for NVLS buffer
  char* ucBuff; // Unicast NVLS buffer address
  char shareableHandle[NVLS_HANDLE_SIZE];
  size_t ucGran;
  int nChannels;
};

#endif /* CUDART_VERSION >= 12010 */

struct ncclCollNetSharedRes {
  int refCount;
  int size;
  char* cudaBuff;
  char* hostBuff;
  struct ncclProxyArgs* proxyAppend[2*NCCL_MAX_NETDEVS];
  void* resources;
  int nChannels;
  size_t buffSize;
};

struct ncclTransportComm {
  ncclResult_t (*setup)(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo*, struct ncclPeerInfo*, struct ncclConnect*, struct ncclConnector*, int channelId, int connIndex);
  ncclResult_t (*connect)(struct ncclComm* comm, struct ncclConnect*, int nranks, int rank, struct ncclConnector*);
  ncclResult_t (*free)(struct ncclConnector*);
  ncclResult_t (*proxySharedInit)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, int nChannels);
  ncclResult_t (*proxySetup)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done);
  ncclResult_t (*proxyConnect)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done);
  ncclResult_t (*proxyFree)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState);
  ncclResult_t (*proxyProgress)(struct ncclProxyState* proxyState, struct ncclProxyArgs*);
};

struct ncclTransport {
  const char name[8];
  ncclResult_t (*canConnect)(int*, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo*, struct ncclPeerInfo*);
  struct ncclTransportComm send;
  struct ncclTransportComm recv;
};

ncclResult_t ncclTransportP2pConnect(struct ncclComm* comm, int channelId, int nrecv, int* peerRecv, int nsend, int* peerSend, int connIndex);
ncclResult_t ncclTransportP2pSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, int connIndex, int* highestTransportType=NULL);

// Currently we only support POSIX_FILE_DESCRIPTOR handle exchange
#define USE_POSIX_FD 1

#if USE_POSIX_FD
#define NVLS_CU_MEM_HANDLE_TYPE CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
#else
#define NVLS_CU_MEM_HANDLE_TYPE CU_MEM_HANDLE_TYPE_NONE
#endif

ncclResult_t ncclNvlsInit(struct ncclComm* comm);
ncclResult_t ncclNvlsSetup(struct ncclComm* comm, struct ncclComm* parent);
ncclResult_t ncclNvlsGraphRegisterBuffer(struct ncclComm *comm, struct ncclKernelPlan *plan, const void *sendbuff, void *recvbuff, size_t sendbuffSize, size_t recvbuffSize, bool *outRegBufUsed, void **outRegBufSend, void **outRegBufRecv);
ncclResult_t ncclNvlsLocalRegisterBuffer(struct ncclComm *comm, const void *sendbuff, void *recvbuff, size_t sendbuffSize, size_t recvbuffSize, bool *outRegBufUsed, void **outRegBufSend, void **outRegBufRecv);
ncclResult_t ncclNvlsDeregBuffer(CUmemGenericAllocationHandle *mcHandler, CUdeviceptr ptr, int dev, size_t size);
ncclResult_t ncclNvlsFree(struct ncclComm* comm);

enum { collNetRecv=0, collNetSend=1 };
int ncclTransportCollNetSetup(struct ncclComm* comm, struct ncclTopoGraph* collNetGraph, struct ncclChannel* channel, int masterRank, int masterPeer, int collNetGraphChannelId, int type);
ncclResult_t ncclTransportCollNetCheck(struct ncclComm* comm, int collNetSetupFail);
ncclResult_t ncclTransportCollNetFree(struct ncclComm* comm);
#endif
