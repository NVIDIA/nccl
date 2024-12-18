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
#define TRANSPORT_UNDEFINED -1
#define TRANSPORT_P2P 0
#define TRANSPORT_SHM 1
#define TRANSPORT_NET 2
#define TRANSPORT_COLLNET 3

#include "proxy.h"
#include "comm.h"

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
  // MNNVL support
  nvmlGpuFabricInfoV_t fabricInfo;
  int cuMemSupport;
  int version;
};

#define CONNECT_SIZE 256
struct ncclConnect {
  char data[CONNECT_SIZE];
};

#if CUDART_VERSION >= 12010

#define NVLS_HANDLE_SIZE 64
struct ncclNvlsSharedRes {
  int refCount;
  bool inited;
  CUmulticastObjectProp bufProp;
  CUmulticastObjectProp signalProp;
  CUmemAccessDesc accessDesc;
  int dev;
  size_t buffSize;
  size_t creditSize;
  CUmemGenericAllocationHandle mcBuffHandle; // Multicast handle for NVLS buffer
  CUmemGenericAllocationHandle mcCreditHandle; // Multicast handle for NVLS credit buffer
  char* mcBuff; // Multicast NVLS buffer address
  char* mcCredit; // Multicast NVLS credit address
  CUmemGenericAllocationHandle ucBuffHandle; // Unicast Handle for NVLS buffer
  CUmemGenericAllocationHandle ucCreditHandle; // Unicast Handle for NVLS credit buffer
  char* ucBuff; // Unicast NVLS buffer address
  char* ucCredit; // Unicast NVLS credit address
  int nChannels;
  struct ncclShmemCollBuff nvlsShmem;
  void *nvlsShmemHandle;
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
  ncclResult_t (*proxyRegister)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done);
  ncclResult_t (*proxyDeregister)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, int* done);
};

struct ncclTransport {
  const char name[8];
  ncclResult_t (*canConnect)(int*, struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo*, struct ncclPeerInfo*);
  struct ncclTransportComm send;
  struct ncclTransportComm recv;
};

ncclResult_t ncclTransportP2pConnect(struct ncclComm* comm, int channelId, int nrecv, int* peerRecv, int nsend, int* peerSend, int connIndex);
ncclResult_t ncclTransportP2pSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, int connIndex);
ncclResult_t ncclTransportCheckP2pType(struct ncclComm* comm, bool* intraNodeP2pSupport, bool* directMode);

ncclResult_t ncclNvlsInit(struct ncclComm* comm);
ncclResult_t ncclNvlsSetup(struct ncclComm* comm, struct ncclComm* parent);
ncclResult_t ncclNvlsBufferSetup(struct ncclComm* comm);
ncclResult_t ncclNvlsTreeConnect(struct ncclComm* comm);
ncclResult_t ncclNvlsGraphRegisterBuffer(struct ncclComm *comm, const void *sendbuff, void *recvbuff, size_t sendbuffSize, size_t recvbuffSize, int *outRegBufUsed, void **outRegBufSend, void **outRegBufRecv, struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, int* nCleanupQueueElts);
ncclResult_t ncclNvlsLocalRegisterBuffer(struct ncclComm *comm, const void *sendbuff, void *recvbuff, size_t sendbuffSize, size_t recvbuffSize, int *outRegBufUsed, void **outRegBufSend, void **outRegBufRecv);
ncclResult_t ncclNvlsDeregBuffer(struct ncclComm* comm, CUmemGenericAllocationHandle *mcHandler, CUdeviceptr ptr, int dev, size_t size);
ncclResult_t ncclNvlsFree(struct ncclComm* comm);

enum { collNetRecv=0, collNetSend=1 };
bool ncclTransportCollNetSetup(struct ncclComm* comm, struct ncclTopoGraph* collNetGraph, struct ncclChannel* channel, int masterRank, int masterPeer, int collNetGraphChannelId, int type, ncclConnect* connect);
ncclResult_t ncclTransportCollNetCheck(struct ncclComm* comm, int collNetSetupFail);
ncclResult_t ncclTransportCollNetFree(struct ncclComm* comm);
ncclResult_t ncclCollnetLocalRegisterBuffer(struct ncclComm* comm, const void* userbuff, size_t buffSize, int type, int* outRegBufUsed, void** outHandle);
ncclResult_t ncclCollnetGraphRegisterBuffer(struct ncclComm* comm, const void* userbuff, size_t buffSize, int type, int* outRegBufFlag, void** outHandle, struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, int* nCleanupQueueElts);
ncclResult_t ncclCollnetDeregBuffer(struct ncclComm* comm, struct ncclProxyConnector* proxyconn, void* handle);

ncclResult_t ncclTransportRingConnect(struct ncclComm* comm);
ncclResult_t ncclTransportTreeConnect(struct ncclComm* comm);
ncclResult_t ncclTransportPatConnect(struct ncclComm* comm);

ncclResult_t ncclCollNetSetup(ncclComm_t comm, ncclComm_t parent, struct ncclTopoGraph* graphs[]);
ncclResult_t ncclCollNetChainBufferSetup(ncclComm_t comm);
ncclResult_t ncclCollNetDirectBufferSetup(ncclComm_t comm);

ncclResult_t ncclNetDeregBuffer(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, void* handle);
ncclResult_t ncclNetLocalRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, struct ncclConnector** peerConns, int nPeers, int* outRegBufFlag, void** outHandle);
ncclResult_t ncclNetGraphRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, struct ncclConnector** peerConns, int nPeers, int* outRegBufFlag, void** outHandle, struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, int* nCleanupQueueElts);

ncclResult_t ncclRegisterP2pIpcBuffer(struct ncclComm* comm, void* userbuff, size_t size, int peerRank, int* regFlag, void** regAddr, struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue);
ncclResult_t ncclRegisterP2pNetBuffer(struct ncclComm* comm, void* userbuff, size_t size, struct ncclConnector* conn, int* regFlag, void** handle, struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue);
ncclResult_t ncclRegisterCollBuffers(struct ncclComm* comm, struct ncclTaskColl* info, void* outRegBufSend[NCCL_MAX_LOCAL_RANKS], void* outRegBufRecv[NCCL_MAX_LOCAL_RANKS], struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, bool* regNeedConnect);
ncclResult_t ncclRegisterCollNvlsBuffers(struct ncclComm* comm, struct ncclTaskColl* info, void* outRegBufSend[NCCL_MAX_LOCAL_RANKS], void* outRegBufRecv[NCCL_MAX_LOCAL_RANKS], struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, bool* regNeedConnect);

#endif
