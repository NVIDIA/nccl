/*************************************************************************
 * Copyright (c) 2016-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "profiler.h"
#include "proxy.h"

NCCL_PARAM(ProxyTraceNvtx, "PROXY_NVTX_ENABLE", 0);

thread_local char buffer[1024];

void ncclProxyInitNvtx(struct ncclProxyState* proxyState) {
  if (ncclParamProxyTraceNvtx() == 0) return;

  proxyState->nvtx.domain = nvtxDomainCreateA("com.nvidia.nccl.proxy");

  nvtxNameCategoryA(NVTX_CATEGORY_STATE, "Proxy Thread State");
  nvtxNameCategoryA(NVTX_CATEGORY_RECV,  "Proxy Recv Progress");
  nvtxNameCategoryA(NVTX_CATEGORY_SEND,  "Proxy Send Progress");

  nvtxEventAttributes_t sleep = {0};
  sleep.version = NVTX_VERSION; 
  sleep.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  sleep.colorType = NVTX_COLOR_ARGB;
  sleep.color = ARGB_BLUE;
  sleep.messageType = NVTX_MESSAGE_TYPE_ASCII; 
  sleep.message.ascii = "Proxy Sleep";

  sleep.category = NVTX_CATEGORY_STATE;
  proxyState->nvtx.sleep = sleep;

  proxyState->nvtx.active = sleep;
  proxyState->nvtx.active.color = ARGB_RED;
  proxyState->nvtx.active.message.ascii = "Proxy Active";

  proxyState->nvtx.append = sleep;
  proxyState->nvtx.append.color = ARGB_PURPLE;
  proxyState->nvtx.append.message.ascii = "Proxy Append";

  proxyState->nvtx.idle = sleep;
  proxyState->nvtx.idle.color = ARGB_PURPLE1;
  proxyState->nvtx.idle.message.ascii = "Proxy Idle";

  proxyState->nvtx.wakeup = sleep;
  proxyState->nvtx.wakeup.color = ARGB_PURPLE2;
  proxyState->nvtx.wakeup.message.ascii = "Proxy Wakeup";

  sleep.category = NVTX_CATEGORY_RECV;
  proxyState->nvtx.recvBegin = sleep;
  proxyState->nvtx.recvBegin.color = ARGB_GREEN;
  proxyState->nvtx.recvBegin.message.ascii = "Recv Begin";

  proxyState->nvtx.recvNetWait = sleep;
  proxyState->nvtx.recvNetWait.color = ARGB_GREEN1;
  proxyState->nvtx.recvNetWait.message.ascii = "Recv Net Wait";

  proxyState->nvtx.recvFlushWait = sleep;
  proxyState->nvtx.recvFlushWait.color = ARGB_GREEN2;
  proxyState->nvtx.recvFlushWait.message.ascii = "Recv Flush Wait";

  proxyState->nvtx.recvGpuWait = sleep;
  proxyState->nvtx.recvGpuWait.color = ARGB_GREEN3;
  proxyState->nvtx.recvGpuWait.message.ascii = "Recv GPU Wait";

  sleep.category = NVTX_CATEGORY_SEND;
  proxyState->nvtx.sendBegin = sleep;
  proxyState->nvtx.sendBegin.color = ARGB_YELLOW;
  proxyState->nvtx.sendBegin.message.ascii = "Send Begin";

  proxyState->nvtx.sendGpuWait = sleep;
  proxyState->nvtx.sendGpuWait.color = ARGB_YELLOW1;
  proxyState->nvtx.sendGpuWait.message.ascii = "Send GPU Wait";

  proxyState->nvtx.sendNetPost = sleep;
  proxyState->nvtx.sendNetPost.color = ARGB_YELLOW2;
  proxyState->nvtx.sendNetPost.message.ascii = "Send Net Post";

  proxyState->nvtx.sendNetWait = sleep;
  proxyState->nvtx.sendNetWait.color = ARGB_YELLOW3;
  proxyState->nvtx.sendNetWait.message.ascii = "Send Net Wait";

  // Set wakeup state
  proxyState->nvtx.rangeStateId = nvtxDomainRangeStartEx(proxyState->nvtx.domain, &proxyState->nvtx.wakeup);
}

// Event should be beginSend or beginRecv
void ncclProxySubArgsInitNvtx(struct ncclProxySubArgs* sub, uint64_t opCount, nvtxDomainHandle_t domain, nvtxEventAttributes_t* event) {
  if (ncclParamProxyTraceNvtx() == 0) return;
  sub->opRangeIds = (nvtxRangeId_t*) malloc(sizeof(nvtxRangeId_t)*sub->nsteps);
  for (uint64_t step=0; step<sub->nsteps; step++) {
    nvtxEventAttributes_t eventCopy = *event;
    snprintf(buffer, 1024, "%s o=%ld s=%ld nbytes=%ld", event->message.ascii, opCount, step+sub->base, sub->nbytes);
    eventCopy.message.ascii = buffer;
    sub->opRangeIds[step] = nvtxDomainRangeStartEx(domain, &eventCopy);
  }
}

void ncclProxySubArgsFreeNvtx(struct ncclProxyArgs* args) {
  if (ncclParamProxyTraceNvtx() == 0) return;
  for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      free(sub->opRangeIds);
  }
}

// Stop prior event, start next event (subarg proxy state)
void ncclProxySubArgsTraceNvtx(struct ncclProxySubArgs* sub, uint64_t opCount, uint64_t step, nvtxDomainHandle_t domain, nvtxEventAttributes_t* event, int size, uint32_t category) {
  if (ncclParamProxyTraceNvtx() == 0) return;
  nvtxRangeEnd(sub->opRangeIds[step]);
  nvtxEventAttributes_t eventCopy = *event;
  eventCopy.category = category;
  snprintf(buffer, 1024, "%s o=%ld s=%ld sz=%d", event->message.ascii, opCount, step, size);
  eventCopy.message.ascii = buffer;
  sub->opRangeIds[step] = nvtxDomainRangeStartEx(domain, &eventCopy);
}

void ncclProxySubArgsStopNvtx(struct ncclProxySubArgs* sub, uint64_t step) {
  if (ncclParamProxyTraceNvtx() == 0) return;
  nvtxRangeEnd(sub->opRangeIds[step]);
}

// Stop prior event, start next event (global proxy state)
void ncclProxyStateTraceNvtx(struct ncclProxyState* proxyState, nvtxEventAttributes_t* event) {
  if (ncclParamProxyTraceNvtx() == 0) return;
  nvtxRangeEnd(proxyState->nvtx.rangeStateId);
  proxyState->nvtx.rangeStateId = nvtxDomainRangeStartEx(proxyState->nvtx.domain, event);
}
