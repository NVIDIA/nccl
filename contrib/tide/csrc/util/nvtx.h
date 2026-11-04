// SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// NVTX in a "TIDE" domain for Nsight Systems: host ranges, one color per phase (nsys projects them onto the GPU rows
// over the work they launched), and names for the internal streams. Without a tool attached a push / pop is one
// branch on a function pointer.
#pragma once

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCudaRt.h>

#include <cstdint>

namespace tide::nvtx {

enum Color : uint32_t {
  kAg = 0xFF4E79A7,      // all-gather issue and rounds
  kAgTail = 0xFFA0CBE8,  // all-gather landing join / done fence
  kSync = 0xFFE15759,    // credits, ready / done fences
  kRsRead = 0xFF59A14F,  // reduce-scatter pushes and level-1 sums
  kRsNet = 0xFFF28E2B,   // reduce-scatter network phase (host rounds or device wave kernels)
  kRsLand = 0xFFB07AA1,  // reduce-scatter wave landing
  kRsTail = 0xFFBAB0AC,  // reduce-scatter end-of-call work
};

inline nvtxDomainHandle_t domain() {
  static const nvtxDomainHandle_t d = nvtxDomainCreateA("TIDE");
  return d;
}

class Range {
 public:
  Range(const char* name, uint32_t color) {
    nvtxEventAttributes_t a{};
    a.version = NVTX_VERSION;
    a.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    a.colorType = NVTX_COLOR_ARGB;
    a.color = color;
    a.messageType = NVTX_MESSAGE_TYPE_ASCII;
    a.message.ascii = name;
    nvtxDomainRangePushEx(domain(), &a);
  }
  ~Range() { nvtxDomainRangePop(domain()); }
  Range(const Range&) = delete;
  Range& operator=(const Range&) = delete;
};

inline void nameStream(cudaStream_t s, const char* name) {
  if (s) nvtxNameCudaStreamA(s, name);
}

}  // namespace tide::nvtx
