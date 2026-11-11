// SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <vector>

#include "runtime/config.h"

namespace tide {

// Copy-engine descriptors for one cudaMemcpyBatchAsync call.
struct CopyBatch {
  std::vector<void*> dst, src;
  std::vector<size_t> len;
  void add(void* d, const void* s, size_t n) {
    dst.push_back(d);
    src.push_back(const_cast<void*>(s));
    len.push_back(n);
  }
};

// Strided descriptors for one cudaMemcpy3DBatchAsync call: `rows` rows of `width` bytes, each
// operand with its own pitch. One covers the pieces of consecutive ranks where the flat batch
// needs a descriptor per rank.
struct StridedBatch {
  std::vector<cudaMemcpy3DBatchOp> ops;
  void add(void* dst, size_t dstPitch, const void* src, size_t srcPitch, size_t width, size_t rows) {
    cudaMemcpy3DBatchOp op{};
    op.src.type = op.dst.type = cudaMemcpyOperandTypePointer;
    op.src.op.ptr.ptr = const_cast<void*>(src);
    op.src.op.ptr.rowLength = srcPitch;
    op.dst.op.ptr.ptr = dst;
    op.dst.op.ptr.rowLength = dstPitch;
    op.extent = {width, rows, 1};
    op.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    op.flags = cudaMemcpyFlagPreferOverlapWithCompute;
    ops.push_back(op);
  }
};

// Rows from which a run of consecutive ranks is one strided descriptor. A flat descriptor costs
// the host 0.75 us and the batch spreads over the copy engines (501 GB/s at 1 MiB ops); a strided
// op 1.7 us whatever its row count and runs one op at a time (1 row 196 GB/s, 2: 314, 3: 391,
// 4: 442, 8: 559) -- one 8-GPU node, CUDA 13.2. TIDE_STRIDED_ROWS
// overrides (0: flat only).
inline size_t stridedRows() { return envConfig().stridedRows; }

// The descriptors of one push: flat and strided, issued as two batches on one stream.
struct PushBatch {
  CopyBatch flat;
  StridedBatch strided;
  void clear() {
    flat.dst.clear();
    flat.src.clear();
    flat.len.clear();
    strided.ops.clear();
  }
  size_t size() const { return flat.len.size() + strided.ops.size(); }
  // `rows` rows of `width` bytes, row r at dst + r*dstPitch from src + r*srcPitch.
  void addRows(char* dst, size_t dstPitch, const char* src, size_t srcPitch, size_t width, size_t rows) {
    if (rows >= stridedRows()) strided.add(dst, dstPitch, src, srcPitch, width, rows);
    else
      for (size_t r = 0; r < rows; ++r) flat.add(dst + r * dstPitch, src + r * srcPitch, width);
  }
};

}  // namespace tide
