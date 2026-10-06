/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/* Provide fallback symbols for older libcudart_static libraries. Static-library
 * consumers use /INCLUDE to load this object and expose the aliases below. */

#pragma comment(linker, "/alternatename:cudaStreamGetCaptureInfo_v2=ncclFallback_cudaStreamGetCaptureInfo_v2")
#pragma comment(linker, "/alternatename:cudaUserObjectCreate=ncclFallback_cudaUserObjectCreate")
#pragma comment(linker, "/alternatename:cudaGraphRetainUserObject=ncclFallback_cudaGraphRetainUserObject")
#pragma comment(linker, \
                "/alternatename:cudaStreamUpdateCaptureDependencies=ncclFallback_cudaStreamUpdateCaptureDependencies")
#pragma comment(linker, "/alternatename:cudaGetDriverEntryPoint=ncclFallback_cudaGetDriverEntryPoint")

enum cudaError_t {
  cudaErrorStubLibrary = 34
};

extern "C" {

void ncclCudaEnhcompatAnchor() {}

cudaError_t ncclFallback_cudaStreamGetCaptureInfo_v2(...) {
  return cudaErrorStubLibrary;
}

cudaError_t ncclFallback_cudaUserObjectCreate(...) {
  return cudaErrorStubLibrary;
}

cudaError_t ncclFallback_cudaGraphRetainUserObject(...) {
  return cudaErrorStubLibrary;
}

cudaError_t ncclFallback_cudaStreamUpdateCaptureDependencies(...) {
  return cudaErrorStubLibrary;
}

cudaError_t ncclFallback_cudaGetDriverEntryPoint(...) {
  return cudaErrorStubLibrary;
}
}
