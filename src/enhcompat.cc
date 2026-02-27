/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/* Define weak symbols used to allow libnccl_static.a to work with older libcudart_static.a */

enum cudaError_t { cudaErrorStubLibrary = 34 };

extern "C" {

cudaError_t cudaStreamGetCaptureInfo_v2(...)         __attribute__((visibility("hidden"))) __attribute((weak));
cudaError_t cudaStreamGetCaptureInfo_v2(...)         { return cudaErrorStubLibrary; }

cudaError_t cudaUserObjectCreate(...)                __attribute__((visibility("hidden"))) __attribute((weak));
cudaError_t cudaUserObjectCreate(...)                { return cudaErrorStubLibrary; }

cudaError_t cudaGraphRetainUserObject(...)           __attribute__((visibility("hidden"))) __attribute((weak));
cudaError_t cudaGraphRetainUserObject(...)           { return cudaErrorStubLibrary; }

cudaError_t cudaStreamUpdateCaptureDependencies(...) __attribute__((visibility("hidden"))) __attribute((weak));
cudaError_t cudaStreamUpdateCaptureDependencies(...) { return cudaErrorStubLibrary; }

cudaError_t cudaGetDriverEntryPoint(...)             __attribute__((visibility("hidden"))) __attribute((weak));
cudaError_t cudaGetDriverEntryPoint(...)             { return cudaErrorStubLibrary; }

}
