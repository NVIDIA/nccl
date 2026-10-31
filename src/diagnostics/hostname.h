/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DIAGNOSTICS_HOSTNAME_H_
#define NCCL_DIAGNOSTICS_HOSTNAME_H_

#include <algorithm>
#include <cstddef>

// True if the hostname in `host` is safe to put in a shell command and in logs.
// It must end with '\0' inside the array, not be empty, not start with '-' (so it
// cannot look like an option), and hold only ASCII letters, digits, '.' and '-'.
// Keep this C++14: NCCL builds with C++14 before CUDA 13.
template <std::size_t N>
bool ncclDiagIsSafeHostname(const char (&host)[N]) {
  const char* end = std::find(host, host + N, '\0');
  return end != host && end != host + N && host[0] != '-' &&
         std::all_of(host, end, [](unsigned char c) {
           return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '.' || c == '-';
         });
}

#endif // NCCL_DIAGNOSTICS_HOSTNAME_H_
