/*************************************************************************
 * Copyright (c) 2018-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_CPUSET_H_
#define NCCL_CPUSET_H_

#include "nccl.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifdef NCCL_OS_LINUX
#include <sched.h>
#elif defined(NCCL_OS_WINDOWS)
#define CPU_SETSIZE 64 // Windows uses DWORD_PTR for affinity
#endif

// Convert local_cpus, e.g. 0003ff,f0003fff to ncclAffinity.
// The bitmask is divided into chunks of 32 bits, each of them represented by 8 hex number.
#define U32_LEN 32 // using uint32_t
#define CPU_SET_N_U32 (CPU_SETSIZE / U32_LEN)

static ncclResult_t ncclStrToCpuset(const char* maskStr, ncclAffinity* set) {
  uint32_t cpumasks[CPU_SET_N_U32] = {0};

  // transform the string into an array of 32 bit masks, starting with the highest mask
  int m = CPU_SET_N_U32;
  char* str = strdup(maskStr);
  char* token = strtok(str, ",");
  while (token != NULL && m > 0) {
    cpumasks[--m] = strtoul(token, NULL, /*base = hex*/ 16);
    token = strtok(NULL, ",");
  }
  free(str);

  // list all the CPUs as part of the CPU set, starting with the lowest mask (= current value of m)
  ncclOsCpuZero(*set);
  for (int a = 0; (a + m) < CPU_SET_N_U32; a++) {
    // each mask is U32_LEN CPUs, list them all if the bit is on
    for (int i = 0; i < U32_LEN; ++i) {
      if (cpumasks[a + m] & (1UL << i)) ncclOsCpuSet(*set, i + a * U32_LEN);
    }
  }
  return ncclSuccess;
}

static char* ncclCpusetToRangeStr(ncclAffinity* mask, char* str, size_t len) {
  int c = 0;
  int start = -1;
  // Iterate through all possible CPU bits plus one extra position
  for (int cpu = 0; cpu <= CPU_SETSIZE; cpu++) {
    int isSet = (cpu == CPU_SETSIZE) ? 0 : ncclOsCpuIsSet(*mask, cpu);
    // Start of a new range
    if (isSet && start == -1) {
      start = cpu;
    }
    // End of a range, add comma between ranges
    if (!isSet && start != -1) {
      if (cpu-1 == start) {
        c += snprintf(str+c, len-c, "%s%d", c ? "," : "", start);
      } else {
        c += snprintf(str+c, len-c, "%s%d-%d", c ? "," : "", start, cpu-1);
      }
      if (c >= len-1) break;
      start = -1;
    }
  }
  if (c == 0) str[0] = '\0';
  return str;
}

static ncclResult_t ncclStrListToCpuset(const char* userStr, ncclAffinity* mask) {
  // reset the CPU set
  ncclOsCpuZero(*mask);
  const char delim[] = ",";
  char* str = strdup(userStr);
  char* token = strtok(str, delim);
  while (token != NULL) {
    uint64_t cpu = strtoull(token, NULL, 0);
    ncclOsCpuSet(*mask, cpu);
    token = strtok(NULL, delim);
  }
  free(str);
  return ncclSuccess;
}

static ncclResult_t ncclCpusetToStrList(ncclAffinity* mask, char* str, size_t len) {
  if (len == 0) return ncclSuccess;
  str[0] = '\0';
  int count = 0;
  for (uint64_t id = 0; id < CPU_SETSIZE; ++id) {
    if (ncclOsCpuIsSet(*mask, id)) {
      snprintf(str + strlen(str), len - strlen(str), "%s%lu", (count++ == 0) ? "" : ",", id);
    }
  }
  return ncclSuccess;
}

#endif
