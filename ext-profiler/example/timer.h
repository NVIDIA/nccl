/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TIMER_H_
#define NCCL_TIMER_H_

#include <unistd.h>
#include <sys/time.h>
#include <x86intrin.h>
static double freq = -1;
static void calibrate() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  uint64_t timeCycles = __rdtsc();
  double time = - tv.tv_sec*1E6 - tv.tv_usec;
  uint64_t total = 0ULL;
  for (int i=0; i<10000; i++) total += __rdtsc();
  gettimeofday(&tv, NULL);
  timeCycles = __rdtsc() - timeCycles;
  time += tv.tv_sec*1E6 + tv.tv_usec;
  freq = timeCycles/time;
}
static inline double gettime() {
  if (freq == -1) calibrate();
  return __rdtsc()/freq;
}

#endif
