/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "cuda_runtime.h"
#include "nccl.h"
#include "param/c_api.h"

int main(int argc, char* argv[]) {
  bool longFormat = false;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-l") == 0) {
      longFormat = true;
    } else {
      fprintf(stderr, "Usage: %s [-l]\n", argv[0]);
      fprintf(stderr, "  -l  long format (show values and descriptions)\n");
      return EXIT_FAILURE;
    }
  }

  bool initialized = false;
  ncclComm_t comm;
  int dev = 0;

  cudaError_t ce = cudaSetDevice(dev);
  if (ce != cudaSuccess) {
    fprintf(stderr, "WARNING: cudaSetDevice failed (%s). "
            "Parameters from plugins may not be available.\n",
            cudaGetErrorString(ce));
  } else {
    ncclResult_t nr = ncclCommInitAll(&comm, 1, &dev);
    if (nr != ncclSuccess) {
      fprintf(stderr, "WARNING: ncclCommInitAll failed (%s). "
              "Parameters from plugins may not be available.\n",
              ncclGetErrorString(nr));
    } else {
      initialized = true;
    }
  }

  if (longFormat) {
    ncclParamDumpAll();
  } else {
    const char** keys = nullptr;
    int nkeys = 0;
    ncclParamGetAllParameterKeys(&keys, &nkeys);
    for (int i = 0; i < nkeys; i++) {
      printf("%s\n", keys[i]);
    }
  }

  if (initialized) {
    ncclCommDestroy(comm);
  }
  return 0;
}
