/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TOPO_H_
#define NCCL_TOPO_H_

#include "nccl.h"
#include <limits.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>

ncclResult_t getCudaPath(int cudaDev, char** path);

static int getNumaId(char *path) {
  char npath[PATH_MAX];
  snprintf(npath, PATH_MAX, "%s/numa_node", path);
  npath[PATH_MAX-1] = '\0';

  int numaId = -1;
  FILE *file = fopen(npath, "r");
  if (file == NULL) return -1;
  if (fscanf(file, "%d", &numaId) == EOF) { fclose(file); return -1; }
  fclose(file);

  return numaId;
}

enum ncclPathDist {
  PATH_PIX  = 0,
  PATH_PXB  = 1,
  PATH_PHB  = 2,
  PATH_NODE = 3,
  PATH_SYS  = 4,
  PATH_ARRAY_SIZE = 5
};

extern const char* pathDists[PATH_ARRAY_SIZE];

int pciDistance(char* path1, char* path2);

#endif
