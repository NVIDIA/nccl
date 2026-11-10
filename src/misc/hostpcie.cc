/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "hostpcie.h"
#include "debug.h"

#include <dirent.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Read one numeric sysfs attribute of a PCI device directory.
// "32.0 GT/s PCIe" -> 32.0 ; "16" -> 16 ; "Unknown" -> failure.
static ncclResult_t ncclReadPciAttr(const char* devDir, const char* attr, double* value) {
  char path[PATH_MAX];
  char buf[64];
  snprintf(path, sizeof(path), "%s/%s", devDir, attr);
  FILE* f = fopen(path, "r");
  if (f == NULL) return ncclSystemError;
  char* line = fgets(buf, sizeof(buf), f);
  fclose(f);
  if (line == NULL) return ncclSystemError;
  double v = atof(buf);
  if (!(v > 0.0)) return ncclSystemError;
  *value = v;
  return ncclSuccess;
}

// The figures are raw: the per-lane transfer rate times the lane count, with no encoding
// overhead removed, which is how a device counts them. Both are returned because devices
// up to CX7 read the host PCIe state from their own PCIe registers, which describe their
// own link alone, while later ones are told about the whole path.
//
// Fails if the path cannot be walked, and also if any link on it is running below what
// NCCL's own PCIe model assumes. That model is built from max_link_speed and
// max_link_width, each minimised against the parent port (ncclTopoGetXmlFromSys() in
// src/graph/xml.cc), so on a link that trained down further the model is optimistic and
// the value the device reports is the only thing accounting for the difference.
ncclResult_t ncclHostPciePath(const char* sysDir, const char* label, uint64_t* ownLink, uint64_t* pathMin) {
  if (sysDir == NULL) return ncclSystemError;
  char* dir = realpath(sysDir, NULL);
  if (dir == NULL) return ncclSystemError;

  ncclResult_t ret = ncclSuccess;
  *ownLink = *pathMin = 0;
  // Walk up towards the root complex. Every PCI device and bridge on the way exposes
  // current_link_speed / current_link_width; the host bridge above them does not.
  while (1) {
    char* slash = strrchr(dir, '/');
    if (slash == NULL || slash == dir) break;
    // Stop once we leave the PCI devices: a bus:device.function component always has ':'.
    if (strchr(slash + 1, ':') == NULL) break;

    double gts = 0.0, width = 0.0, maxGts = 0.0, maxWidth = 0.0, parent = 0.0;
    if (ncclReadPciAttr(dir, "current_link_speed", &gts) == ncclSuccess &&
        ncclReadPciAttr(dir, "current_link_width", &width) == ncclSuccess) {
      if (ncclReadPciAttr(dir, "max_link_speed", &maxGts) != ncclSuccess) maxGts = gts;
      if (ncclReadPciAttr(dir, "max_link_width", &maxWidth) != ncclSuccess) maxWidth = width;
      if (ncclReadPciAttr(dir, "../max_link_speed", &parent) == ncclSuccess && parent < maxGts) maxGts = parent;
      if (ncclReadPciAttr(dir, "../max_link_width", &parent) == ncclSuccess && parent < maxWidth) maxWidth = parent;
      if (gts < 0.99 * maxGts || width < 0.99 * maxWidth) {
        INFO(NCCL_INIT | NCCL_NET,
             "NET : %s sits behind PCIe link %s running at %g GT/s x%g where %g GT/s x%g is advertised; keeping the "
             "speed reported by the device, which accounts for it.",
             label, slash + 1, gts, width, maxGts, maxWidth);
        ret = ncclSystemError;
        break;
      }
      uint64_t mbps = (uint64_t)(gts * width * 1000.0);
      if (*ownLink == 0) *ownLink = mbps;
      if (*pathMin == 0 || mbps < *pathMin) *pathMin = mbps;
    }
    *slash = '\0'; // move to the parent device
  }

  free(dir);
  if (ret == ncclSuccess && *pathMin == 0) ret = ncclSystemError;
  return ret;
}

// 1% of slack absorbs rounding differences between the device and sysfs.
bool ncclBwMatches(uint64_t value, uint64_t reference) {
  return reference > 0 && (double)value >= 0.99 * (double)reference && (double)value <= 1.01 * (double)reference;
}

// The file holds a line such as "800 Gb/sec (4X NDR)". Only the number and its unit are
// used; the parenthesised part names the width and the signalling rate separately and is
// already accounted for in the number.
static ncclResult_t ncclReadRateFile(const char* path, uint64_t* mbps) {
  char buf[128];
  FILE* f = fopen(path, "r");
  if (f == NULL) return ncclSystemError;
  char* line = fgets(buf, sizeof(buf), f);
  fclose(f);
  if (line == NULL) return ncclSystemError;
  double v = atof(buf);
  if (!(v > 0.0)) return ncclSystemError;
  if (strstr(buf, "Gb") != NULL) v *= 1000.0;
  else if (strstr(buf, "Tb") != NULL) v *= 1000000.0;
  else if (strstr(buf, "Mb") == NULL) return ncclSystemError;
  *mbps = (uint64_t)v;
  return ncclSuccess;
}

ncclResult_t ncclNetSysfsPortRate(const char* pciPath, int port, uint64_t* mbps) {
  if (pciPath == NULL) return ncclSystemError;
  char ibDir[PATH_MAX];
  snprintf(ibDir, sizeof(ibDir), "%s/infiniband", pciPath);
  DIR* d = opendir(ibDir);
  if (d == NULL) return ncclSystemError;

  // A PCI function carries one RDMA device in all the cases this matters for, but the
  // directory is read rather than guessed, and the first port that answers is taken.
  ncclResult_t ret = ncclSystemError;
  struct dirent* entry;
  while (ret != ncclSuccess && (entry = readdir(d)) != NULL) {
    if (entry->d_name[0] == '.') continue;
    char ratePath[PATH_MAX];
    int n = snprintf(ratePath, sizeof(ratePath), "%s/%s/ports/%d/rate", ibDir, entry->d_name, port > 0 ? port : 1);
    if (n < 0 || n >= (int)sizeof(ratePath)) continue;
    ret = ncclReadRateFile(ratePath, mbps);
  }
  closedir(d);
  return ret;
}

// A net plugin reports the bandwidth available to this host function, that is
// min(network port rate, host PCIe link, host rate limit). The host PCIe term does not
// belong in a NIC speed. NCCL derives the bandwidth of every PCIe link from sysfs in
// ncclTopoAddPci() and applies it to the paths that traverse that link, so a NET node
// carrying it as well counts it twice; and where a GPU and its NIC sit under a common
// PCIe bridge, GPUDirect RDMA traffic never crosses the host link at all, so the
// GPU-to-NIC path ends up sized by a limit that is not on it.
//
// This is the same correction ncclIbPortSpeed() makes for the built-in IB transport, at
// the one place where the figure a plugin reports enters NCCL's model. It is applied here
// as well because the double counting belongs to NCCL, not to the plugin: a plugin that
// carries its own copy of the IB transport has no way of knowing that NCCL models PCIe
// separately.
//
// The host PCIe term is recognised by its value, since it is the bandwidth of one of the
// PCIe links between the NIC and the root complex, and only then is the network port rate
// used in its place. Everything else is kept as reported, so a rate limit on the function,
// which NCCL has no other source for, survives. A rate limit set to exactly the bandwidth
// of one of those links cannot be told apart and is dropped; NCCL_IB_QUERY_PORT_SPEED=1
// covers that case.
int ncclNetAdjustSpeed(const char* name, const char* pciPath, int port, int speed) {
  if (ncclParamIbQueryPortSpeed() > 0) return speed;
  if (speed <= 0 || pciPath == NULL) return speed;

  uint64_t portRate = 0;
  if (ncclNetSysfsPortRate(pciPath, port, &portRate) != ncclSuccess) return speed;
  // Nothing is capping the port, so there is nothing to attribute. This is also the case
  // once the built-in IB transport has already made the same correction.
  if (portRate == 0 || (uint64_t)speed >= portRate) return speed;

  uint64_t ownLink = 0, pathMin = 0;
  if (ncclHostPciePath(pciPath, name, &ownLink, &pathMin) != ncclSuccess) return speed;
  if (!ncclBwMatches((uint64_t)speed, ownLink) && !ncclBwMatches((uint64_t)speed, pathMin)) return speed;

  INFO(NCCL_INIT | NCCL_NET | NCCL_GRAPH,
       "TOPO/NET : %s is reported at %d Mbps, the bandwidth of its host PCIe link (own link %lu, path %lu); using the "
       "%lu Mbps network port rate instead, since NCCL models PCIe separately. Set NCCL_IB_QUERY_PORT_SPEED=1 to keep "
       "the reported value.",
       name, speed, (unsigned long)ownLink, (unsigned long)pathMin, (unsigned long)portRate);
  return (int)portRate;
}
