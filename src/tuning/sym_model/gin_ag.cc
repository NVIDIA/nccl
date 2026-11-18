/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "gin_ag.h"
#include "cudawrap.h"
#include "../../include/sym_partition.h"

#include <algorithm>
#include <cmath>

namespace {
constexpr uint64_t ginAgCellBytes = NCCL_SYM_KERNEL_CELL_SIZE;

struct ginAgPartition {
  uint64_t cells;
  uint64_t perCta;
  int activeCtas;
};

// Match the scheduler's partition into 1 KiB cells, including a partial last cell.
struct ginAgPartition ginAgPartitionForCtas(uint64_t rankBytes, int requestedCtas) {
  const uint64_t cells = ncclSymkDivUp(rankBytes, ginAgCellBytes);
  const uint64_t perCta = ncclSymkCellsPerChannel(rankBytes, requestedCtas, ginAgCellBytes);
  return {cells, perCta, int(ncclSymkDivUp(cells, perCta))};
}

// The scheduler rounds cell boundaries up to 16-bit fractions; work.cuh
// converts those fractions back to cells. There are at most 64 CTA spans.
int ginAgCtaSpans(uint64_t rankBytes, const struct ginAgPartition& partition, uint64_t* spans) {
  uint64_t previousEnd = 0;
  for (int cta = 1; cta <= partition.activeCtas; cta++) {
    const uint64_t cellEnd = std::min(cta * partition.perCta, partition.cells);
    const uint64_t fractionEnd = ncclSymkCellFraction(cellEnd, partition.cells);
    const uint64_t byteEnd = std::min(rankBytes, ncclSymkFractionCell(fractionEnd, partition.cells) * ginAgCellBytes);
    spans[cta - 1] = byteEnd - previousEnd;
    previousEnd = byteEnd;
  }
  // Only distinct lengths are needed to predict the slowest CTA.
  std::sort(spans, spans + partition.activeCtas);
  return std::unique(spans, spans + partition.activeCtas) - spans;
}

// Bandwidths are GB/s and latencies are microseconds.
constexpr struct ncclSymkGinAgParameters ginAgTuningParameters[ncclSymkGinAgArch_Count] = {
  // ctaBw, peakBw, gpuNetBw, baseUs, hopUs, shareUs, ctaNetBw, postUs, localChunkUs, firstContextUs, localPower.
  {68.9771, 1044.3587, 178.2701, 37.2824, 10.8427, 6.7661, 52.1764, 7.7635, 3.1359, .3621, 9.1813}, // Rubin.
  {48.7899, 719.4619, 53.9302, 33.1417, 8.9505, 4.0338, 40.3560, 5.2754, 1.3961, .3621, 9.1813}, // GB200.
  {52.0514, 708.2904, 106.7490, 37.9138, 7.2639, 6.0339, 65.3548, 8.1772, 1.8258, .3621, 9.1813}, // GB300.
};

} // namespace

enum ncclSymkGinAgArch ncclSymkGinAgArchForCapability(int capability) {
  if (capability == 100) return ncclSymkGinAgArch_Gb200;
  if (capability == 103) return ncclSymkGinAgArch_Gb300;
  if (RUBIN_AND_LATER(capability)) return ncclSymkGinAgArch_Rubin;
  return ncclSymkGinAgArch_Count;
}

struct ncclSymkGinAgParameters ncclSymkGinAgDefaults(enum ncclSymkGinAgArch arch) {
  return ginAgTuningParameters[arch];
}

struct ncclSymkGinAgPipeline ncclSymkGinAgPrepare(const struct ncclSymkGinAgParameters& parameters, size_t rankBytes,
                                                  int requestedCtas, int localRanks, int remoteRanks, int contexts) {
  struct ncclSymkGinAgPipeline pipeline = {};
  const struct ginAgPartition partition = ginAgPartitionForCtas(rankBytes, requestedCtas);
  pipeline.distinctSpanCount = ginAgCtaSpans(rankBytes, partition, pipeline.distinctSpanBytes);

  pipeline.lsaRanks = localRanks;
  pipeline.railRanks = remoteRanks;
  pipeline.baseUs = parameters.baseUs;
  pipeline.hopUs = parameters.hopUs;
  pipeline.localChunkUs = parameters.localChunkUs;

  const double ctaLoad = double(rankBytes) / pipeline.distinctSpanBytes[pipeline.distinctSpanCount - 1];
  pipeline.networkUsPerByte = std::max(1 / parameters.ctaNetBw, ctaLoad / parameters.gpuNetBw) / 1000;
  // Multicast fanout and send-to-self accounting cancel to this payload limit.
  // Scale the smooth maximum by its larger term to keep the fitted power numerically stable.
  const double localHigh = std::max(1 / parameters.ctaBw, ctaLoad * pipeline.lsaRanks / parameters.peakBw);
  const double localLow = std::min(1 / parameters.ctaBw, ctaLoad * pipeline.lsaRanks / parameters.peakBw);
  pipeline.localUsPerByte =
    localHigh * std::pow(1 + std::pow(localLow / localHigh, parameters.localPower), 1 / parameters.localPower) / 1000;
  pipeline.postingUs = parameters.shareUs * ((partition.activeCtas - 1) / contexts) + parameters.postUs;
  pipeline.firstIssueUs = (std::min(partition.activeCtas, contexts) - 1) * parameters.firstContextUs;

  return pipeline;
}

double ncclSymkGinAgPredict(const struct ncclSymkGinAgPipeline& pipeline, size_t chunkBytes) {
  double finishUs = 0;
  for (int spanIndex = 0; spanIndex < pipeline.distinctSpanCount; spanIndex++) {
    const uint64_t bytes = pipeline.distinctSpanBytes[spanIndex];
    const uint64_t full = bytes / chunkBytes;
    const uint64_t tail = bytes % chunkBytes;
    uint64_t rotations[ncclSymkGinAgMaxLocalRanks] = {0};
    int rotationCount = 1;
    // Rotation matters only when it moves a partial tail among full chunks.
    if (full != 0 && tail != 0) {
      for (int rank = 0; rank < pipeline.lsaRanks; rank++) {
        const int offsetMultiplier = rank ^ ((rank >> 1) & 1);
        rotations[rank] = (uint64_t(offsetMultiplier) * (32 << 20) % bytes) / chunkBytes;
      }
      std::sort(rotations, rotations + pipeline.lsaRanks);
      rotationCount = std::unique(rotations, rotations + pipeline.lsaRanks) - rotations;
    }
    for (int rotation = 0; rotation < rotationCount; rotation++) {
      const uint64_t start = rotations[rotation];
      // Equal full chunks before and after the tail form constant-length runs.
      const uint64_t counts[3] = {full - start, tail != 0 ? uint64_t(1) : uint64_t(0), start};
      const uint64_t lengths[3] = {chunkBytes, tail, chunkBytes};
      // Per chunk, the pipeline follows:
      //   sendDone = max(sendDone, previousHopArrival) + injectUs
      //   arrival = sendDone + hopUs
      //   copyDone = max(copyDone, arrival) + copyUs
      //
      // Equal-size chunks let us evaluate a complete run analytically.
      // Each firstArrivalUsByRun entry holds the preceding hop's first arrival
      // until we overwrite it with the current hop's first arrival.
      // The network and local-copy clocks continue across hops.
      double copyUsPerChunk[3];
      double firstArrivalUsByRun[3] = {};
      double networkAvailableUs = pipeline.firstIssueUs;
      double localCompleteUs = 0;
      for (int run = 0; run < 3; run++) {
        copyUsPerChunk[run] = lengths[run] * pipeline.localUsPerByte + pipeline.localChunkUs;
        localCompleteUs += counts[run] * copyUsPerChunk[run];
      }
      for (int hop = 1; hop < pipeline.railRanks; hop++) {
        for (int run = 0; run < 3; run++) {
          if (counts[run] == 0) continue;
          const double injectUs = lengths[run] * pipeline.networkUsPerByte + pipeline.postingUs;
          const double firstArrivalUs =
            std::max(networkAvailableUs, firstArrivalUsByRun[run]) + injectUs + pipeline.hopUs;
          firstArrivalUsByRun[run] = firstArrivalUs;
          networkAvailableUs = firstArrivalUs + (counts[run] - 1) * injectUs - pipeline.hopUs;
          const double localWorkUs = counts[run] * copyUsPerChunk[run];
          const double arrivalStallUs = (counts[run] - 1) * std::max(0., injectUs - copyUsPerChunk[run]);
          localCompleteUs = std::max(localCompleteUs + localWorkUs, firstArrivalUs + localWorkUs + arrivalStallUs);
        }
      }
      finishUs = std::max(finishUs, localCompleteUs);
    }
  }
  return pipeline.baseUs + finishUs;
}
