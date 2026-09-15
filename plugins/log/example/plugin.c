/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/* Example NCCL log plugin.
 *
 * Receives NCCL's log records as structured data rather than as lines of text, and prints them in a
 * key=value form that shows what a sink actually gets. The point of interest is the distinction between
 * a root-cause error and a warning re-reporting one: an origin arrives at level ERROR carrying the
 * ncclResult_t it is about to return, while a propagated report arrives at level WARN with no code.
 *
 * Build and use:
 *   make
 *   NCCL_LOG_PLUGIN=$PWD/libnccl-log-example.so NCCL_DEBUG=WARN  ./your_app   # errors and warnings
 *   NCCL_LOG_PLUGIN=$PWD/libnccl-log-example.so NCCL_DEBUG=ERROR ./your_app   # root causes only
 */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "nccl/log.h"

struct exampleState {
  /* onRecord can be called from several threads at once, so a sink owns its own serialization. */
  pthread_mutex_t lock;
  unsigned long records;
  unsigned long errors;
};

static const char* levelName(int level) {
  switch (level) {
    case NCCL_LOG_VERSION: return "VERSION";
    case NCCL_LOG_WARN: return "WARN";
    case NCCL_LOG_INFO: return "INFO";
    case NCCL_LOG_ABORT: return "ABORT";
    case NCCL_LOG_TRACE: return "TRACE";
    case NCCL_LOG_ATTN: return "ATTN";
    case NCCL_LOG_ERROR: return "ERROR";
    default: return "UNKNOWN";
  }
}

static const char* codeName(ncclResult_t code) {
  switch (code) {
    case ncclSuccess: return "success";
    case ncclUnhandledCudaError: return "unhandled cuda error";
    case ncclSystemError: return "system error";
    case ncclInternalError: return "internal error";
    case ncclInvalidArgument: return "invalid argument";
    case ncclInvalidUsage: return "invalid usage";
    case ncclRemoteError: return "remote error";
    case ncclInProgress: return "in progress";
    case ncclTimeout: return "timeout";
    default: return "unknown";
  }
}

static ncclResult_t exampleInit(void** context) {
  struct exampleState* state = (struct exampleState*)calloc(1, sizeof(*state));
  if (state == NULL) return ncclSystemError;
  if (pthread_mutex_init(&state->lock, NULL) != 0) {
    free(state);
    return ncclSystemError;
  }
  *context = state;
  fprintf(stderr, "LOG/Plugin: example sink installed\n");
  return ncclSuccess;
}

static ncclResult_t exampleOnRecord(void* context, const ncclDebugLogRecord_v1_t* record) {
  struct exampleState* state = (struct exampleState*)context;

  pthread_mutex_lock(&state->lock);
  state->records++;
  if (record->code != ncclSuccess) state->errors++;

  if (record->code != ncclSuccess) {
    /* Root cause: NCCL named the error it is about to return. This is the record a telemetry pipeline
     * wants, and the one to capture a stack for -- exactly once, here, rather than at every layer that
     * re-reports it on the way out. */
    fprintf(stderr, "LOG/Plugin: level=%s code=%d(%s) subsys=0x%lx rank=%s:%d:%d dev=%d at=%s:%d(%s)"
                    " event=\"%s\" msg=\"%s\"\n",
            levelName(record->level), (int)record->code, codeName(record->code), record->subSys,
            record->hostname ? record->hostname : "?", record->pid, record->tid, record->cudaDev,
            record->file ? record->file : "?", record->line, record->func ? record->func : "?",
            record->format ? record->format : "", record->message ? record->message : "");
  } else {
    fprintf(stderr, "LOG/Plugin: level=%s subsys=0x%lx rank=%s:%d:%d msg=\"%s\"\n", levelName(record->level),
            record->subSys, record->hostname ? record->hostname : "?", record->pid, record->tid,
            record->message ? record->message : "");
  }
  pthread_mutex_unlock(&state->lock);
  return ncclSuccess;
}

static ncclResult_t exampleFinalize(void* context) {
  struct exampleState* state = (struct exampleState*)context;
  fprintf(stderr, "LOG/Plugin: example sink removed after %lu records, %lu of them root-cause errors\n",
          state->records, state->errors);
  pthread_mutex_destroy(&state->lock);
  free(state);
  return ncclSuccess;
}

ncclLogSink_v1_t NCCL_LOG_PLUGIN_SYMBOL = {
  "example",
  exampleInit,
  exampleOnRecord,
  exampleFinalize,
};
