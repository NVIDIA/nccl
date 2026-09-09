/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "utils.h"
#include <ctype.h>
#include <errno.h>
#include <getopt.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

/**
 * NCCL RAS Example: Fault Detection
 * ===================================
 *
 * WHAT THIS DEMONSTRATES:
 * RAS (Reliability, Availability, Serviceability) is a diagnostic subsystem
 * built into NCCL.  It starts automatically when the first communicator is
 * created — no application code changes needed.  Each process runs a
 * lightweight background thread that connects to peer processes over a ring
 * overlay and maintains a cluster-wide view of every process, communicator,
 * and GPU in the job.
 *
 * RAS exposes this view over a plain TCP socket (default localhost:28028).
 * Any external tool — ncclras, nc, telnet — can connect at any time and
 * query the live state of the running job, completely independently of the
 * application itself.
 *
 * This example keeps a small AllReduce workload running (one operation per
 * second), injects a configurable fault, and then simply stays alive so you
 * can drive the RAS client from another terminal and watch the report
 * change at your own pace:
 *
 *   HEALTHY     → all ranks RUNNING, operation counts ticking in lockstep
 *   MISMATCH    → a straggler falls behind on collective launches (sleep mode)
 *   INCOMPLETE  → a rank stops responding; its data is missing from reports
 *   DEAD        → the rank has been unreachable for 60s; declared permanently dead
 *
 * HOW TO USE:
 *   1. Launch this program (e.g. via SLURM or mpirun).
 *   2. On startup it prints the hostname and port where RAS is listening.
 *   3. Query RAS whenever you like (find the node with `squeue`):
 *        echo "verbose status" | nc <hostname> 28028
 *        ncclras -h <hostname> -p 28028 -v
 *   4. The fault is injected --wait seconds (default 60) after the workload
 *      starts, and the job then stays alive for --observe seconds (default
 *      120) so you can watch the detection timeline.  See README.md for the
 *      expected output at each stage.
 *
 * FAULT MODES:
 *   --type exit     (default) The failing rank calls exit(0).  RAS detects
 *                   the failure immediately via socket close.  DEAD in ~60s.
 *
 *   --type suspend  The failing rank receives SIGSTOP, freezing every thread
 *                   including the RAS thread.  No keep-alive messages are
 *                   sent; RAS peers detect the gap.  DEAD in ~80s.
 *
 *   --type sleep    The failing rank stays alive but stops launching
 *                   collectives for a while.  RAS counts collective LAUNCHES
 *                   per rank, and launches are asynchronous, so the surviving
 *                   ranks' counts race ahead and RAS reports MISMATCH while
 *                   every rank is still RUNNING.  Unlike the other modes this
 *                   one recovers: the straggler catches up and the report
 *                   returns to OK.
 *
 * COMMAND-LINE OPTIONS:
 *   -t, --type [exit|suspend|sleep]   Fault mode (default: exit)
 *   -r, --ranks N[,M,...]       Comma/space-separated ranks to fail
 *                               (default: last rank; rank 0 always survives)
 *   -w, --wait S                Seconds of healthy AllReduce activity before
 *                               the fault is injected (default: 60).  This is
 *                               your budget for logging in to a job node and
 *                               attaching the RAS client.
 *   -o, --observe S             Seconds the job stays alive (and queryable)
 *                               after the fault (default: 120 — long enough
 *                               to see DEAD at ~60s / ~80s in exit/suspend
 *                               mode, or the full MISMATCH → OK cycle in
 *                               sleep mode)
 *   -h, --help                  Show usage
 *
 * ENVIRONMENT VARIABLES:
 *   NCCL_RAS_ADDR            Override the RAS client listening address
 *                            (default localhost:28028).  Set to
 *                            <host-ip>:28028 to accept connections from other
 *                            hosts; <host-ip>=0.0.0.0 accepts any host (mind
 *                            the security exposure).
 *
 * NOTE ON MPI TEARDOWN:
 *   In exit/suspend mode MPI_Finalize() is called before fault injection so
 *   the dying rank is no longer managed by mpirun; no --continuous flag is
 *   needed.  In sleep mode every rank stays alive, so MPI remains active for
 *   the whole run and teardown is a normal collective ncclCommDestroy().
 *
 * Error-checking macros (NCCLCHECK, CUDACHECK, MPICHECK) and util_broadcast()
 * are shared helpers from docs/examples/common/include/.
 *
 * Rank-selection logic adapted from test/ras/single_comm_exit.cu
 * (Roberto Gioiosa, MR 2348).
 */

enum TestType {
  TEST_EXIT = 1,
  TEST_SUSPEND = 2,
  TEST_SLEEP = 3
};

/* How long a sleep-mode straggler stops launching collectives. */
#define STRAGGLER_NAP_SEC 40

/* -------------------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------------- */

/**
 * Launch one AllReduce per second for the given number of seconds.
 *
 * With sync=true the stream is synchronized after every launch, keeping all
 * ranks in lockstep — this is the healthy workload.  With sync=false the
 * launches just queue up on the stream; the surviving ranks use this while a
 * straggler naps, so their launch counts race ahead even though the kernels
 * cannot complete without the missing rank.
 */
static void allreduceLoop(int seconds, bool sync, float* sendbuf, float* recvbuf, size_t count, ncclComm_t comm,
                          cudaStream_t stream) {
  for (int i = 0; i < seconds; i++) {
    NCCLCHECK(ncclAllReduce(sendbuf, recvbuf, count, ncclFloat, ncclSum, comm, stream));
    if (sync) CUDACHECK(cudaStreamSynchronize(stream));
    sleep(1);
  }
}

/* Skip whitespace and commas; used when parsing a rank list. */
static const char* skipSpacesAndCommas(const char* p) {
  while (*p && (isspace((unsigned char)*p) || *p == ',')) ++p;
  return p;
}

/**
 * Parse a comma/space-separated list of rank indices into failed_ranks[].
 * Rank 0 is never allowed to fail (it drives the example timeline).
 * Returns 0 on success, -1 on malformed input or empty list.
 */
static int parseFailedRanks(const char* env, int n_ranks, int* failed_ranks, int* n_failed_out) {
  int n = 0;
  const char* p;
  char* end;
  long v;

  if (env == NULL || failed_ranks == NULL || n_failed_out == NULL) goto err;

  p = skipSpacesAndCommas(env);
  while (*p) {
    v = strtol(p, &end, 10);
    if (end == p) goto err;
    if (v < 0 || v >= (long)n_ranks) goto err;
    if (v != 0) {
      if (n >= n_ranks - 1) goto err;
      failed_ranks[n++] = (int)v;
    }
    p = skipSpacesAndCommas(end);
  }

  if (n == 0) goto err;

  *n_failed_out = n;
  return 0;

err:
  if (n_failed_out) *n_failed_out = 0;
  return -1;
}

static bool shouldFail(const int* failed_ranks, int n_failed, int my_rank) {
  for (int i = 0; i < n_failed; i++) {
    if (failed_ranks[i] == my_rank) return true;
  }
  return false;
}

/* Parse a non-negative integer number of seconds; keeps *out on error. */
static void parseSeconds(const char* arg, const char* optname, int* out) {
  char* end = NULL;
  errno = 0;
  long v = strtol(arg, &end, 10);
  if (errno != 0 || end == arg || *end != '\0' || v < 0) {
    fprintf(stderr, "Invalid %s \"%s\" — keeping default %ds\n", optname, arg, *out);
  } else {
    *out = (int)v;
  }
}

static void printUsage(FILE* f, const char* prog) {
  fprintf(f,
          "Usage: %s [OPTIONS]\n"
          "\n"
          "  -t, --type T     Fault mode:\n"
          "                     exit     (default) rank calls exit(0); detected via socket close\n"
          "                     suspend  rank receives SIGSTOP; detected via keep-alive timeout\n"
          "                     sleep    rank stops launching collectives; RAS reports\n"
          "                              MISMATCH, then OK again once it catches up\n"
          "  -r, --ranks L    Comma/space-separated MPI ranks to fail\n"
          "                     (default: last rank; rank 0 is always preserved)\n"
          "  -w, --wait S     Seconds of healthy AllReduce activity before the fault\n"
          "                     (default: 60)\n"
          "  -o, --observe S  Seconds the job stays queryable after the fault\n"
          "                     (default: 120)\n"
          "  -h, --help       Show this help\n"
          "\n"
          "Environment variables:\n"
          "  NCCL_RAS_ADDR            RAS client bind address (default localhost:28028;\n"
          "                           set to <host-ip>:28028 for remote access, or\n"
          "                           <host-ip>=0.0.0.0 for any host)\n",
          prog);
}

static int parseArgs(int argc, char** argv, int n_mpi_ranks, int* test_type, int* failed_ranks, int* n_failed,
                     int* wait_sec, int* observe_sec) {
  static struct option longopts[] = {
    {"type", required_argument, NULL, 't'}, {"ranks", required_argument, NULL, 'r'},
    {"wait", required_argument, NULL, 'w'}, {"observe", required_argument, NULL, 'o'},
    {"help", no_argument, NULL, 'h'},       {NULL, 0, NULL, 0},
  };
  int opt;

  optind = 1;
  opterr = 0;
  while ((opt = getopt_long(argc, argv, "t:r:w:o:h", longopts, NULL)) != -1) {
    switch (opt) {
    case 't':
      if (strcasecmp(optarg, "exit") == 0) {
        *test_type = TEST_EXIT;
      } else if (strcasecmp(optarg, "suspend") == 0) {
        *test_type = TEST_SUSPEND;
      } else if (strcasecmp(optarg, "sleep") == 0) {
        *test_type = TEST_SLEEP;
      } else {
        fprintf(stderr, "Invalid --type \"%s\" (use exit, suspend, or sleep)\n", optarg);
        return -1;
      }
      break;
    case 'r':
      if (parseFailedRanks(optarg, n_mpi_ranks, failed_ranks, n_failed) != 0) {
        fprintf(stderr, "Invalid --ranks \"%s\" — defaulting to last rank\n", optarg);
        *n_failed = 1;
        failed_ranks[0] = n_mpi_ranks - 1;
      }
      break;
    case 'w':
      parseSeconds(optarg, "--wait", wait_sec);
      break;
    case 'o':
      parseSeconds(optarg, "--observe", observe_sec);
      break;
    case 'h':
      return 1;
    default:
      fprintf(stderr, "Unknown option.  Run with --help for usage.\n");
      return -1;
    }
  }
  return 0;
}

/* Determine this process's local rank on its physical node (for GPU
 * selection in multi-node runs). */
static int getLocalRank(MPI_Comm comm) {
  int world_rank;
  MPI_Comm_rank(comm, &world_rank);
  MPI_Comm node_comm;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &node_comm);
  int node_rank;
  MPI_Comm_rank(node_comm, &node_rank);
  MPI_Comm_free(&node_comm);
  return node_rank;
}

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */

int main(int argc, char* argv[]) {
  int mpi_rank, mpi_size, local_rank;
  ncclComm_t comm = NULL;
  cudaStream_t stream = NULL;
  ncclUniqueId nccl_id;
  float *sendbuf = NULL, *recvbuf = NULL;
  const size_t count = 1024 * 1024; /* 1 M floats */

  int test_type = TEST_EXIT;
  int n_failed = 1;
  int* failed_ranks = NULL;
  int wait_sec = 60;  /* healthy activity before the fault */
  int observe_sec = 120; /* post-fault lifetime */
  int cli;

  /* =========================================================================
   * STEP 1: Initialize MPI and parse options
   * ====================================================================== */

  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));

  if (mpi_size < 2) {
    if (mpi_rank == 0) fprintf(stderr, "ERROR: at least 2 MPI ranks required.\n");
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  failed_ranks = (int*)malloc(mpi_size * sizeof(int));
  if (failed_ranks == NULL) {
    fprintf(stderr, "rank %d: malloc failed\n", mpi_rank);
    MPI_Finalize();
    return EXIT_FAILURE;
  }
  failed_ranks[0] = mpi_size - 1; /* default: fail the last rank */

  cli = parseArgs(argc, argv, mpi_size, &test_type, failed_ranks, &n_failed, &wait_sec, &observe_sec);
  if (cli != 0) {
    if (mpi_rank == 0) printUsage(stdout, argv[0] ? argv[0] : "ras_fault_detection");
    free(failed_ranks);
    MPI_Finalize();
    return cli == 1 ? 0 : EXIT_FAILURE;
  }

  local_rank = getLocalRank(MPI_COMM_WORLD);
  CUDACHECK(cudaSetDevice(local_rank));
  CUDACHECK(cudaStreamCreate(&stream));

  /* =========================================================================
   * STEP 2: Create the NCCL communicator.
   *
   * This is where RAS starts.  ncclCommInitRank spawns the background RAS
   * thread, binds a peer-overlay socket on the bootstrap NIC, and binds the
   * client-facing socket on localhost:28028 (or NCCL_RAS_ADDR if set).
   * No application code is required — it all happens automatically.
   * ====================================================================== */

  if (mpi_rank == 0) NCCLCHECK(ncclGetUniqueId(&nccl_id));
  util_broadcast(0, mpi_rank, &nccl_id);
  NCCLCHECK(ncclCommInitRank(&comm, mpi_size, nccl_id, mpi_rank));

  CUDACHECK(cudaMalloc(&sendbuf, count * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuf, count * sizeof(float)));
  CUDACHECK(cudaMemset(sendbuf, 1, count * sizeof(float)));

  /* =========================================================================
   * STEP 3: Print connection info.
   *
   * Rank 0 prints the hostname so the user knows which node to connect to.
   * ====================================================================== */

  if (mpi_rank == 0) {
    char hostname[256] = "localhost";
    gethostname(hostname, sizeof(hostname));

    const char* rasAddr = getenv("NCCL_RAS_ADDR");
    int rasPort = 28028; /* default; NCCL may use a different port if set */
    if (rasAddr) {
      const char* colon = strrchr(rasAddr, ':');
      if (colon) rasPort = atoi(colon + 1);
    }

    printf("\n");
    printf("════════════════════════════════════════════════════════════\n");
    printf("  NCCL RAS Fault Detection Example\n");
    printf("════════════════════════════════════════════════════════════\n");
    printf("  Ranks   : %d\n", mpi_size);
    printf("  Mode    : %s\n", test_type == TEST_EXIT    ? "exit  (fast — socket close)" :
                               test_type == TEST_SUSPEND ? "suspend (slow — SIGSTOP / keep-alive)" :
                                                           "sleep (straggler — MISMATCH, recoverable)");
    printf("  Fault   : rank(s)");
    for (int i = 0; i < n_failed; i++) printf(" %d", failed_ranks[i]);
    printf(", injected after %ds of healthy activity\n", wait_sec);
    printf("  Observe : job stays queryable for %ds after the fault\n", observe_sec);
    printf("\n");
    printf("  RAS is now listening on  %s:%d\n", hostname, rasPort);
    printf("    echo \"verbose status\" | nc %s %d\n", hostname, rasPort);
    printf("    ncclras -h %s -p %d -v\n", hostname, rasPort);
    printf("════════════════════════════════════════════════════════════\n\n");
    fflush(stdout);
  }

  /* =========================================================================
   * STEP 4: Healthy phase — one AllReduce per second, all ranks in lockstep.
   *
   * RAS tracks how many collectives of each type every rank has launched.
   * While the job is healthy, successive RAS queries show the AllReduce
   * counts ticking up together on every rank.
   * ====================================================================== */

  if (mpi_rank == 0) {
    printf("Healthy phase: running one AllReduce per second for %ds — "
           "query RAS now.\n\n",
           wait_sec);
    fflush(stdout);
  }
  allreduceLoop(wait_sec, /*sync*/ true, sendbuf, recvbuf, count, comm, stream);
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  if (test_type == TEST_SLEEP) {
    /* =======================================================================
     * STEP 5 (sleep mode): straggler injection.
     *
     * The straggler stays alive — its RAS thread keeps answering keepalives,
     * so the job never turns INCOMPLETE and nobody is declared DEAD.  It just
     * stops launching collectives while the survivors keep going (without
     * waiting for completion), so the per-rank launch counts diverge and RAS
     * reports MISMATCH.  Once the straggler catches up the counts converge
     * again and the report returns to OK.
     * ==================================================================== */

    if (mpi_rank == 0) {
      printf("Fault injected: straggler(s) stop launching collectives for %ds "
             "— query RAS to see MISMATCH.\n\n",
             STRAGGLER_NAP_SEC);
      fflush(stdout);
    }

    if (shouldFail(failed_ranks, n_failed, mpi_rank)) {
      sleep(STRAGGLER_NAP_SEC);
      printf("  [rank %d] straggler: awake — catching up on the missed "
             "AllReduce launches\n",
             mpi_rank);
      fflush(stdout);
      for (int i = 0; i < STRAGGLER_NAP_SEC; i++)
        NCCLCHECK(ncclAllReduce(sendbuf, recvbuf, count, ncclFloat, ncclSum, comm, stream));
    } else {
      allreduceLoop(STRAGGLER_NAP_SEC, /*sync*/ false, sendbuf, recvbuf, count, comm, stream);
    }

    /* Drains once the straggler has caught up on every queued collective. */
    CUDACHECK(cudaStreamSynchronize(stream));
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (mpi_rank == 0) {
      printf("Straggler caught up — counts are in lockstep again; RAS reports "
             "OK for the remaining %ds.\n\n",
             observe_sec > STRAGGLER_NAP_SEC ? observe_sec - STRAGGLER_NAP_SEC : 0);
      fflush(stdout);
    }

    /* Back to the healthy workload for the rest of the observation time. */
    if (observe_sec > STRAGGLER_NAP_SEC)
      allreduceLoop(observe_sec - STRAGGLER_NAP_SEC, /*sync*/ true, sendbuf, recvbuf, count, comm, stream);

    if (mpi_rank == 0) {
      printf("Example complete.  Cleaning up.\n");
      fflush(stdout);
    }

    /* Every rank is alive, so a normal collective teardown is safe here —
     * no need for the ncclCommAbort escape hatch the fatal modes use. */
    CUDACHECK(cudaFree(sendbuf));
    CUDACHECK(cudaFree(recvbuf));
    CUDACHECK(cudaStreamDestroy(stream));
    NCCLCHECK(ncclCommDestroy(comm));

    free(failed_ranks);
    MPICHECK(MPI_Finalize());
    return 0;
  }

  /* =========================================================================
   * STEP 5 (exit/suspend): tear down MPI, then inject the fault.
   *
   * With MPI finalized the dying rank is no longer owned by the launcher,
   * so no --continuous flag is needed.
   *
   * exit mode:
   *   The failing rank calls exit(0).  The OS closes all sockets, including
   *   the RAS TCP connection.  Peer daemons see an immediate EOF, set a retry
   *   timer, and after 60s declare the peer DEAD.
   *
   * suspend mode:
   *   SIGSTOP freezes every thread in this process — including the RAS thread.
   *   No keep-alive messages are sent.  Peers flag the connection as delayed
   *   at ~5s, abort the socket at ~20s, and declare the peer DEAD at ~80s.
   * ====================================================================== */

  MPICHECK(MPI_Finalize());

  if (mpi_rank == 0) {
    printf("Fault injected (%s).  The job stays queryable for %ds — query "
           "RAS to watch INCOMPLETE turn into DEAD.\n\n",
           test_type == TEST_EXIT ? "exit" : "suspend", observe_sec);
    fflush(stdout);
  }

  if (shouldFail(failed_ranks, n_failed, mpi_rank)) {
    if (test_type == TEST_EXIT) {
      free(failed_ranks);
      exit(0);
    } else {
      kill(getpid(), SIGSTOP);
    }
  }

  /* =========================================================================
   * STEP 6: Observation — surviving ranks simply stay alive.
   *
   * Their collective launch counts freeze at the pre-fault values while RAS
   * walks its detection timeline (INCOMPLETE within seconds; DEAD after 60s
   * of unreachability — see README.md).
   * ====================================================================== */

  sleep(observe_sec);

  /* =========================================================================
   * STEP 7: Cleanup.
   *
   * ncclCommAbort, NOT ncclCommFinalize.  ncclCommFinalize is a collective
   * that synchronises all ranks and would block forever waiting on the dead
   * peer.  ncclCommAbort tears down the local communicator unconditionally.
   * ====================================================================== */

  if (mpi_rank == 0) {
    printf("Example complete.  Cleaning up.\n");
    fflush(stdout);
  }

  CUDACHECK(cudaFree(sendbuf));
  CUDACHECK(cudaFree(recvbuf));
  CUDACHECK(cudaStreamDestroy(stream));
  NCCLCHECK(ncclCommAbort(comm));

  free(failed_ranks);
  return 0;
}
