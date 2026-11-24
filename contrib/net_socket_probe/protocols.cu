// SPDX-License-Identifier: Apache-2.0
// Exercise slot wraparound, zero/unaligned lengths and fresh data on every iteration.
// Two hosts, one rank/GPU each. No framework, MPI, graphs or user registration.
#include <cuda_runtime.h>
#include <nccl.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CU(x) \
  do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
      fprintf(stderr, "CUDA line %d: %s\n", __LINE__, cudaGetErrorString(e)); \
      exit(2); \
    } \
  } while (0)
#define NC(x) \
  do { \
    ncclResult_t e = (x); \
    if (e != ncclSuccess) { \
      fprintf(stderr, "NCCL line %d: %s\n", __LINE__, ncclGetErrorString(e)); \
      exit(3); \
    } \
  } while (0)
static double now() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}
static void io(int fd, void* buf, size_t n, int wr) {
  char* p = (char*)buf;
  while (n) {
    ssize_t k = wr ? send(fd, p, n, MSG_NOSIGNAL) : recv(fd, p, n, 0);
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) {
      perror("control socket");
      exit(4);
    }
    p += k;
    n -= k;
  }
}
static void barrier(int fd, int rank) {
  char c = 1;
  if (rank == 0) {
    io(fd, &c, 1, 1);
    io(fd, &c, 1, 0);
  } else {
    io(fd, &c, 1, 0);
    io(fd, &c, 1, 1);
  }
}
int main(int argc, char** argv) {
  if (argc != 5) {
    fprintf(stderr, "Usage: %s RANK RANK0_IPV4 PORT TIMED_ITERS\n", argv[0]);
    return 1;
  }
  setvbuf(stdout, NULL, _IOLBF, 0);
  int rank = atoi(argv[1]), iters = atoi(argv[4]);
  if (rank < 0 || rank > 1 || iters < 1) return 1;
  int fd = socket(AF_INET, SOCK_STREAM, 0), one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  struct sockaddr_in a = {};
  a.sin_family = AF_INET;
  a.sin_port = htons(atoi(argv[3]));
  if (inet_pton(AF_INET, argv[2], &a.sin_addr) != 1) return 1;
  if (rank == 0) {
    if (bind(fd, (struct sockaddr*)&a, sizeof(a)) || listen(fd, 1)) {
      perror("bind/listen");
      return 4;
    }
    int s = accept(fd, NULL, NULL);
    close(fd);
    fd = s;
  } else {
    double deadline = now() + 30;
    while (connect(fd, (struct sockaddr*)&a, sizeof(a))) {
      if (now() > deadline) {
        perror("connect");
        return 4;
      }
      usleep(100000);
    }
  }
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  CU(cudaSetDevice(0));
  int cv, dv, nv;
  CU(cudaRuntimeGetVersion(&cv));
  CU(cudaDriverGetVersion(&dv));
  NC(ncclGetVersion(&nv));
  printf("ENV rank=%d pid=%d nccl=%d runtime=%d driver_api=%d\n", rank, getpid(), nv, cv, dv);
  ncclUniqueId id;
  if (rank == 0) {
    NC(ncclGetUniqueId(&id));
    io(fd, &id, sizeof(id), 1);
  } else io(fd, &id, sizeof(id), 0);
  ncclComm_t comm;
  NC(ncclCommInitRank(&comm, 2, id, rank));
  cudaStream_t stream;
  CU(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  const size_t counts[] = {0, 1, 17, 1024, 65536, 1048576};
  for (size_t count : counts) {
    int *x, *y;
    size_t cap = (count ? count : 1) * 2;
    CU(cudaMalloc(&x, cap * sizeof(int)));
    CU(cudaMalloc(&y, cap * sizeof(int)));
    int* input = (int*)malloc(cap * sizeof(int));
    int* output = (int*)malloc(cap * sizeof(int));
    for (int op = 0; op < 7; op++)
      for (int iter = 0; iter < iters; iter++) {
        if (iter == 0) printf("BEGIN rank=%d count=%zu op=%d\n", rank, count, op);
        int value = iter + 1 + rank * 101;
        for (size_t j = 0; j < cap; j++) input[j] = value + j % 97;
        CU(cudaMemcpyAsync(x, input, cap * sizeof(int), cudaMemcpyHostToDevice, stream));
        CU(cudaMemsetAsync(y, 0xff, cap * sizeof(int), stream));
        switch (op) {
        case 0:
          NC(ncclAllReduce(x, y, count, ncclInt, ncclSum, comm, stream));
          break;
        case 1:
          NC(ncclBroadcast(x, y, count, ncclInt, iter % 2, comm, stream));
          break;
        case 2:
          NC(ncclReduce(x, y, count, ncclInt, ncclSum, iter % 2, comm, stream));
          break;
        case 3:
          NC(ncclAllGather(x, y, count, ncclInt, comm, stream));
          break;
        case 4:
          NC(ncclReduceScatter(x, y, count, ncclInt, ncclSum, comm, stream));
          break;
        case 5:
          NC(ncclGroupStart());
          NC(ncclSend(x, count, ncclInt, 1 - rank, comm, stream));
          NC(ncclRecv(y, count, ncclInt, 1 - rank, comm, stream));
          NC(ncclGroupEnd());
          break;
        case 6:
          for (int step = 0; step < 12; step++) {
            NC(ncclGroupStart());
            NC(ncclSend(x, count, ncclInt, 1 - rank, comm, stream));
            NC(ncclRecv(y, count, ncclInt, 1 - rank, comm, stream));
            NC(ncclGroupEnd());
          }
          break;
        }
        CU(cudaMemcpyAsync(output, y, cap * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CU(cudaStreamSynchronize(stream));
        size_t n = op == 3 ? count * 2 : count;
        if (op == 2 && rank != iter % 2) n = 0;
        for (size_t j = 0; j < n; j++) {
          int expected = op == 1 ? iter + 1 + (iter % 2) * 101 + j % 97 :
                         op == 3 ? iter + 1 + (j / count) * 101 + (j % count) % 97 :
                         op == 4 ? 2 * (iter + 1 + (rank * count + j) % 97) + 101 :
                         op >= 5 ? iter + 1 + (1 - rank) * 101 + j % 97 :
                                   2 * (iter + 1 + j % 97) + 101;
          if (output[j] != expected) {
            fprintf(stderr, "BAD rank=%d op=%d count=%zu iter=%d index=%zu got=%d expected=%d\n", rank, op, count, iter,
                    j, output[j], expected);
            return 5;
          }
        }
      }
    printf("PASS rank=%d count=%zu operations=7 iterations=%d\n", rank, count, iters);
    free(input);
    free(output);
    CU(cudaFree(x));
    CU(cudaFree(y));
  }
  barrier(fd, rank);
  NC(ncclCommDestroy(comm));
  CU(cudaStreamDestroy(stream));
  close(fd);
  printf("COMPLETE rank=%d\n", rank);
  return 0;
}
