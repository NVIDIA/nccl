/*
 * NIIN hello world: put_signal between two GPUs.
 *
 * PE 0 writes a message to PE 1's buffer and signals completion.
 * PE 1 waits for the signal, then reads the message.
 *
 * Build:
 *   NIIN_HOME=<nccl>/contrib/niin
 *   nvcc hello_put_signal.cu -o hello_put_signal -DNIIN_HAS_MPI \
 *       -I ${NIIN_HOME}/include \
 *       -L ${NIIN_HOME}/lib -lnvshmem_host -lmpi \
 *       --expt-relaxed-constexpr -std=c++17 -arch=sm_89 \
 *       -Xlinker -rpath,${NIIN_HOME}/lib
 *
 * Run:
 *   mpirun -np 2 ./hello_put_signal
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include <mpi.h>

__global__ void hello_kernel(int* data, uint64_t* sig) {
  int pe   = nvshmem_my_pe();
  int npes = nvshmem_n_pes();

  if (pe == 0 && threadIdx.x == 0) {
    // Write a pattern into local buffer
    for (int i = 0; i < 4; i++) data[i] = 100 + i;
    __threadfence();

    // Put data + signal to PE 1
    int target = (npes > 1) ? 1 : 0;
    nvshmem_int_put_signal(data, data, 4, sig, 1, NVSHMEM_SIGNAL_ADD, target);
    printf("PE 0: sent [%d, %d, %d, %d] to PE %d\n",
           data[0], data[1], data[2], data[3], target);
  }

  if (pe == (npes > 1 ? 1 : 0) && threadIdx.x == 0) {
    // Wait for the signal
    nvshmem_signal_wait_until(sig, NVSHMEM_CMP_GE, 1);
    printf("PE %d: received [%d, %d, %d, %d]\n",
           pe, data[0], data[1], data[2], data[3]);
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // Bootstrap NIIN with MPI
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  MPI_Comm comm = MPI_COMM_WORLD;
  attr.mpi_comm = &comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

  char lib_name[256];
  nvshmem_info_get_name(lib_name);
  printf("[%s] Hello from PE %d of %d\n", lib_name, nvshmem_my_pe(), nvshmem_n_pes());

  // Allocate symmetric memory
  int*      data = (int*)nvshmem_malloc(64 * sizeof(int));
  uint64_t* sig  = (uint64_t*)nvshmem_malloc(sizeof(uint64_t));
  cudaMemset(data, 0, 64 * sizeof(int));
  cudaMemset(sig, 0, sizeof(uint64_t));

  nvshmem_barrier_all();

  hello_kernel<<<1, 32>>>(data, sig);
  cudaDeviceSynchronize();

  nvshmem_barrier_all();
  nvshmem_free(sig);
  nvshmem_free(data);
  nvshmem_finalize();

  MPI_Finalize();
  return 0;
}
