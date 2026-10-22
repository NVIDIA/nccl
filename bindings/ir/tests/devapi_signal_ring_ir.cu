#include "nccl_device_ir.h"

constexpr bool Prints = true;

extern "C" __global__ void runDevice(ncclIrDevComm comm) {
#if __CUDA_ARCH__ >= 700
  int t = threadIdx.x;
  ncclTeam world = ncclIrTeamWorld(&comm);

  // Initialize the opaque CTA coop
  ncclIrCoop coop;
  ncclIrCoopInitCta(&coop);

  // Initialize the ALL-mask GIN object
  ncclIrGin net;
  ncclIrGinInit(&net, &comm, 0);

  // Initialize thread-level coop for signal
  ncclIrCoop thread_coop;
  ncclIrCoopInitThread(&thread_coop);

  for (int round=0; round < 10; round++) {
    if (t==0) {
      ncclIrGinSignal(&net, world, (world.rank + 1) % world.nRanks,
                    NCCL_GIN_SIGNAL_TYPE_INDEXED, /*signalWin=*/nullptr, /*signalOffset=*/0,
                    /*signalId=*/0, /*isStrong=*/true,
                    /*signalOp=*/ncclGinSignalAdd, /*signalOpArg=*/1,
                    &thread_coop,
                    /*isDescriptor=*/false, /*descriptor=*/nullptr,
                    cuda::thread_scope_thread, cuda::thread_scope_device, ncclGinOptFlagsDefault);
    }

    ncclIrGinWaitSignal(&net, &coop, /*signal=*/0, /*least=*/round+1, /*bits=*/64, cuda::memory_order_acquire);

    if (world.rank == round % world.nRanks && t==0) {
      if (Prints) printf("[Rank %d] Round %d\n", world.rank, round);
    }
  }
#endif
}
