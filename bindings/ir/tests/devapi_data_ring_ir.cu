#include "nccl_device_ir.h"

#if __CUDA_ARCH__ >= 700
constexpr int BufElts = 1<<20;
constexpr bool Prints = false;
#endif

extern "C" __global__ void runDevice(ncclIrDevComm comm, ncclDevResourceHandle hbuf) {
#if __CUDA_ARCH__ >= 700
  int t = threadIdx.x;
  int tn = blockDim.x;
  ncclTeam world;
  ncclIrTeamWorld(&comm, &world);
  int b = world.rank*gridDim.x + blockIdx.x;
  int bn = world.nRanks*gridDim.x;
  int up_rank = (b+bn-1)%bn / gridDim.x;
  int up_block = (b+bn-1)%bn % gridDim.x;
  int down_rank = (b+1)%bn / gridDim.x;
  int down_block = (b+1)%bn % gridDim.x;

  // Get local pointer for direct memory access
  int* sbuf_local = (int*)ncclIrGetResourceBufferLocalPointer(&comm, hbuf);
  int* rbuf_local = sbuf_local + gridDim.x*BufElts;

  // Get window and offset for GIN operations
  ncclWindow_t bufWindow;
  size_t bufBaseOffset;
  ncclIrGetResourceBuffer(&comm, hbuf, &bufWindow, &bufBaseOffset);

  unsigned sigData0 = 0;
  unsigned sigFree0 = gridDim.x;

  // Initialize the opaque CTA coop
  ncclIrCoop coop;
  ncclIrCoopInitCta(&coop);

  // Initialize the ALL-mask GIN object
  ncclIrGin net;
  ncclIrGinInit(&net, &comm, 0);

  // Allocate storage ONCE outside loop to reduce register pressure
  ncclIrCoop thread_coop;
  ncclIrCoopInitThread(&thread_coop);

  int accum = 0;
  for (uint32_t round=0; round < 10; round++) {
    if (t==0 && b == round%bn) printf("Round %d\n", round);
    // Consume recv buf (rbuf) and populate send buf (sbuf)
    for (int i=t; i < BufElts; i += tn) {
      int got = rbuf_local[blockIdx.x*BufElts + i];
      if (Prints) {
        if (got != (int)round) {
          printf("[%d:%d]: ERROR round=%d rbuf[%d]=%d but expected %d\n", world.rank, blockIdx.x, round, i, got, round);
        }
      } else {
        assert(got == (int)round);
      }
      sbuf_local[blockIdx.x*BufElts + i] = 1 + got;
    }
    __syncthreads();

    if (t==0) {
      ncclIrGinSignal(&net, &world, up_rank,
                    NCCL_GIN_SIGNAL_TYPE_INDEXED, /*signalWin=*/nullptr, /*signalOffset=*/0,
                    /*signalId=*/sigFree0 + up_block, /*isStrong=*/true,
                    /*signalOp=*/ncclGinSignalInc, /*signalOpArg=*/1,
                    &thread_coop,
                    /*isDescriptor=*/false, /*descriptor=*/nullptr,
                    cuda::thread_scope_thread, cuda::thread_scope_device, ncclGinOptFlagsDefault);
    }

    // Wait for downstream to give us free space
    if (Prints) if (t==0) printf("[%d:%d] round=%d Send free signal\n", world.rank, blockIdx.x, round);
    ncclIrGinWaitSignal(&net, &coop, sigFree0 + blockIdx.x, round+1, 64, cuda::memory_order_acquire);
    if (Prints) if (t==0) printf("[%d:%d] round=%d Got free signal\n", world.rank, blockIdx.x, round);

    // Send data downstream in chunks
    int nChunks = min(BufElts, 1 + (round*0xdeadbeefu >> (32-10)));
    int chunkElts = BufElts/nChunks;
    nChunks = (BufElts + chunkElts-1)/chunkElts;
    if (Prints) if (t==0) printf("[%d:%d] round=%d Sending %d chunks\n", world.rank, blockIdx.x, round, nChunks);

    #pragma unroll 1
    for (int i=t; i < nChunks; i += tn) {
      ncclWindow_t dstWnd = bufWindow;
      size_t dstOffset = bufBaseOffset + (gridDim.x*BufElts + down_block*BufElts + i*chunkElts) * sizeof(int);
      ncclWindow_t srcWnd = bufWindow;
      size_t srcOffset = bufBaseOffset + (blockIdx.x*BufElts + i*chunkElts) * sizeof(int);
      size_t bytes = min(chunkElts, BufElts - i*chunkElts) * sizeof(int);

      ncclIrGinPut(&net, &world, down_rank, dstWnd, dstOffset, srcWnd, srcOffset, bytes,
                    NCCL_GIN_SIGNAL_TYPE_INDEXED, /*signalWin=*/nullptr, /*signalOffset=*/0,
                    /*signalId=*/sigData0 + down_block, /*isStrong=*/false,
                    /*signalOp=*/ncclGinSignalInc, /*signalOpArg=*/1,
                    /*isCounter=*/false, /*counterId=*/0,
                    &thread_coop,
                    /*isDescriptor=*/false, /*descriptor=*/nullptr,
                    cuda::thread_scope_thread, cuda::thread_scope_device, ncclGinOptFlagsDefault,
                    ncclGinSegmentTypeDevice);
    }

    if (Prints) if (t==0) printf("[%d:%d] round=%d wait chunks\n", world.rank, blockIdx.x, round);
    // Wait for all chunks from upstream
    ncclIrGinWaitSignal(&net, &coop, sigData0 + blockIdx.x, accum + nChunks, 64, cuda::memory_order_acquire);
    // Wait for all outgoing to settle
    ncclIrGinFlush(&net, &coop, cuda::memory_order_acquire, /*isDescriptor=*/false, /*descriptor=*/nullptr);
    if (Prints) if (t==0) printf("[%d:%d] round=%d Got chunks\n", world.rank, blockIdx.x, round);

    accum += nChunks;
  }
#endif
}
