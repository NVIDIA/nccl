// Control experiment: same persistent GPU handshake, device-resident flags.
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#define CUDA(call) do { auto e=(call); if(e!=cudaSuccess) { fprintf(stderr,"line %d: %s\n",__LINE__,cudaGetErrorString(e)); exit(2); } } while(0)
#define DRIVER(call) do { auto e=(call); if(e!=CUDA_SUCCESS) { const char* msg;cuGetErrorString(e,&msg);fprintf(stderr,"line %d: %s\n",__LINE__,msg);exit(3); } } while(0)
__global__ void handshake(volatile unsigned long long* ready,volatile unsigned long long* ack,int n) {
  for(int i=1;i<=n;i++) {
    __threadfence_system(); *ready=i;
    while(*ack<(unsigned long long)i) {}
  }
}
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main() {
  setvbuf(stdout,nullptr,_IOLBF,0);
  unsigned long long *d;CUDA(cudaMalloc(&d,8192));CUDA(cudaMemset(d,0,8192));
  cudaStream_t kernel,control;CUDA(cudaStreamCreateWithFlags(&kernel,cudaStreamNonBlocking));CUDA(cudaStreamCreateWithFlags(&control,cudaStreamNonBlocking));
  cudaEvent_t event;CUDA(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
  handshake<<<1,1,0,kernel>>>(d,d+512,100);CUDA(cudaGetLastError());
  double start=now(),last=start;
  for(unsigned long long i=1;i<=100;i++) {
    DRIVER(cuStreamWaitValue64(control,(CUdeviceptr)d,i,CU_STREAM_WAIT_VALUE_GEQ));
    CUDA(cudaEventRecord(event,control));CUDA(cudaEventSynchronize(event));
    DRIVER(cuStreamWriteValue64(control,(CUdeviceptr)(d+512),i,0));
    double t=now();printf("STEP i=%llu ms=%.6f\n",i,(t-last)*1000);last=t;
  }
  CUDA(cudaStreamSynchronize(control));CUDA(cudaStreamSynchronize(kernel));
  printf("COMPLETE total_ms=%.6f\n",(now()-start)*1000);
  CUDA(cudaEventDestroy(event));CUDA(cudaStreamDestroy(control));CUDA(cudaStreamDestroy(kernel));CUDA(cudaFree(d));
}
