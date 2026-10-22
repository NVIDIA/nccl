// CPU/GPU FIFO handoff without NCCL or a network. Run under an external timeout.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#define CUDA(call) do { auto e=(call); if(e!=cudaSuccess) { fprintf(stderr,"line %d: %s\n",__LINE__,cudaGetErrorString(e)); exit(2); } } while(0)
__global__ void handshake(volatile unsigned long long* ready,volatile unsigned long long* ack,int n) {
  for(int i=1;i<=n;i++) {
    __threadfence_system(); *ready=i;
    while(*ack<(unsigned long long)i) {}
  }
}
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc,char**argv) {
  setvbuf(stdout,nullptr,_IOLBF,0);int delay=argc>1?atoi(argv[1]):0;int offset=argc>2?atoi(argv[2]):4096;
  if(offset<8||offset>65536)return 1;
  unsigned char *h,*d;CUDA(cudaHostAlloc(&h,131072,cudaHostAllocMapped));memset(h,0,131072);CUDA(cudaHostGetDevicePointer(&d,h,0));
  cudaPointerAttributes attr;CUDA(cudaPointerGetAttributes(&attr,h));
  printf("ENV type=%d host=%p device=%p delay_us=%d offset=%d\n",attr.type,h,d,delay,offset);
  cudaStream_t s;CUDA(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
  handshake<<<1,1,0,s>>>((unsigned long long*)d,(unsigned long long*)(d+offset),100);
  CUDA(cudaGetLastError());
  volatile auto* ready=(volatile unsigned long long*)h;volatile auto* ack=(volatile unsigned long long*)(h+offset);
  double start=now(),last=start;
  for(unsigned long long i=1;i<=100;i++) {
    while(*ready<i) { if(delay)usleep(delay);if(now()-start>20) { fprintf(stderr,"HOST_TIMEOUT step=%llu ready=%llu\n",i,*ready);_Exit(4); } }
    __atomic_store_n((unsigned long long*)ack,i,__ATOMIC_RELEASE);
    double t=now();printf("STEP i=%llu ms=%.6f\n",i,(t-last)*1000);last=t;
  }
  CUDA(cudaStreamSynchronize(s));printf("COMPLETE total_ms=%.6f\n",(now()-start)*1000);
  CUDA(cudaStreamDestroy(s));CUDA(cudaFreeHost(h));
}
