// Isolate CUDA mapped-host accesses from NCCL and the network.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CUDA(call) do { auto e=(call); if(e!=cudaSuccess) { fprintf(stderr,"line %d: %s\n",__LINE__,cudaGetErrorString(e)); exit(2); } } while(0)
__global__ void copyWords(unsigned* dst, const unsigned* src, size_t n) {
  for (size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x) dst[i]=src[i];
}
__global__ void emptyKernel() {}
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static void report(const char* mode,size_t bytes,std::vector<double>& a) {
  std::sort(a.begin(),a.end());
  printf("RESULT mode=%s bytes=%zu samples=%zu median_ms=%.6f max_ms=%.6f validated=1\n",mode,bytes,a.size(),a[a.size()/2],a.back());
}
int main() {
  setvbuf(stdout,nullptr,_IOLBF,0);
  CUDA(cudaSetDevice(0));
  int r,d;CUDA(cudaRuntimeGetVersion(&r));CUDA(cudaDriverGetVersion(&d));
  printf("ENV runtime=%d driver_api=%d\n",r,d);
  cudaStream_t stream;CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  unsigned *host,*mapped,*gpu;
  CUDA(cudaHostAlloc(&host,1048576,cudaHostAllocMapped));
  CUDA(cudaHostGetDevicePointer(&mapped,host,0));CUDA(cudaMalloc(&gpu,1048576));
  for (size_t bytes: {size_t(4096),size_t(65536),size_t(1048576)}) {
    for(int mode=0;mode<5;mode++) {
      const char* name[]={"empty_launch","mapped_H2D","memcpy_H2D","mapped_D2H","memcpy_D2H"};
      std::vector<double> a;
      for(int i=0;i<25;i++) {
        unsigned char value=1+i;
        if(mode<=2) { memset(host,value,bytes);CUDA(cudaMemsetAsync(gpu,0,bytes,stream)); }
        else { CUDA(cudaMemsetAsync(gpu,value,bytes,stream));memset(host,0,bytes); }
        CUDA(cudaStreamSynchronize(stream));
        printf("BEGIN mode=%s bytes=%zu iter=%d\n",name[mode],bytes,i);
        double t=now();
        if(mode==0) emptyKernel<<<1,1,0,stream>>>();
        if(mode==1) copyWords<<<1,256,0,stream>>>(gpu,mapped,bytes/4);
        if(mode==2) CUDA(cudaMemcpyAsync(gpu,host,bytes,cudaMemcpyHostToDevice,stream));
        if(mode==3) copyWords<<<1,256,0,stream>>>(mapped,gpu,bytes/4);
        if(mode==4) CUDA(cudaMemcpyAsync(host,gpu,bytes,cudaMemcpyDeviceToHost,stream));
        CUDA(cudaGetLastError());CUDA(cudaStreamSynchronize(stream));
        double ms=(now()-t)*1000;
        printf("ITER mode=%s bytes=%zu iter=%d ms=%.6f\n",name[mode],bytes,i,ms);
        if(mode==1||mode==2) CUDA(cudaMemcpy(host,gpu,bytes,cudaMemcpyDeviceToHost));
        if(mode) for(size_t j=0;j<bytes;j++) if(((unsigned char*)host)[j]!=value) { fprintf(stderr,"BAD DATA %zu\n",j);return 3; }
        if(i>=5)a.push_back(ms);
      }
      report(name[mode],bytes,a);
    }
  }
  CUDA(cudaFree(gpu));CUDA(cudaFreeHost(host));CUDA(cudaStreamDestroy(stream));puts("COMPLETE");
}
