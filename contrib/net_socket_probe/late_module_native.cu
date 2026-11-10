// Two hosts, one rank/GPU each. No framework, MPI, graphs or user registration.
#include <cuda_runtime.h>
#include <nccl.h>
#include <cuda.h>
#include <vector>
#include <fstream>
#include <iterator>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CU(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA line %d: %s\n",__LINE__,cudaGetErrorString(e));exit(2);} } while(0)
#define NC(x) do { ncclResult_t e=(x); if(e!=ncclSuccess){fprintf(stderr,"NCCL line %d: %s\n",__LINE__,ncclGetErrorString(e));exit(3);} } while(0)
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void io(int fd,void *buf,size_t n,int wr){char*p=(char*)buf;while(n){ssize_t k=wr?send(fd,p,n,MSG_NOSIGNAL):recv(fd,p,n,0);if(k<0&&errno==EINTR)continue;if(k<=0){perror("control socket");exit(4);}p+=k;n-=k;}}
static void barrier(int fd,int rank){char c=1;if(rank==0){io(fd,&c,1,1);io(fd,&c,1,0);}else{io(fd,&c,1,0);io(fd,&c,1,1);}}
static int cmp(const void*a,const void*b){double x=*(double*)a,y=*(double*)b;return(x>y)-(x<y);}
int main(int argc,char**argv){
 if(argc!=7){fprintf(stderr,"Usage: %s RANK HOST PORT CUBIN BLOCK PRELOAD\n",argv[0]);return 1;}
 setvbuf(stdout,NULL,_IOLBF,0);int rank=atoi(argv[1]),block=atoi(argv[5]),preload=atoi(argv[6]);
 int fd=socket(AF_INET,SOCK_STREAM,0),one=1;setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));
 struct sockaddr_in a={};a.sin_family=AF_INET;a.sin_port=htons(atoi(argv[3]));inet_pton(AF_INET,argv[2],&a.sin_addr);
 if(rank==0){if(bind(fd,(struct sockaddr*)&a,sizeof(a))||listen(fd,1))return 4;int s=accept(fd,NULL,NULL);close(fd);fd=s;}
 else{double until=now()+20;while(connect(fd,(struct sockaddr*)&a,sizeof(a))){if(now()>until)return 4;usleep(10000);}}
 CU(cudaSetDevice(0));int cv,dv;CU(cudaRuntimeGetVersion(&cv));CU(cudaDriverGetVersion(&dv));printf("ENV rank=%d runtime=%d driver_api=%d\n",rank,cv,dv);
 ncclUniqueId id;if(rank==0){NC(ncclGetUniqueId(&id));io(fd,&id,sizeof(id),1);}else io(fd,&id,sizeof(id),0);
 ncclComm_t comm;NC(ncclCommInitRank(&comm,2,id,rank));cudaStream_t stream;CU(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 int*x;CU(cudaMalloc(&x,65536));CU(cudaMemset(x,0x2a,65536));
 std::ifstream file(argv[4],std::ios::binary);std::vector<char> bytes((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());if(bytes.empty())return 6;
 CUmodule module;CUfunction func;
 auto load=[&](){puts("MODULE_LOAD_BEGIN");if(cuModuleLoadData(&module,bytes.data())!=CUDA_SUCCESS)exit(7);if(cuModuleGetFunction(&func,module,"late_kernel")!=CUDA_SUCCESS)exit(7);puts("MODULE_LOAD_END");};
 if(preload)load();
 if(rank==0){NC(ncclSend(x,16384,ncclInt,1,comm,stream));}else{NC(ncclRecv(x,16384,ncclInt,0,comm,stream));}
 CU(cudaStreamSynchronize(stream));barrier(fd,rank);puts("WARMUP_DONE");
 if(rank==0){usleep(1000000);puts("SEND_BEGIN");NC(ncclSend(x,16384,ncclInt,1,comm,stream));CU(cudaStreamSynchronize(stream));puts("SEND_DONE");}
 else{puts("RECV_BEGIN");NC(ncclRecv(x,16384,ncclInt,0,comm,stream));if(block)CU(cudaStreamSynchronize(stream));puts("RECV_API_RETURNED");if(!preload)load();CU(cudaStreamSynchronize(stream));std::vector<int> host(16384);CU(cudaMemcpy(host.data(),x,65536,cudaMemcpyDeviceToHost));for(int v:host)if(v!=0x2a2a2a2a)return 8;puts("VALIDATED");}
 barrier(fd,rank);NC(ncclCommDestroy(comm));CU(cudaFree(x));CU(cudaStreamDestroy(stream));close(fd);puts("COMPLETE");return 0;
}
