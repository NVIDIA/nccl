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
#define CU(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA line %d: %s\n",__LINE__,cudaGetErrorString(e));exit(2);} } while(0)
#define NC(x) do { ncclResult_t e=(x); if(e!=ncclSuccess){fprintf(stderr,"NCCL line %d: %s\n",__LINE__,ncclGetErrorString(e));exit(3);} } while(0)
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void io(int fd,void *buf,size_t n,int wr){char*p=(char*)buf;while(n){ssize_t k=wr?send(fd,p,n,MSG_NOSIGNAL):recv(fd,p,n,0);if(k<0&&errno==EINTR)continue;if(k<=0){perror("control socket");exit(4);}p+=k;n-=k;}}
static void barrier(int fd,int rank){char c=1;if(rank==0){io(fd,&c,1,1);io(fd,&c,1,0);}else{io(fd,&c,1,0);io(fd,&c,1,1);}}
static int cmp(const void*a,const void*b){double x=*(double*)a,y=*(double*)b;return(x>y)-(x<y);}
int main(int argc,char**argv){
 if(argc!=5){fprintf(stderr,"Usage: %s RANK RANK0_IPV4 PORT TIMED_ITERS\n",argv[0]);return 1;}
 setvbuf(stdout,NULL,_IOLBF,0);int rank=atoi(argv[1]),iters=atoi(argv[4]);if(rank<0||rank>1||iters<1)return 1;
 int fd=socket(AF_INET,SOCK_STREAM,0),one=1;setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));
 struct sockaddr_in a={};a.sin_family=AF_INET;a.sin_port=htons(atoi(argv[3]));if(inet_pton(AF_INET,argv[2],&a.sin_addr)!=1)return 1;
 if(rank==0){if(bind(fd,(struct sockaddr*)&a,sizeof(a))||listen(fd,1)){perror("bind/listen");return 4;}int s=accept(fd,NULL,NULL);close(fd);fd=s;}
 else{double deadline=now()+30;while(connect(fd,(struct sockaddr*)&a,sizeof(a))){if(now()>deadline){perror("connect");return 4;}usleep(100000);}}
 setsockopt(fd,IPPROTO_TCP,TCP_NODELAY,&one,sizeof(one));
 CU(cudaSetDevice(0));int cv,dv,nv;CU(cudaRuntimeGetVersion(&cv));CU(cudaDriverGetVersion(&dv));NC(ncclGetVersion(&nv));
 printf("ENV rank=%d pid=%d nccl=%d runtime=%d driver_api=%d\n",rank,getpid(),nv,cv,dv);
 ncclUniqueId id;if(rank==0){NC(ncclGetUniqueId(&id));io(fd,&id,sizeof(id),1);}else io(fd,&id,sizeof(id),0);
 ncclComm_t comm;NC(ncclCommInitRank(&comm,2,id,rank));cudaStream_t stream;CU(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 const size_t sizes[]={0,1,4095,4096,65535,65536,131072,1048576,4194304,8388608};
 for(int s=0;s<int(sizeof(sizes)/sizeof(sizes[0]));s++){
  size_t bytes=sizes[s],count=bytes;float*x,*ack;CU(cudaMalloc(&x,bytes?bytes:1));CU(cudaMalloc(&ack,4));
  double*ts=(double*)malloc(iters*sizeof(double));
  for(int i=0;i<iters+5;i++){
   unsigned char pattern=1+i%251;
   if(bytes) CU(cudaMemsetAsync(x,rank?0xa5:pattern,bytes,stream));
   CU(cudaMemsetAsync(ack,rank?pattern:0xa5,4,stream));
   barrier(fd,rank);CU(cudaDeviceSynchronize());printf("BEGIN rank=%d bytes=%zu iter=%d\n",rank,bytes,i);double start=now();
   if(rank==0){NC(ncclSend(x,count,ncclChar,1,comm,stream));NC(ncclRecv(ack,1,ncclFloat,1,comm,stream));}
   else{NC(ncclRecv(x,count,ncclChar,0,comm,stream));NC(ncclSend(ack,1,ncclFloat,0,comm,stream));}
   CU(cudaDeviceSynchronize());double ms=(now()-start)*1000;printf("ITER rank=%d bytes=%zu iter=%d ms=%.6f\n",rank,bytes,i,ms);if(i>=5)ts[i-5]=ms;
   unsigned char* verify=(unsigned char*)malloc(bytes?bytes:1);
   if(bytes) CU(cudaMemcpy(verify,x,bytes,cudaMemcpyDeviceToHost));
   for(size_t j=0;j<bytes;j++)if(verify[j]!=pattern){fprintf(stderr,"BAD DATA rank=%d bytes=%zu iter=%d offset=%zu got=%u expected=%u\n",rank,bytes,i,j,verify[j],pattern);return 5;}
   free(verify);unsigned char checkAck[4];CU(cudaMemcpy(checkAck,ack,4,cudaMemcpyDeviceToHost));
   for(int j=0;j<4;j++)if(checkAck[j]!=pattern){fprintf(stderr,"BAD ACK rank=%d iter=%d\n",rank,i);return 5;}

  }
  qsort(ts,iters,sizeof(double),cmp);double median=iters%2?ts[iters/2]:(ts[iters/2-1]+ts[iters/2])/2;
  printf("RESULT rank=%d bytes=%zu samples=%d median_ms=%.6f max_ms=%.6f validated=1\n",rank,bytes,iters,median,ts[iters-1]);
  free(ts);CU(cudaFree(x));CU(cudaFree(ack));
 }
 barrier(fd,rank);NC(ncclCommDestroy(comm));CU(cudaStreamDestroy(stream));close(fd);printf("COMPLETE rank=%d\n",rank);return 0;
}
