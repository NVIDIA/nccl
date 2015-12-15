/*************************************************************************
 * Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ************************************************************************/


#ifndef SRC_TEST_UTILITIES_H_
#define SRC_TEST_UTILITIES_H_

#include <curand.h>

#define CUDACHECK(cmd) do {                              \
    cudaError_t e = cmd;                                 \
    if( e != cudaSuccess ) {                             \
        printf("Cuda failure %s:%d '%s'\n",              \
               __FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                              \
    }                                                    \
} while(false)

template<typename T>
void Randomize(T* const dest, const int N, const int randomSeed);

template<typename T>
void Accumulate(T* dest, const T* contrib, int N, ncclRedOp_t op);

template<typename T>
double CheckDelta(const T* results, const T* expected, int N);

#define CURAND_CHK(cmd)                                                         \
    do {                                                                        \
      curandStatus_t error = (cmd);                                             \
      if (error != CURAND_STATUS_SUCCESS) {                                     \
        printf("CuRAND error %i at %s:%i\n", error, __FILE__ , __LINE__);       \
        exit(EXIT_FAILURE);                                                     \
      }                                                                         \
    } while (false)


template<typename T>
void GenerateRandom(curandGenerator_t generator, T * const dest,
    const int N);

template<>
void GenerateRandom<char>(curandGenerator_t generator, char * const dest,
    const int N) {
  CURAND_CHK(curandGenerate(generator, (unsigned int*)dest,
      N * sizeof(char) / sizeof(int)));
}

template<>
void GenerateRandom<int>(curandGenerator_t generator, int * const dest,
    const int N) {
  CURAND_CHK(curandGenerate(generator, (unsigned int*)dest, N));
}

template<>
void GenerateRandom<float>(curandGenerator_t generator, float * const dest,
    const int N) {
  CURAND_CHK(curandGenerateUniform(generator, dest, N));
}

template<>
void GenerateRandom<double>(curandGenerator_t generator, double * const dest,
    const int N) {
  CURAND_CHK(curandGenerateUniformDouble(generator, dest, N));
}

template<>
void GenerateRandom<unsigned long long>(curandGenerator_t generator, unsigned long long * const dest,
    const int N) {
  CURAND_CHK(curandGenerateLongLong(generator, dest, N));
}


template<typename T>
void Randomize(T* const dest, const int N, const int randomSeed) {
  curandGenerator_t gen;
  CURAND_CHK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937));
  CURAND_CHK(curandSetPseudoRandomGeneratorSeed(gen, randomSeed));
  GenerateRandom<T>(gen, dest, N);
  CURAND_CHK(curandDestroyGenerator(gen));
  CUDACHECK(cudaDeviceSynchronize());
}

template<>
void Randomize(unsigned long long* const dest, const int N, const int randomSeed) {
  curandGenerator_t gen;
  CURAND_CHK(curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL64));
  GenerateRandom<unsigned long long>(gen, dest, N);
  CURAND_CHK(curandDestroyGenerator(gen));
  CUDACHECK(cudaDeviceSynchronize());
}

template<>
void Randomize(long long* const dest, const int N, const int randomSeed) {
  curandGenerator_t gen;
  CURAND_CHK(curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL64));
  GenerateRandom<unsigned long long>(gen, (unsigned long long *)dest, N);
  CURAND_CHK(curandDestroyGenerator(gen));
  CUDACHECK(cudaDeviceSynchronize());
}

#ifdef CUDA_HAS_HALF
__global__ void halve(const float * src, half* dest, int N) {
  for(int tid = threadIdx.x + blockIdx.x*blockDim.x;
      tid < N; tid += blockDim.x * gridDim.x)
    dest[tid] = __float2half(src[tid]);
}

template<>
void Randomize<half>(half* const dest, const int N, const int randomSeed) {
  curandGenerator_t gen;
  CURAND_CHK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937));
  CURAND_CHK(curandSetPseudoRandomGeneratorSeed(gen, randomSeed));

  float* temp;
  CUDACHECK(cudaMalloc(&temp, N*sizeof(float)));
  GenerateRandom<float>(gen, temp, N);
  halve<<<128, 512>>>(temp, dest, N);
  CURAND_CHK(curandDestroyGenerator(gen));
  CUDACHECK(cudaFree(temp));
  CUDACHECK(cudaDeviceSynchronize());
}
#endif

template<typename T, int OP> __global__ static
void accumKern(T* acum, const T* contrib, int N) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int offset = blockDim.x*gridDim.x;
  for(int i=tid; i<N; i+=offset) {
    T c = contrib[i];
    T a = acum[i];
    if(OP == ncclSum) {
      acum[i] = a+c;
    } else if(OP == ncclProd) {
      acum[i] = a*c;
    } else if(OP == ncclMax) {
      acum[i] = (a > c) ? a : c;
    } else if(OP == ncclMin) {
      acum[i] = (a < c) ? a : c;
    }
  }
}

#ifdef CUDA_HAS_HALF
template<> __global__
void accumKern<half, ncclSum>(half* acum, const half* contrib, int N) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int offset = blockDim.x*gridDim.x;
  for(int i=tid; i<N; i+=offset) {
    float c = __half2float(contrib[i]);
    float a = __half2float(acum[i]);
    acum[i] = __float2half( a + c );
  }
}

template<> __global__
void accumKern<half, ncclProd>(half* acum, const half* contrib, int N) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int offset = blockDim.x*gridDim.x;
  for(int i=tid; i<N; i+=offset) {
    float c = __half2float(contrib[i]);
    float a = __half2float(acum[i]);
    acum[i] = __float2half( a * c );
  }
}

template<> __global__
void accumKern<half, ncclMax>(half* acum, const half* contrib, int N) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int offset = blockDim.x*gridDim.x;
  for(int i=tid; i<N; i+=offset) {
    float c = __half2float(contrib[i]);
    float a = __half2float(acum[i]);
    acum[i] = __float2half( (a>c) ? a : c );
  }
}

template<> __global__
void accumKern<half, ncclMin>(half* acum, const half* contrib, int N) {
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  int offset = blockDim.x*gridDim.x;
  for(int i=tid; i<N; i+=offset) {
    float c = __half2float(contrib[i]);
    float a = __half2float(acum[i]);
    acum[i] = __float2half( (a<c) ? a : c );
  }
}
#endif

template<typename T>
void Accumulate(T* dest, const T* contrib, int N, ncclRedOp_t op) {

  T* devdest;
  CUDACHECK(cudaHostRegister(dest, N*sizeof(T), 0));
  CUDACHECK(cudaHostGetDevicePointer(&devdest, dest, 0));
  switch(op) {
    case ncclSum:  accumKern<T, ncclSum> <<<256,256>>>(devdest, contrib, N); break;
    case ncclProd: accumKern<T, ncclProd><<<256,256>>>(devdest, contrib, N); break;
    case ncclMax:  accumKern<T, ncclMax> <<<256,256>>>(devdest, contrib, N); break;
    case ncclMin:  accumKern<T, ncclMin> <<<256,256>>>(devdest, contrib, N); break;
    default:
      printf("Unknown reduction operation.\n");
      exit(EXIT_FAILURE);
  }
  CUDACHECK(cudaHostUnregister(dest));
}

template<typename T> __device__
double absDiff(T a, T b) {
  return fabs((double)(b - a));
}

#ifdef CUDA_HAS_HALF
template<> __device__
double absDiff<half>(half a, half b) {
  float x = __half2float(a);
  float y = __half2float(b);
  return fabs((double)(y-x));
}
#endif

template<typename T, int BSIZE> __global__
void deltaKern(const T* A, const T* B, int N, double* max) {
  __shared__ double temp[BSIZE];
  int tid = threadIdx.x;
  double locmax = 0.0;
  for(int i=tid; i<N; i+=blockDim.x) {
    
    double delta = absDiff(A[i], B[i]);
    if( delta > locmax )
      locmax = delta;
  }

  temp[tid] = locmax;
  for(int stride = BSIZE/2; stride > 1; stride>>=1) {
    __syncthreads();
    if( tid < stride )
      temp[tid] = temp[tid] > temp[tid+stride] ? temp[tid] : temp[tid+stride];
  }
  __syncthreads();
  if( threadIdx.x == 0)
    *max = temp[0] > temp[1] ? temp[0] : temp[1];
}

template<typename T>
double CheckDelta(const T* results, const T* expected, int N) {
  T* devexp;
  double maxerr;
  double* devmax;
  CUDACHECK(cudaHostRegister((void*)expected, N*sizeof(T), 0));
  CUDACHECK(cudaHostGetDevicePointer((void**)&devexp, (void*)expected, 0));
  CUDACHECK(cudaHostRegister((void*)&maxerr, sizeof(double), 0));
  CUDACHECK(cudaHostGetDevicePointer(&devmax, &maxerr, 0));
  deltaKern<T, 512><<<1, 512>>>(results, devexp, N, devmax);
  CUDACHECK(cudaHostUnregister(&maxerr));
  CUDACHECK(cudaHostUnregister((void*)devexp));
  return maxerr;
}


std::string TypeName(const ncclDataType_t type) {
  switch (type) {
    case ncclChar:   return "char";
    case ncclInt:    return "int";
#ifdef CUDA_HAS_HALF
    case ncclHalf:   return "half";
#endif
    case ncclFloat:  return "float";
    case ncclDouble: return "double";
    case ncclInt64:  return "int64";
    case ncclUint64: return "uint64";
    default:         return "unknown";
  }
}

std::string OperationName(const ncclRedOp_t op) {
  switch (op) {
    case ncclSum:  return "sum";
    case ncclProd: return "prod";
    case ncclMax:  return "max";
    case ncclMin:  return "min";
    default:       return "unknown";
  }
}


#endif // SRC_TEST_UTILITIES_H_
