---
name: NCCL - NVIDIA Collective Communications Library
description: NCCL (pronounced "Nickel") is NVIDIA's high-performance multi-GPU and multi-node communication library. It implements optimized collective operations (all-reduce, all-gather, broadcast, etc.) and point-to-point primitives for efficient GPU-to-GPU communication across PCIe, NVLink, InfiniBand, and TCP/IP networks.
---

## Quick Start

```bash
# Install NCCL from NVIDIA repository (Ubuntu)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install libnccl2 libnccl-dev

# Verify installation
ldconfig -p | grep nccl

# Clone and run NCCL tests
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 4
```

## When to Use This Skill

Use NCCL when you need to:
- Implement distributed deep learning training across multiple GPUs
- Perform efficient all-reduce operations for gradient synchronization
- Build multi-GPU applications with collective communication
- Optimize data parallelism in machine learning frameworks
- Achieve maximum bandwidth for GPU-to-GPU transfers
- Support multi-node training with InfiniBand or RoCE networks
- Implement custom distributed algorithms requiring collectives
- Replace MPI with GPU-optimized communication primitives

## Prerequisites

**Platform**: Linux (x86_64, aarch64, ppc64le)

**Required Dependencies**:
- NVIDIA GPU with Compute Capability 3.5+ (Kepler or newer)
- CUDA Toolkit 9.0 or later (11.0+ recommended)
- NVIDIA Driver 418+ (470+ recommended)
- GCC 4.8+ or compatible C++ compiler

**Optional Dependencies**:
- InfiniBand Verbs (libibverbs) for RDMA support
- RDMA-capable NICs (Mellanox ConnectX-5/6/7)
- UCX (Unified Communication X) for advanced RDMA features
- hwloc for topology detection
- libxml2 for configuration parsing

**Network Requirements**:
- PCIe 3.0+ (for intra-node)
- NVLink (recommended for intra-node)
- InfiniBand or RoCE (for inter-node)
- High-speed Ethernet (25GbE+) as alternative

## Compatibility

| NCCL Version | CUDA Version | GPU Architecture | Network Support | Notes |
|--------------|--------------|------------------|-----------------|-------|
| 2.20+        | 11.0 - 12.x  | Kepler - Hopper  | PCIe, NVLink, IB, TCP | Latest stable |
| 2.18+        | 11.0 - 12.x  | Kepler - Hopper  | PCIe, NVLink, IB, TCP | Wide adoption |
| 2.15+        | 11.0 - 12.x  | Kepler - Hopper  | PCIe, NVLink, IB, TCP | LTS candidate |
| 2.12+        | 10.2 - 11.x  | Kepler - Ampere  | PCIe, NVLink, IB, TCP | Legacy support |

**Supported Communication Primitives**:
- AllReduce, AllGather, ReduceScatter
- Broadcast, Reduce
- AllToAll, Gather, Scatter
- Send/Recv (point-to-point)

## Installation

### From NVIDIA Repository (Recommended)

#### Ubuntu/Debian

```bash
# Add NVIDIA CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install NCCL runtime and development packages
sudo apt-get install libnccl2 libnccl-dev

# Verify installation
dpkg -l | grep nccl
```

#### RedHat/CentOS/Fedora

```bash
# Add NVIDIA CUDA repository
sudo dnf config-manager --add-repo \
    https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

# Install NCCL
sudo dnf install libnccl libnccl-devel

# Verify
rpm -qa | grep nccl
```

### From Tarball (Portable)

```bash
# Download from developer.nvidia.com/nccl
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nccl-local-repo-ubuntu2204-2.20.5-cuda12.0_1.0-1_amd64.deb

# Or use prebuilt tarball
tar xvf nccl_2.20.5-1+cuda12.0_x86_64.txz

# Copy libraries
sudo cp -r nccl_2.20.5-1+cuda12.0_x86_64/lib/* /usr/local/lib/
sudo cp -r nccl_2.20.5-1+cuda12.0_x86_64/include/* /usr/local/include/

# Update library cache
sudo ldconfig
```

### Build from Source

```bash
# Clone repository
git clone https://github.com/NVIDIA/nccl.git
cd nccl

# Build for specific GPU architectures (reduces build time)
make -j src.build NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 \
                                -gencode=arch=compute_90,code=sm_90"

# Or build for all architectures
make -j src.build

# Install
sudo make install

# Verify
ls -l /usr/local/lib/libnccl*
```

### Build with Customizations

```bash
# Enable debug symbols
make -j src.build DEBUG=1

# Build with specific CUDA path
make -j src.build CUDA_HOME=/usr/local/cuda-12.0

# Build static library
make -j src.build BUILDDIR=/tmp/nccl-static

# Cross-compile for different architecture
make -j src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
```

## Configuration

### Environment Variables

NCCL behavior is extensively configurable through environment variables:

#### Core Configuration

```bash
# Logging and debugging
export NCCL_DEBUG=INFO              # WARN, INFO, TRACE (verbose)
export NCCL_DEBUG_SUBSYS=ALL        # INIT, COLL, P2P, SHM, NET, GRAPH, TUNING
export NCCL_DEBUG_FILE=/tmp/nccl-%h-%p.log  # Log to file (%h=host, %p=pid)

# GPU selection
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Visible GPUs
export NCCL_IGNORE_CPU_AFFINITY=1    # Disable CPU affinity checking

# Communication parameters
export NCCL_BUFFSIZE=4194304         # Buffer size (4MB default)
export NCCL_NTHREADS=512            # Threads per NCCL comm (default auto)
export NCCL_MAX_NCHANNELS=16        # Max channels per operation
export NCCL_MIN_NCHANNELS=1         # Min channels per operation
```

#### Network Configuration

```bash
# Network interface selection
export NCCL_SOCKET_IFNAME=eth0      # TCP/IP interface
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1  # InfiniBand HCAs (port 1)
export NCCL_NET_GDR_LEVEL=2         # GPUDirect RDMA (0=off, 5=full)

# Network protocol tuning
export NCCL_IB_DISABLE=0            # Enable InfiniBand (0=enable, 1=disable)
export NCCL_P2P_DISABLE=0           # Enable P2P (NVLink/PCIe)
export NCCL_SHM_DISABLE=0           # Enable shared memory
export NCCL_NET_GDR_READ=1          # Enable GDR for reads

# InfiniBand specific
export NCCL_IB_TIMEOUT=22           # IB timeout (default 18)
export NCCL_IB_RETRY_CNT=7          # IB retry count (default 7)
export NCCL_IB_QPS_PER_CONN=4       # QPs per connection (default 1)
export NCCL_IB_TC=106               # Traffic class for RoCE
export NCCL_IB_SL=0                 # Service level

# TCP/IP tuning
export NCCL_SOCKET_NTHREADS=4       # Socket threads
export NCCL_NSOCKS_PERTHREAD=4      # Sockets per thread
```

#### Performance Tuning

```bash
# Algorithm selection
export NCCL_ALGO=Ring               # Ring, Tree, or Auto
export NCCL_PROTO=Simple            # Simple, LL, or LL128
export NCCL_CROSS_NIC=2             # Cross-NIC communication strategy

# Memory and timing
export NCCL_LL_THRESHOLD=16384      # Low-latency threshold
export NCCL_GRAPH_MIXING_SUPPORT=0  # Graph mixing (0=off)
export NCCL_BUFFSIZE=8388608        # Increase buffer for large messages

# Topology
export NCCL_TOPO_FILE=/path/to/topo.xml  # Custom topology file
export NCCL_IGNORE_DISABLED_P2P=0   # Respect disabled P2P links
export NCCL_P2P_LEVEL=SYS           # P2P level: NVL, PIX, PXB, PHB, SYS

# SHARP (Scalable Hierarchical Aggregation and Reduction Protocol)
export NCCL_COLLNET_ENABLE=1        # Enable SHARP (if available)
export NCCL_COLLNET_NODE_THRESHOLD=2  # Min nodes for SHARP
```

#### Multi-Node Configuration

```bash
# Distributed setup
export NCCL_SOCKET_IFNAME=eth0      # Network interface for bootstrap
export NCCL_IB_HCA=mlx5_0:1         # InfiniBand adapter

# Coordinated launch (rank-based)
export NCCL_COMM_ID=tcp://master-node:12345  # Rendezvous endpoint
export RANK=0                       # Current rank
export WORLD_SIZE=8                 # Total ranks

# Or use MPI
mpirun -np 8 -H node1:4,node2:4 \
       -x NCCL_DEBUG=INFO \
       -x NCCL_IB_HCA=mlx5_0 \
       ./my_app
```

## Usage Patterns

### Basic All-Reduce Example

```c
#include <nccl.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    int nDevices = 4;
    ncclComm_t comms[4];
    cudaStream_t streams[4];
    float **sendbuff, **recvbuff;

    // Allocate device buffers
    sendbuff = (float**)malloc(nDevices * sizeof(float*));
    recvbuff = (float**)malloc(nDevices * sizeof(float*));

    for (int i = 0; i < nDevices; i++) {
        cudaSetDevice(i);
        cudaMalloc(&sendbuff[i], 1024 * 1024 * sizeof(float));
        cudaMalloc(&recvbuff[i], 1024 * 1024 * sizeof(float));
        cudaStreamCreate(&streams[i]);

        // Initialize with rank ID
        float init_val = (float)i;
        cudaMemset(sendbuff[i], init_val, 1024 * 1024 * sizeof(float));
    }

    // Initialize NCCL communicators
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    ncclGroupStart();
    for (int i = 0; i < nDevices; i++) {
        cudaSetDevice(i);
        ncclCommInitRank(&comms[i], nDevices, id, i);
    }
    ncclGroupEnd();

    // Perform AllReduce
    ncclGroupStart();
    for (int i = 0; i < nDevices; i++) {
        ncclAllReduce(sendbuff[i], recvbuff[i],
                     1024 * 1024, ncclFloat, ncclSum,
                     comms[i], streams[i]);
    }
    ncclGroupEnd();

    // Synchronize
    for (int i = 0; i < nDevices; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
    }

    // Cleanup
    for (int i = 0; i < nDevices; i++) {
        cudaSetDevice(i);
        ncclCommDestroy(comms[i]);
        cudaFree(sendbuff[i]);
        cudaFree(recvbuff[i]);
        cudaStreamDestroy(streams[i]);
    }

    printf("AllReduce completed successfully\n");
    return 0;
}
```

**Compile and run:**
```bash
nvcc -o all_reduce all_reduce.cu -lnccl
./all_reduce
```

### Multi-Process (MPI) Pattern

```c
#include <nccl.h>
#include <mpi.h>
#include <cuda_runtime.h>

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set device based on local rank
    int localRank = rank % 4;  // Assuming 4 GPUs per node
    cudaSetDevice(localRank);

    // Create NCCL unique ID on rank 0 and broadcast
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Initialize NCCL communicator
    ncclComm_t comm;
    ncclCommInitRank(&comm, size, id, rank);

    // Allocate device memory
    float *sendbuff, *recvbuff;
    size_t count = 1024 * 1024;
    cudaMalloc(&sendbuff, count * sizeof(float));
    cudaMalloc(&recvbuff, count * sizeof(float));

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Perform AllReduce
    ncclAllReduce(sendbuff, recvbuff, count, ncclFloat,
                 ncclSum, comm, stream);

    cudaStreamSynchronize(stream);

    // Cleanup
    cudaFree(sendbuff);
    cudaFree(recvbuff);
    cudaStreamDestroy(stream);
    ncclCommDestroy(comm);

    MPI_Finalize();
    return 0;
}
```

**Compile and run:**
```bash
nvcc -o mpi_allreduce mpi_allreduce.cu -lnccl
mpicc -o mpi_allreduce mpi_allreduce.cu -lnccl -lcudart

mpirun -np 8 -hostfile hosts.txt \
       -x NCCL_DEBUG=INFO \
       -x NCCL_IB_HCA=mlx5_0 \
       ./mpi_allreduce
```

### Point-to-Point Communication

```c
#include <nccl.h>

// Send from rank 0 to rank 1
void p2p_communication(ncclComm_t* comms, cudaStream_t* streams) {
    float *sendbuff, *recvbuff;
    size_t count = 1024 * 1024;

    // Rank 0: allocate and send
    if (rank == 0) {
        cudaSetDevice(0);
        cudaMalloc(&sendbuff, count * sizeof(float));
        ncclSend(sendbuff, count, ncclFloat, 1, comms[0], streams[0]);
    }

    // Rank 1: allocate and receive
    if (rank == 1) {
        cudaSetDevice(1);
        cudaMalloc(&recvbuff, count * sizeof(float));
        ncclRecv(recvbuff, count, ncclFloat, 0, comms[1], streams[1]);
    }

    // Synchronize
    cudaStreamSynchronize(streams[rank]);
}
```

### Grouped Operations

```c
// Perform multiple operations atomically
ncclGroupStart();

// Operation 1: AllReduce on GPU 0
ncclAllReduce(sendbuff0, recvbuff0, count, ncclFloat,
             ncclSum, comms[0], streams[0]);

// Operation 2: Broadcast from GPU 1
ncclBroadcast(sendbuff1, recvbuff1, count, ncclFloat,
             1, comms[1], streams[1]);

// Operation 3: AllGather on GPU 2
ncclAllGather(sendbuff2, recvbuff2, count, ncclFloat,
             comms[2], streams[2]);

ncclGroupEnd();  // All operations launched together
```

### Error Handling

```c
#include <nccl.h>

const char* ncclGetErrorString(ncclResult_t result);

void check_nccl(ncclResult_t result, const char* call) {
    if (result != ncclSuccess) {
        fprintf(stderr, "NCCL error at %s: %s\n",
                call, ncclGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

// Usage
ncclResult_t result = ncclAllReduce(...);
check_nccl(result, "ncclAllReduce");

// Asynchronous error checking
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm,
                                   ncclResult_t* asyncError);
```

## Key Features

- **High Performance**: Optimized kernels for maximum bandwidth utilization
- **Topology Aware**: Automatic detection and optimization for PCIe, NVLink, InfiniBand
- **Multiple Algorithms**: Ring, Tree, and hybrid algorithms for different scales
- **Protocol Variants**: Simple, Low-Latency (LL), and LL128 protocols
- **GPUDirect Support**: RDMA and GPUDirect Storage integration
- **MPI Compatible**: Drop-in replacement for MPI collectives
- **Zero-Copy**: Direct GPU-to-GPU transfers without CPU involvement
- **Fault Tolerance**: Connection recovery and error detection

## Performance Optimization

### Best Practices

1. **Use Group Operations**
   ```c
   // Good: All operations launch together
   ncclGroupStart();
   for (int i = 0; i < nGPUs; i++) {
       ncclAllReduce(..., comms[i], streams[i]);
   }
   ncclGroupEnd();

   // Bad: Sequential launches add overhead
   for (int i = 0; i < nGPUs; i++) {
       ncclAllReduce(..., comms[i], streams[i]);
   }
   ```

2. **Optimize Buffer Sizes**
   ```bash
   # For large messages (>1MB)
   export NCCL_BUFFSIZE=8388608  # 8MB

   # For small messages (<1KB)
   export NCCL_LL_THRESHOLD=0    # Always use LL
   ```

3. **Enable GPUDirect RDMA**
   ```bash
   export NCCL_NET_GDR_LEVEL=5    # Maximum GDR
   export NCCL_P2P_LEVEL=NVL      # Prefer NVLink
   export NCCL_IB_GID_INDEX=3     # RoCE GID
   ```

4. **Tune for Multi-Node**
   ```bash
   # Use multiple rails
   export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1

   # Increase QPs per connection
   export NCCL_IB_QPS_PER_CONN=4

   # Enable SHARP if available
   export NCCL_COLLNET_ENABLE=1
   ```

5. **Profile and Debug**
   ```bash
   # Detailed logging
   export NCCL_DEBUG=TRACE
   export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV,TUNING

   # Use NCCL tests for benchmarking
   ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8
   ```

6. **NUMA Awareness**
   ```bash
   # Pin processes to NUMA nodes
   numactl --cpunodebind=0 --membind=0 -- ./app &
   numactl --cpunodebind=1 --membind=1 -- ./app &
   ```

### Expected Performance

| Configuration | Operation | Size | Bandwidth | Latency | Notes |
|--------------|-----------|------|-----------|---------|-------|
| 8x A100 (NVLink) | AllReduce | 128MB | 280 GB/s | 120 μs | NVSwitch |
| 8x H100 (NVLink) | AllReduce | 128MB | 450 GB/s | 90 μs | NVLink 4.0 |
| 4x A100 (PCIe) | AllReduce | 128MB | 80 GB/s | 250 μs | PCIe 4.0 x16 |
| 16x V100 (IB) | AllReduce | 128MB | 190 GB/s | 180 μs | 2x IB HDR |
| 2 nodes x 8 GPUs | AllReduce | 1GB | 160 GB/s | 2 ms | 8x IB HDR |

**Small Message Performance:**
- 4KB AllReduce: ~20 μs (NVLink), ~50 μs (IB)
- 32KB AllReduce: ~40 μs (NVLink), ~80 μs (IB)

**Scaling Efficiency:**
- 8 GPUs: 95-98% of peak bandwidth
- 16 GPUs: 90-95% of peak bandwidth
- 64 GPUs: 85-92% of peak bandwidth
- 128 GPUs: 80-88% of peak bandwidth

## Use Cases

1. **Distributed Deep Learning**: Gradient synchronization across GPUs
2. **Data Parallelism**: Model training with synchronized updates
3. **HPC Applications**: Scientific computing with GPU clusters
4. **Federated Learning**: Aggregating updates from multiple nodes
5. **Distributed Inference**: Model parallelism for large models
6. **Parameter Servers**: Efficient parameter distribution
7. **MapReduce on GPUs**: Collective reductions for analytics
8. **Multi-GPU Simulations**: Data exchange in physics simulations

## Examples

### Example 1: Complete Multi-GPU Training Skeleton

```c
#include <nccl.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define NCCL_CHECK(cmd) do {                          \
    ncclResult_t r = cmd;                             \
    if (r != ncclSuccess) {                           \
        printf("NCCL error %s:%d '%s'\n",             \
            __FILE__, __LINE__, ncclGetErrorString(r)); \
        exit(EXIT_FAILURE);                           \
    }                                                 \
} while(0)

#define CUDA_CHECK(cmd) do {                          \
    cudaError_t e = cmd;                              \
    if (e != cudaSuccess) {                           \
        printf("CUDA error %s:%d '%s'\n",             \
            __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                           \
    }                                                 \
} while(0)

void train_epoch(ncclComm_t* comms, cudaStream_t* streams,
                int nGPUs, int epoch) {
    size_t model_size = 1024 * 1024;  // 1M parameters
    float **gradients, **weights;

    gradients = (float**)malloc(nGPUs * sizeof(float*));
    weights = (float**)malloc(nGPUs * sizeof(float*));

    // Allocate per-GPU buffers
    for (int i = 0; i < nGPUs; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaMalloc(&gradients[i], model_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&weights[i], model_size * sizeof(float)));
    }

    // Training loop
    int num_batches = 1000;
    for (int batch = 0; batch < num_batches; batch++) {
        // Forward pass (not shown)
        // Backward pass (not shown)
        // ... compute gradients ...

        // AllReduce gradients across all GPUs
        ncclGroupStart();
        for (int i = 0; i < nGPUs; i++) {
            NCCL_CHECK(ncclAllReduce(
                gradients[i], gradients[i], model_size,
                ncclFloat, ncclSum, comms[i], streams[i]
            ));
        }
        ncclGroupEnd();

        // Update weights (not shown)
        // ... apply optimizer ...

        if (batch % 100 == 0) {
            // Synchronize to check progress
            for (int i = 0; i < nGPUs; i++) {
                CUDA_CHECK(cudaStreamSynchronize(streams[i]));
            }
            printf("Epoch %d, Batch %d completed\n", epoch, batch);
        }
    }

    // Cleanup
    for (int i = 0; i < nGPUs; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaFree(gradients[i]));
        CUDA_CHECK(cudaFree(weights[i]));
    }
    free(gradients);
    free(weights);
}

int main(int argc, char* argv[]) {
    int nGPUs = 4;
    ncclComm_t comms[4];
    cudaStream_t streams[4];

    // Initialize NCCL
    ncclUniqueId id;
    NCCL_CHECK(ncclGetUniqueId(&id));

    ncclGroupStart();
    for (int i = 0; i < nGPUs; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        NCCL_CHECK(ncclCommInitRank(&comms[i], nGPUs, id, i));
    }
    ncclGroupEnd();

    // Train for multiple epochs
    int num_epochs = 10;
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        train_epoch(comms, streams, nGPUs, epoch);
    }

    // Cleanup
    for (int i = 0; i < nGPUs; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        NCCL_CHECK(ncclCommDestroy(comms[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    printf("Training completed successfully\n");
    return 0;
}
```

### Example 2: Multi-Node with MPI

```c
#include <nccl.h>
#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Determine local GPU
    int nLocalGPUs = 4;  // GPUs per node
    int localRank = rank % nLocalGPUs;
    cudaSetDevice(localRank);

    printf("MPI Rank %d using GPU %d\n", rank, localRank);

    // Create NCCL unique ID
    ncclUniqueId id;
    if (rank == 0) {
        ncclGetUniqueId(&id);
    }
    MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Initialize NCCL communicator
    ncclComm_t comm;
    ncclCommInitRank(&comm, size, id, rank);

    // Allocate buffers
    size_t count = 32 * 1024 * 1024;  // 32M floats = 128MB
    float *sendbuff, *recvbuff;
    cudaMalloc(&sendbuff, count * sizeof(float));
    cudaMalloc(&recvbuff, count * sizeof(float));

    // Initialize data
    cudaMemset(sendbuff, rank, count * sizeof(float));

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Benchmark AllReduce
    int warmup_iters = 10;
    int benchmark_iters = 100;

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        ncclAllReduce(sendbuff, recvbuff, count, ncclFloat,
                     ncclSum, comm, stream);
    }
    cudaStreamSynchronize(stream);
    MPI_Barrier(MPI_COMM_WORLD);

    // Benchmark
    double start = MPI_Wtime();
    for (int i = 0; i < benchmark_iters; i++) {
        ncclAllReduce(sendbuff, recvbuff, count, ncclFloat,
                     ncclSum, comm, stream);
    }
    cudaStreamSynchronize(stream);
    double end = MPI_Wtime();

    double elapsed = (end - start) / benchmark_iters;
    double data_size_gb = (count * sizeof(float)) / 1e9;
    double bandwidth = data_size_gb / elapsed;

    if (rank == 0) {
        printf("AllReduce bandwidth: %.2f GB/s\n", bandwidth);
        printf("Latency: %.2f ms\n", elapsed * 1000);
    }

    // Cleanup
    cudaFree(sendbuff);
    cudaFree(recvbuff);
    cudaStreamDestroy(stream);
    ncclCommDestroy(comm);
    MPI_Finalize();

    return 0;
}
```

**Compile and run:**
```bash
mpicc -o nccl_mpi_benchmark nccl_mpi_benchmark.c -lnccl -lcudart -L/usr/local/cuda/lib64

# Run on 2 nodes with 4 GPUs each
mpirun -np 8 \
       -host node1:4,node2:4 \
       -x NCCL_DEBUG=INFO \
       -x NCCL_IB_HCA=mlx5_0:1,mlx5_1:1 \
       -x NCCL_SOCKET_IFNAME=eth0 \
       ./nccl_mpi_benchmark
```

### Example 3: Python with PyTorch Integration

```python
import torch
import torch.distributed as dist
import os

def setup_nccl(rank, world_size):
    """Initialize NCCL backend for PyTorch"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # NCCL configuration
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_IB_HCA'] = 'mlx5_0'
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'

    # Initialize process group with NCCL backend
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

def distributed_training(rank, world_size):
    """Distributed training with NCCL"""
    setup_nccl(rank, world_size)

    # Set device
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # Create model
    model = torch.nn.Linear(1000, 1000).to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        output_device=rank
    )

    # Training loop
    for epoch in range(10):
        # Forward pass
        inputs = torch.randn(32, 1000, device=device)
        outputs = model(inputs)
        loss = outputs.sum()

        # Backward pass (gradients automatically all-reduced by DDP)
        loss.backward()

        # Synchronize before next iteration
        torch.cuda.synchronize()

        if rank == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    import torch.multiprocessing as mp

    world_size = 4  # 4 GPUs
    mp.spawn(distributed_training,
             args=(world_size,),
             nprocs=world_size,
             join=True)
```

### Example 4: Topology Detection and Tuning

```bash
#!/bin/bash
# detect_and_tune_nccl.sh - Automatic NCCL tuning based on topology

set -e

echo "=== NCCL Topology Detection and Tuning ==="

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Check NVLink connectivity
echo -e "\n=== NVLink Topology ==="
nvidia-smi topo -m

# Detect InfiniBand adapters
if [ -d /sys/class/infiniband ]; then
    echo -e "\n=== InfiniBand Adapters ==="
    ls /sys/class/infiniband/
    IB_DEVICES=$(ls /sys/class/infiniband/ | tr '\n' ',' | sed 's/,$//')
    export NCCL_IB_HCA="$IB_DEVICES"
    echo "Using IB devices: $NCCL_IB_HCA"
else
    echo "No InfiniBand adapters detected"
    export NCCL_IB_DISABLE=1
fi

# Detect network interfaces
echo -e "\n=== Network Interfaces ==="
FAST_NIC=$(ip link show | grep -E "mlx|eth" | grep "state UP" | head -1 | cut -d: -f2 | xargs)
if [ -n "$FAST_NIC" ]; then
    export NCCL_SOCKET_IFNAME=$FAST_NIC
    echo "Using network interface: $FAST_NIC"
fi

# Tune based on GPU count
if [ $NUM_GPUS -ge 8 ]; then
    echo "Configuring for 8+ GPU system"
    export NCCL_MAX_NCHANNELS=16
    export NCCL_MIN_NCHANNELS=8
    export NCCL_BUFFSIZE=8388608
elif [ $NUM_GPUS -ge 4 ]; then
    echo "Configuring for 4-7 GPU system"
    export NCCL_MAX_NCHANNELS=8
    export NCCL_MIN_NCHANNELS=4
    export NCCL_BUFFSIZE=4194304
else
    echo "Configuring for 2-3 GPU system"
    export NCCL_MAX_NCHANNELS=4
    export NCCL_MIN_NCHANNELS=2
fi

# Enable optimal settings
export NCCL_P2P_LEVEL=NVL
export NCCL_NET_GDR_LEVEL=5
export NCCL_DEBUG=INFO

# Print configuration
echo -e "\n=== NCCL Configuration ==="
env | grep NCCL_ | sort

# Run NCCL tests
echo -e "\n=== Running NCCL Tests ==="
if [ -f ./nccl-tests/build/all_reduce_perf ]; then
    ./nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g $NUM_GPUS
else
    echo "NCCL tests not found. Clone from https://github.com/NVIDIA/nccl-tests"
fi
```

### Example 5: Custom Topology File

```xml
<?xml version="1.0"?>
<!-- custom_topology.xml - Manual topology specification -->
<system version="1">
  <cpu numaid="0" affinity="0000ffff" arch="x86_64" vendor="GenuineIntel" familyid="6" modelid="85">
    <pci busid="0000:00:00.0" class="0x060000" link_speed="8.0 GT/s" link_width="16">
      <!-- GPU 0 -->
      <pci busid="0000:3b:00.0" class="0x030000" link_speed="16.0 GT/s" link_width="16">
        <gpu dev="0" sm="80" rank="0" gdr="1"/>
      </pci>
      <!-- GPU 1 -->
      <pci busid="0000:af:00.0" class="0x030000" link_speed="16.0 GT/s" link_width="16">
        <gpu dev="1" sm="80" rank="1" gdr="1"/>
      </pci>
      <!-- NVLink connections -->
      <nvlink target="0" count="12" tclass="0x030000"/>
      <nvlink target="1" count="12" tclass="0x030000"/>

      <!-- InfiniBand HCA -->
      <pci busid="0000:1a:00.0" class="0x020700" link_speed="8.0 GT/s" link_width="16">
        <nic>
          <net name="mlx5_0" port="1" gdr="1" speed="100" latency="1" maxconn="131072" />
        </nic>
      </pci>
    </pci>
  </cpu>
</system>
```

**Use custom topology:**
```bash
export NCCL_TOPO_FILE=/path/to/custom_topology.xml
export NCCL_DEBUG=INFO
./my_nccl_app
```

### Example 6: Benchmarking Script

```bash
#!/bin/bash
# nccl_comprehensive_benchmark.sh

set -e

NUM_GPUS=${1:-4}
OUTPUT_DIR="nccl_benchmarks_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "=== NCCL Comprehensive Benchmark ==="
echo "GPUs: $NUM_GPUS"
echo "Output: $OUTPUT_DIR"

# Check if nccl-tests is available
if [ ! -d "nccl-tests" ]; then
    echo "Cloning nccl-tests..."
    git clone https://github.com/NVIDIA/nccl-tests.git
    cd nccl-tests
    make -j
    cd ..
fi

NCCL_TEST="./nccl-tests/build"

# Configure NCCL
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=$OUTPUT_DIR/nccl_debug.log

# Test 1: AllReduce - varying message sizes
echo -e "\n=== Test 1: AllReduce (varying sizes) ==="
$NCCL_TEST/all_reduce_perf \
    -b 8 -e 1G -f 2 -g $NUM_GPUS \
    | tee $OUTPUT_DIR/allreduce_sizes.txt

# Test 2: AllGather
echo -e "\n=== Test 2: AllGather ==="
$NCCL_TEST/all_gather_perf \
    -b 1M -e 128M -f 2 -g $NUM_GPUS \
    | tee $OUTPUT_DIR/allgather.txt

# Test 3: Broadcast
echo -e "\n=== Test 3: Broadcast ==="
$NCCL_TEST/broadcast_perf \
    -b 1M -e 128M -f 2 -g $NUM_GPUS \
    | tee $OUTPUT_DIR/broadcast.txt

# Test 4: ReduceScatter
echo -e "\n=== Test 4: ReduceScatter ==="
$NCCL_TEST/reduce_scatter_perf \
    -b 1M -e 128M -f 2 -g $NUM_GPUS \
    | tee $OUTPUT_DIR/reducescatter.txt

# Test 5: SendRecv (point-to-point)
echo -e "\n=== Test 5: SendRecv ==="
$NCCL_TEST/sendrecv_perf \
    -b 1M -e 128M -f 2 -g $NUM_GPUS \
    | tee $OUTPUT_DIR/sendrecv.txt

# Test 6: AllReduce with different algorithms
echo -e "\n=== Test 6: Algorithm comparison ==="
for algo in Ring Tree; do
    echo "Testing $algo algorithm..."
    NCCL_ALGO=$algo $NCCL_TEST/all_reduce_perf \
        -b 8M -e 128M -f 2 -g $NUM_GPUS \
        | tee $OUTPUT_DIR/allreduce_${algo}.txt
done

# Parse results
python3 << 'EOF'
import re
import sys

def parse_nccl_output(filename):
    """Parse NCCL test output"""
    with open(filename) as f:
        content = f.read()

    # Extract bandwidth numbers
    pattern = r'(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
    matches = re.findall(pattern, content)

    if matches:
        print(f"\n{filename}:")
        print(f"  Size | Out-of-place BW | In-place BW")
        for match in matches[-5:]:  # Last 5 entries
            size = int(match[0])
            oop_bw = float(match[2])
            ip_bw = float(match[3])
            print(f"  {size:>10} | {oop_bw:>14.2f} | {ip_bw:>12.2f}")

import glob
output_dir = sys.argv[1] if len(sys.argv) > 1 else "."
for file in glob.glob(f"{output_dir}/*.txt"):
    try:
        parse_nccl_output(file)
    except:
        pass
EOF

echo -e "\n=== Benchmark Complete ==="
echo "Results saved to $OUTPUT_DIR/"
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Low Bandwidth / Poor Performance

**Problem**: Achieving much lower bandwidth than expected.

**Solution**:
```bash
# Check topology detection
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH

# Verify P2P is enabled
nvidia-smi topo -p2p w
# Should show "OK" for NVLink connections

# Force optimal path
export NCCL_P2P_LEVEL=NVL     # Use NVLink
export NCCL_NET_GDR_LEVEL=5   # Max GPUDirect

# Increase buffer size for large messages
export NCCL_BUFFSIZE=8388608  # 8MB

# Check if using correct algorithm
export NCCL_ALGO=Ring         # Or Tree for large node counts
```

#### 2. NCCL Initialization Hangs

**Problem**: `ncclCommInitRank` or `ncclCommInitAll` hangs indefinitely.

**Solution**:
```bash
# Enable debug logging
export NCCL_DEBUG=INFO

# Check network connectivity
ping -c 3 <other_node_ip>

# Verify firewall allows traffic
sudo ufw allow 12345/tcp  # Or your chosen port

# For multi-node, ensure consistent NCCL_SOCKET_IFNAME
export NCCL_SOCKET_IFNAME=eth0  # Same on all nodes

# Check for stale processes
ps aux | grep nccl
killall -9 <stale_process>

# Try with timeout
timeout 60 ./my_nccl_app || echo "Initialization timeout"
```

#### 3. InfiniBand Not Working

**Problem**: NCCL falls back to TCP/IP despite InfiniBand being available.

**Solution**:
```bash
# Check IB adapters
ibstat
ibv_devinfo

# Verify RDMA stack
rdma link show

# Configure NCCL to use IB
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1
export NCCL_NET_GDR_LEVEL=5

# For RoCE, set GID index
export NCCL_IB_GID_INDEX=3

# Check GDR is enabled
cat /proc/driver/nvidia/capabilities/gpu*/config | grep "4: GPUDirectRDMACapable: 1"

# Test IB performance
ib_write_bw -d mlx5_0 -a --report_gbits
```

#### 4. NCCL Errors with Large Models

**Problem**: "unhandled system error" or "Out of memory" errors.

**Solution**:
```bash
# Increase system limits
ulimit -n 65536           # Open files
ulimit -s unlimited       # Stack size
ulimit -l unlimited       # Locked memory

# For IB, increase max locked memory
echo "* soft memlock unlimited" | sudo tee -a /etc/security/limits.conf
echo "* hard memlock unlimited" | sudo tee -a /etc/security/limits.conf

# Reduce NCCL memory usage
export NCCL_BUFFSIZE=2097152     # 2MB (from 4MB default)
export NCCL_MAX_NCHANNELS=4      # Fewer channels

# Check GPU memory
nvidia-smi --query-gpu=memory.used --format=csv
```

#### 5. Different Performance Across Runs

**Problem**: Inconsistent bandwidth measurements.

**Solution**:
```bash
# Disable CPU frequency scaling
sudo cpupower frequency-set -g performance

# Set GPU clocks to max
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 1410  # Set to max clock

# Disable interrupts on NCCL cores
# ... (similar to NIXL SKILL.md)

# Add warmup iterations
for i in {1..10}; do
    ncclAllReduce(...)
done
# Now measure

# Check for thermal throttling
nvidia-smi dmon -s pucvmet -c 100
```

#### 6. Multi-Node Communication Fails

**Problem**: Works within node but fails across nodes.

**Solution**:
```bash
# Verify network interface
export NCCL_SOCKET_IFNAME=eth0  # Must match actual interface
ip addr show

# Check routing
ip route get <remote_node_ip>

# For IB, check subnet manager
ibstat | grep SM

# Test basic connectivity
./nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 8 -c 0

# Enable detailed network logging
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=NET,INIT

# Verify MPI configuration (if using MPI)
mpirun --hostfile hosts.txt -np 2 hostname
```

#### 7. Slow AllReduce for Small Messages

**Problem**: High latency for small (<1MB) messages.

**Solution**:
```bash
# Use Low-Latency protocol
export NCCL_PROTO=LL
export NCCL_LL_THRESHOLD=0  # Always use LL

# Reduce number of threads
export NCCL_NTHREADS=256

# For very small messages, use LL128
export NCCL_PROTO=LL128

# Verify CPU affinity
numactl --hardware
```

### Getting Help

1. **Enable comprehensive debugging**:
   ```bash
   export NCCL_DEBUG=TRACE
   export NCCL_DEBUG_SUBSYS=ALL
   export NCCL_DEBUG_FILE=/tmp/nccl_debug_%h_%p.log
   ```

2. **Collect system information**:
   ```bash
   nvidia-smi topo -m > topology.txt
   ibstat > ibstat.txt
   lspci | grep -i nvidia > gpus.txt
   ```

3. **Run NCCL tests**:
   ```bash
   cd nccl-tests
   ./build/all_reduce_perf -b 8 -e 128M -f 2 -g <ngpus>
   ```

4. **Check GitHub issues**: https://github.com/NVIDIA/nccl/issues

5. **Report bugs with**:
   - NCCL version (`ldconfig -p | grep nccl`)
   - CUDA version (`nvcc --version`)
   - GPU topology (`nvidia-smi topo -m`)
   - Network configuration
   - Full debug logs

## Advanced Topics

### Custom Topology Configuration

```c
// Override automatic topology detection
export NCCL_TOPO_FILE=/path/to/custom.xml

// Or disable specific paths
export NCCL_IGNORE_DISABLED_P2P=0
export NCCL_P2P_DISABLE=1  // Disable all P2P
```

### SHARP (Scalable Hierarchical Aggregation Protocol)

```bash
# Enable SHARP for InfiniBand networks
export NCCL_COLLNET_ENABLE=1
export NCCL_COLLNET_NODE_THRESHOLD=2  # Min 2 nodes

# Requires SHARP support in network switches
# Check with: ibv_devinfo | grep -i sharp
```

### Plugin Development

```c
// Custom network plugin for NCCL
// See https://github.com/NVIDIA/nccl/tree/master/ext-net

typedef struct {
    ncclResult_t (*init)(ncclDebugLogger_t logFunction);
    ncclResult_t (*devices)(int* ndev);
    // ... other function pointers
} ncclNet_v5_t;

// Compile plugin
gcc -shared -fPIC -o libnccl-net.so my_plugin.c

// Use plugin
export NCCL_NET_PLUGIN=/path/to/libnccl-net.so
```

### Profiling NCCL

```bash
# Use NVIDIA Nsight Systems
nsys profile -o nccl_profile ./my_app

# Use NVIDIA Nsight Compute for kernel analysis
ncu --set full --target-processes all ./my_app

# Analyze in GUI
nsys-ui nccl_profile.qdrep
```

## Security Considerations

### Network Isolation

```bash
# Restrict to specific network interface
export NCCL_SOCKET_IFNAME=eth0  # Only use eth0

# For IB, specify exact adapters
export NCCL_IB_HCA=mlx5_0:1  # Only port 1 of mlx5_0
```

### Access Control

```bash
# Run with limited privileges
# Avoid running as root

# Use cgroups to isolate resources
cgcreate -g memory,cpu:nccl_group
cgexec -g memory,cpu:nccl_group ./my_app
```

## Resources

- **Repository**: https://github.com/NVIDIA/nccl
- **Documentation**: https://docs.nvidia.com/deeplearning/nccl/
- **Downloads**: https://developer.nvidia.com/nccl
- **NCCL Tests**: https://github.com/NVIDIA/nccl-tests
- **Issue Tracker**: https://github.com/NVIDIA/nccl/issues
- **Developer Forums**: https://forums.developer.nvidia.com/c/accelerated-computing/deep-learning/

## Notes

### Platform Support
- **Linux**: Full support (x86_64, aarch64, ppc64le)
- **Windows**: Limited support (WSL2 recommended)
- **Cloud**: AWS, Azure, GCP with specialized instances

### GPU Requirements
- **Minimum**: Compute Capability 3.5 (Kepler)
- **Recommended**: Compute Capability 8.0+ (Ampere, Hopper)
- **Optimal**: NVLink or NVSwitch connectivity

### Network Requirements
- **Intra-node**: PCIe 3.0+ or NVLink
- **Inter-node**: InfiniBand HDR/NDR or 100GbE+
- **Latency sensitive**: InfiniBand with GPUDirect RDMA

### Performance Characteristics
- **NVLink bandwidth**: Up to 450 GB/s (8x H100)
- **InfiniBand bandwidth**: Up to 160 GB/s (8x HDR)
- **Small message latency**: 20-50 μs (NVLink)
- **Scaling efficiency**: 85-98% up to 128 GPUs

### Production Readiness
- Battle-tested in all major deep learning frameworks
- Used by PyTorch, TensorFlow, JAX, MXNet
- Supports NVIDIA DGX systems
- Regular updates and bug fixes
- Backward compatibility maintained

### Known Limitations
- Requires homogeneous GPU types for optimal performance
- Cross-vendor communication not supported
- Windows support is experimental
- Very large scale (1000+ GPUs) may require tuning

### Version Notes
- 2.x series is stable and widely deployed
- Semantic versioning not strictly followed
- ABI compatibility within minor versions
- Check release notes for breaking changes

## Related Technologies

- **MPI** (Message Passing Interface): Standard for parallel computing
- **UCX** (Unified Communication X): Communication framework for HPC
- **NVSHMEM**: SHMEM implementation for NVIDIA GPUs
- **NVIDIA NVLink**: High-bandwidth GPU interconnect
- **GPUDirect RDMA**: Direct GPU-to-network transfers
- **InfiniBand**: High-performance network fabric
- **Mellanox SHARP**: In-network computing for collectives
- **GDRCopy**: Low-latency GPU memory access
- **CUDA IPC**: Inter-process communication for CUDA
