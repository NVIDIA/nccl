# Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms

**Paper:** [arXiv:2507.04786v2](https://arxiv.org/abs/2507.04786v2)  
**Authors:** Zhiyi Hu, Siyuan Shen, Tommaso Bonato, Sylvain Jeaugey, Cedell Alexander, Eric Spada, James Dinan, Jeff Hammond, Torsten Hoefler  
**Institutions:** ETH Zürich, NVIDIA Corporation, Broadcom Inc.  
**Date:** July 2025

---

## Overview

This paper presents a comprehensive analysis of NVIDIA's Collective Communication Library (NCCL), focusing on its internal architecture, communication protocols, data transfer mechanisms, and collective algorithms. The insights have been used to develop ATLAHS, an application-trace-driven network simulation toolchain for AI training workloads.

---

## Key Findings

### 1. Communication Protocols

NCCL employs three communication protocols with different trade-offs:

| Protocol | Design Goal | Synchronization | Payload | Bandwidth Utilization |
|----------|-------------|-----------------|---------|----------------------|
| **Simple** | High bandwidth | Memory fences (high overhead) | Data chunks | Near peak |
| **LL** | Low latency | Flag-based | 4B data + 4B flag | ~25-50% of peak |
| **LL128** | Low latency + high bandwidth | Flag-based | 120B data + 8B flag | ~95% of peak |

**Key insights:**
- **Simple** excels for large messages but suffers high latency for small payloads
- **LL** reduces latency using lightweight flag-based synchronization but limits bandwidth
- **LL128** combines low latency with high throughput but requires atomic 128-byte writes (NVLink)

### 2. Data Transfer Methods

#### Intra-node Communication
- **P2P Transport**: Prioritizes NVLink for direct GPU-to-GPU transfers
- **P2P_DIRECT mode**: Bypasses IPC handles within same process, eliminates intermediate FIFO buffers
- **Shared Memory (SHM)**: Used when P2P is unavailable or suboptimal (e.g., inter-socket PCIe)

#### Inter-node Communication
- **Socket Transport**: Uses host memory staging with cudaMemcpy, standard TCP/IP
- **IB Verbs Transport**: Leverages RDMA for minimal CPU intervention
- **GPUDirect RDMA**: NIC directly accesses GPU memory (requires same PCIe switch)

### 3. Communication Channels

- Each collective is subdivided into **communication channels**
- Each channel runs as a separate CUDA block on its own SM
- Enables parallel processing of disjoint data chunks
- Trade-off: Too many channels can cause under-utilization of NIC FIFO buffers

### 4. Collective Algorithms

#### Supported Algorithms
| Operation | Ring | Tree | CollNet | NVLS |
|-----------|------|------|---------|------|
| AllReduce | ✓ | ✓ | ✓ | ✓ |
| Broadcast | ✓ | ✗ | ✗ | ✗ |
| Reduce | ✓ | ✗ | ✗ | ✗ |
| ReduceScatter | ✓ | ✗ | ✗ | ✓ |
| AllGather | ✓ | ✗ | ✗ | ✓ |

#### Algorithm Patterns

**Non-pipelined (Ring):**
- Ring AllReduce: 2k-1 steps (ReduceScatter + AllGather phases)
- Ring AllGather: k-1 steps
- Ring ReduceScatter: k steps

**Pipelined (Tree):**
- Tree AllReduce: Reduce phase (upward) + Broadcast phase (downward)
- Uses double binary tree structure for bandwidth utilization

### 5. CUDA Hierarchy Mapping

- **Grid**: `(nChannels, 1, 1)` - one block per channel
- **Block**: Variable threads (NCCL_MIN_NTHREADS to NCCL_MAX_NTHREADS)
- **Warps**: First two warps handle initialization; remaining warps do communication
- **Slots**: 8 slots per channel (NCCL_STEPS) for fine-grained pipelining

---

## Benchmarking Results

Testing on Alps supercomputer (16 nodes, NVIDIA GH200):

### Inter-node Communication
- **Small messages (<64 KiB)**: LL and LL128 perform best
- **Large messages (GB range)**: Simple protocol significantly outperforms LL/LL128
- Flag-based synchronization overhead becomes prohibitive at scale

### Intra-node Communication
- **LL128** provides consistent performance across all message sizes (NVLink advantage)
- **Simple** best for large messages
- **LL** best for small messages

### Algorithm Selection
- **Ring**: Excels for large messages
- **Tree**: Better for smaller messages

---

## Key Takeaways

1. **Protocol selection matters**: NCCL's autotuning generally provides robust performance, but understanding protocol trade-offs enables targeted optimization.

2. **Topology awareness**: Performance varies significantly between intra-node (NVLink) and inter-node (RoCE/IB) communication paths.

3. **Channel count trade-off**: More channels increase GPU-side parallelism but can degrade network efficiency for smaller messages.

4. **GPUDirect RDMA**: Critical optimization when NIC and GPU share the same PCIe switch - eliminates host memory staging entirely.

5. **Pipelining**: Tree algorithms enable overlapping reduce and broadcast phases; ring algorithms require sequential processing within loop iterations.

---

## Relevance to This Project

This analysis directly informs the Windows port of NCCL:

- **Transport layer abstraction**: Understanding P2P, SHM, and NET transports helps design Windows equivalents
- **IB Verbs limitations**: Confirms InfiniBand transport is Linux-only; Windows would require Network Direct API
- **Socket fallback**: Socket transport provides cross-platform compatibility for inter-node communication
- **Protocol selection**: Same tuning logic applies regardless of platform

---

## References

- NCCL GitHub: https://github.com/NVIDIA/nccl
- ATLAHS Toolchain: Application-trace-driven network simulation for AI workloads
- Analysis based on NCCL version 2.19.1
