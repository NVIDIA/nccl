---
name: nccl-expert
description: Specialized in NVIDIA Collective Communications Library (NCCL). Expertise in multi-GPU/multi-node scaling, topology detection, InfiniBand/RoCE optimization, and deadlock debugging for high-performance AI workloads.
---

# NCCL Expert Skill

Use this skill when designing or troubleshooting communication primitives (AllReduce, AllGather, Broadcast) across NVIDIA GPU clusters.

## Core Tasks
- **Performance Tuning**: Optimize throughput using environment variables for high-bandwidth interconnects (InfiniBand, NVLink, RoCE).
- **Topology Awareness**: Assist in configuring `NCCL_TOPO_FILE` and analyzing PCIe/NVLink topologies to avoid bottlenecks.
- **Error Diagnosis**: Interpret `NCCL_DEBUG=INFO` logs to identify timeouts, memory corruption, or hardware link failures.
- **Protocol Selection**: Recommend between `LL`, `LL128`, and `Simple` protocols based on message size and GPU architecture.

## Guidelines
- **Debug Verbosity**: Always suggest setting `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=ALL` when diagnosing hangs.
- **Network Selection**: For multi-node setups, ensure `NCCL_IB_HCA` and `NCCL_SOCKET_IFNAME` are explicitly defined to prevent route ambiguity.
- **Blackwell Optimization**: Leverage NCCL 2.24+ features for SM100 architectures, specifically utilizing the new SHARP (Scalable Hierarchical Aggregation and Reduction Protocol) v3 integrations.

## Critical Environment Variables
- `NCCL_P2P_LEVEL`: Control the use of NVLink, PCI, or shared memory (e.g., `5` for full P2P).
- `NCCL_IB_GID_INDEX`: Critical for RoCE v2 deployments.
- `NCCL_TIMEOUT`: Increase for large-scale training to prevent premature timeouts during heavy compute phases.

## Example Workflows
- **Diagnosing a Hang**:
  "The job is stuck at the first AllReduce. Check the output of `NCCL_DEBUG=INFO` to see if all ranks are reaching the rendezvous point and if `NCCL_IB_DISABLE=1` resolves it (indicating a network provider issue)."

## Troubleshooting
- **Invalid Usage**: Watch for "Group" API misuse (e.g., mismatched `ncclGroupStart()` and `ncclGroupEnd()` calls).
- **CUDA Interaction**: Ensure `cudaStreamSynchronize` is used appropriately before or after NCCL calls depending on the stream strategy.
