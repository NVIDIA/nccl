# Hierarchical CE AllGather Ring implementation

This note documents the current retained hierarchical CE allgather implementations in `src/ce_coll.cc`.

- `src/ce_coll.cc` now contains only the retained hierarchical allgather implementations described in this note:
  - `ncclHierCeAllGather()` `(NV)`
  - `ncclHierCeAllGatherRing()` `(BD)`
- `ncclHierCeAllGatherDispatch()` is the active selector that chooses between those two implementations based on `NCCL_ENABLE_HCE_AG_RING`

## Current Selection

`NCCL_ENABLE_HCE_AG_RING` now acts as a direct-vs-ring selector for hierarchical allgather:

| `NCCL_ENABLE_HCE_AG_RING` | Selected implementation | Ownership |
| --- | --- | --- |
| `0` or unset | `ncclHierCeAllGather()` | `(NV)` |
| nonzero | `ncclHierCeAllGatherRing()` | `(BD)` |

The tuning log is correspondingly simplified:

- direct path: `RMA proxy + CE`
- ring path: `RMA proxy ring + CE`

## Implementation Comparison

| Property | `ncclHierCeAllGather()` `(NV)` | `ncclHierCeAllGatherRing()` `(BD)` |
| --- | --- | --- |
| High-level structure | Direct hierarchical allgather | Bidirectional hierarchical ring |
| Inter-node schedule | Direct sends to each remote node | Clockwise and counterclockwise half-ring forwarding |
| Wait helper | Per-peer generic wait path | `ncclProxyWaitPeersRing()` fast path |
| Context regime | Generic `1 -> full` | Generic `1 -> full` |
| Large-message chunking | Legacy full-message chunk width | Half-aware large-message chunk width for `perRankBytes >= 128 MiB` and `numCtx >= 4`; otherwise legacy |
| Main strength | Baseline direct hierarchical implementation | Better large-message overlap and improved extreme-size behavior |

## Current Ring Design

The current `ncclHierCeAllGatherRing()` `(BD)` is the current `HCE-Ring AllGather` implementation and keeps the large-message improvements that previously lived in historical `RingV5`:

- same fast ring wait path as the historical ring-family implementations, now named `ncclProxyWaitPeersRing()`
- same generic `1 -> full` context regime as `RingV2`
- a half-aware chunk-width rule for clearly large messages

The half-aware chunk-width rule activates only when:

- `perRankBytes >= 128 MiB`
- `numCtx >= 4`

Below that regime, the current ring implementation uses the same legacy chunk width as the baseline ring-family path.

## Why The Ring Differs From The Direct Path

The current ring implementation changes the inter-node schedule and, for large messages, changes the chunk geometry so the clockwise and counterclockwise half transfers are not too coarse.

Illustrative `numCtx = 4` examples:

| Per-rank input | `cwBytes` / `ccwBytes` | Legacy width | Current ring width | Legacy chunks per half | Current ring chunks per half |
| --- | --- | --- | --- | --- | --- |
| `128 MiB` | `64 MiB` / `64 MiB` | `32 MiB` | `16 MiB` | `2` | `4` |
| `256 MiB` | `128 MiB` / `128 MiB` | `64 MiB` | `32 MiB` | `2` | `4` |
| `512 MiB` | `256 MiB` / `256 MiB` | `64 MiB` | `64 MiB` | `4` | `4` |

This keeps the small and medium regime unchanged while improving chunk geometry first in the `128 MiB` to `256 MiB` per-rank range.

## Benchmark Results

The table below records the benchmark results shared during this discussion. This benchmark section now compares only:

- `[SM] Cpu-Proxy AllGather (NVIDIA)`: SM-based version using CPU proxy for remote communication.
- `[SM-Free] HCE-Direct AllGather (NVIDIA)`: SM-Free version developed by NVIDIA at commit `8a074116f105b7febdc6761115a46e5673dbf045`.
- `[SM-Free] HCE-Ring AllGather (BD)`: SM-Free version developed by BD.

### Setup

- `GIN_NCONNECTIONS=4`
- multicast is on
- communication pattern is global
- number of ranks: `32`
- number of nodes: `8`
- throughput metric: bus bandwidth
- throughput unit: `GB/s`
- operation mode: inplace

### Inplace Bus Bandwidth Table (`GB/s`)

| Output size | [SM] Cpu-Proxy AllGather (NVIDIA) | [SM-Free] HCE-Direct AllGather (NVIDIA) | [SM-Free] HCE-Ring AllGather (BD) |
| --- | --- | --- | --- |
| `1K` | `0.01` | `0` | `0.01` |
| `2K` | `0.02` | `0.01` | `0.01` |
| `4K` | `0.04` | `0.04` | `0.02` |
| `8K` | `0.09` | `0.08` | `0.05` |
| `16K` | `0.17` | `0.16` | `0.09` |
| `32K` | `0.34` | `0.07` | `0.18` |
| `64K` | `0.65` | `0.2` | `0.36` |
| `128K` | `1.25` | `1.25` | `0.69` |
| `256K` | `2.45` | `2.44` | `1.37` |
| `512K` | `4.58` | `4.74` | `2.65` |
| `1M` | `4.79` | `9.29` | `5.25` |
| `2M` | `5.47` | `14.52` | `10.18` |
| `4M` | `9.92` | `16.74` | `20.12` |
| `8M` | `19.94` | `62.12` | `36.17` |
| `16M` | `66.78` | `94.27` | `59.37` |
| `32M` | `123.84` | `124.03` | `97.59` |
| `64M` | `146.12` | `146` | `131.34` |
| `128M` | `252.75` | `128.88` | `225.35` |
| `256M` | `271.34` | `171.11` | `282.45` |
| `512M` | `319.64` | `229.09` | `322.33` |
| `1G` | `370.92` | `260.81` | `352.78` |
| `2G` | `379.37` | `261.86` | `369.3` |
| `4G` | `382.61` | `275.41` | `377.42` |
| `8G` | `382.87` | `277.11` | `380.88` |
| `16G` | `383.45` | `276.94` | `381.74` |