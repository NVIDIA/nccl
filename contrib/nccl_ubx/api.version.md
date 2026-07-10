# UB-X public API catalog

**API contract version: `0.01`** — `ubx.version` / `ubx.get_version()`

---

## Design: the registry IS the changelog

There is **no hand-maintained "current state"**. The source of truth is
`API_LOG` in `ubx/_api_registry.py` — an append-only chronological list
of `Event`s. The current public surface (parameters, hardware support,
who-was-removed-when) is computed by **replaying the log** at module
load time.

Each release that touches the public API:
1. Appends one or more `Event` entries to `API_LOG`.
2. Bumps `API_VERSION` (`X.YZ → X.(YZ+1)` additive,
   `(X+1).00` for removals / renames / required-param changes).
3. Adds a per-version row at the bottom of this file (so humans can
   read the history without grepping Python).

First release (`0.01`) bootstraps the log with one `api_added` event
per existing public API. Subsequent releases append whatever changed:
could be 0 events (e.g. a docs / perf release with no API drift) or
many (e.g. a refactor).

This means **history is preserved automatically**: `query_api()` can
look back through `API_LOG` and tell a user "your param was renamed in
v0.07, pin to v0.06 or update the call site".

---

## Runtime query API

```python
import ubx

ubx.version                 # "0.01" (the API contract version, not the package version)
ubx.get_version()           # "0.01"

# Did my API + params survive the most recent ubx release?
result = ubx.query_api(
    "SymmAllocator.alltoallv",
    params=["tensor_in", "output_split_sizes", "input_split_sizes"],
)
# result["ok"]               → True
# result["status"]           → "ok" | "not_found" | "removed" | "renamed" | "params_changed"
# result["current_params"]   → ["tensor_in", "output_split_sizes", "input_split_sizes", "smlimit"]
# result["transports"]       → ["nvlink"]
# result["sm_min"]           → "9.0a"
# result["advice"]           → human-readable diagnostic

# Hardware capability — global aggregate (union across all current APIs)
ubx.query_supported_hw()
# {"transports": ["nvlink"], "sm_archs": [...], ...}

# Hardware capability — per-API (different APIs may support different transports)
ubx.query_supported_hw("SymmAllocator.alltoall")
# {"api": "SymmAllocator.alltoall", "transports": ["nvlink"], "sm_min": "9.0a", ...}
```

The contract is lenient on additive change: passing a subset of current
params returns `ok`. Passing a name NOT in the current signature returns
`params_changed` with advice (and a hint about when it was removed /
renamed, pulled from the changelog).

---

## Event kinds

Recognized values of `Event.kind`:

| Kind | Required fields | Meaning |
|---|---|---|
| `api_added` | `api`, `description`, `params`, [`optional_params`], [`transports`], [`sm_min`] | New API surface |
| `api_removed` | `api` | API gone entirely |
| `api_renamed` | `api`, `new_name` | Old name → new name (spec carries over) |
| `param_added` | `api`, `param`, [`optional`] | New param on existing API |
| `param_removed` | `api`, `param` | Param dropped |
| `param_renamed` | `api`, `param`, `new_name` | Param renamed |
| `param_made_optional` | `api`, `param` | Required → optional (additive, won't break callers) |
| `param_made_required` | `api`, `param` | Optional → required (BREAKING) |
| `hw_added` | `api`, `transport` | API gains a transport (e.g. `roce`) |
| `hw_removed` | `api`, `transport` | API loses a transport |
| `sm_min_changed` | `api`, `sm_min` | Minimum SM arch changed |
| `description_changed` | `api`, `description` | Doc-string update only |
| `behavior_changed` | `api`, `description` | Semantics changed (no derived-state diff) |

Unknown `kind` values are ignored by the replayer (forward-compat:
an older ubx can load a log written by a newer version without crashing).

---

## Current API surface (as of v0.01)

All APIs below were added in v0.01. Hardware support: `nvlink` only, SM `9.0a`+.

### `SymmAllocator` — class

| API | Params | Optional |
|---|---|---|
| `SymmAllocator` (constructor) | `size_bytes`, `device`, `dist_group` | — |
| `.close()` | — | — |
| `.create_tensor()` | `shape`, `dtype`, `blocked` | `blocked` |
| `.barrier()` | `smlimit` | `smlimit` |

### Allreduce

| API | Params | Optional |
|---|---|---|
| `.allreduce()` | `tensor_in`, `tensor_out`, `smlimit`, `nthreads`, `gamma`, `residual_in`, `residual_out`, `eps` | all except `tensor_in` |
| `.allreduce_mc()` | same | same |
| `.allreduce_uc()` | `tensor_in`, `tensor_out`, `smlimit`, `nthreads` | all except `tensor_in` |
| `.allreduce_lamport()` | `tensor_in`, `tensor_out`, `smlimit`, `nthreads` | all except `tensor_in` |

### Allgather

| API | Params | Optional |
|---|---|---|
| `.allgather()` | `tensor_in`, `smlimit`, `nthreads` | `smlimit`, `nthreads` |
| `.allgather_uc()` | `tensor_in`, `smlimit`, `nthreads` | `smlimit`, `nthreads` |

### Alltoall (fixed-size)

| API | Params | Optional |
|---|---|---|
| `.alltoall()` | `tensor_in`, `smlimit`, `nthreads` | `smlimit`, `nthreads` |
| `.alltoall_lamport()` | `tensor_in`, `smlimit`, `nthreads` | `smlimit`, `nthreads` |
| `.alltoall_auto()` | `tensor_in`, `smlimit`, `nthreads` | `smlimit`, `nthreads` |

### Alltoallv (variable-length)

| API | Params | Optional |
|---|---|---|
| `.alltoallv()` | `tensor_in`, `output_split_sizes`, `input_split_sizes`, `smlimit` | `smlimit` |
| `.alltoallv_prepare()` | `output_split_sizes`, `input_split_sizes`, `dtype` | — |
| `.alltoallv_run()` | `tensor_in`, `state`, `smlimit` | `smlimit` |

### MoE fused dispatch

| API | Params | Optional |
|---|---|---|
| `.a2av_token_bf16_bf16()` | `tokens_bf16`, `token_offsets`, `experts_per_rank`, `output`, `smlimit`, `sync`, `expert_start`, `expert_count` | all except first 3 |
| `.a2av_token_bf16_bf16_topk()` | `tokens_bf16`, `topk_expert`, `topk_slot`, `experts_per_rank`, `output`, `smlimit`, `sync` | last 3 |
| `.a2av_token_bf16_mxfp8()` | `tokens_bf16`, `token_offsets`, `experts_per_rank`, `output`, `smlimit`, `sync`, `expert_start`, `expert_count` | last 5 |
| `.a2av_token_bf16_mxfp8_persistent()` | same minus expert_start/count | `smlimit`, `sync` |
| `.a2av_wait()` | — | — |

### MoE combine

| API | Params | Optional | Notes |
|---|---|---|---|
| `.combine_push3_bf16_bf16()` | `expert_outputs`, `inverse_map`, `topk_idx`, `experts_per_rank`, `max_tokens_per_rank`, `gate_weights`, `smlimit` | `gate_weights`, `smlimit` | **RECOMMENDED** |
| `.combine_bf16_bf16_push()` | same | same | Single-kernel PUSH + MC barrier |
| `.combine_bf16_bf16_lamport_push()` | same | same | PUSH + Lamport polling |
| `.combine_v2_bf16_bf16()` | `expert_outputs`, `token_offsets`, `experts_per_rank`, `smlimit` | `smlimit` | 2-kernel (E22) |
| `.combine_bf16_bf16()` | `expert_outputs`, `token_offsets`, `experts_per_rank`, `combine_temp`, `smlimit` | `combine_temp`, `smlimit` | PULL — unstable at EP≥32 |
| `.combine_mxfp8_bf16()` | `expert_outputs`, `token_offsets`, `experts_per_rank`, `combine_temp`, `smlimit` | `combine_temp`, `smlimit` | mxfp8 in, bf16 accum |
| `.combine_wait()` | — | — | Wait kernel for split variants |

### `SymmTensor` — class

| API | Params | Optional |
|---|---|---|
| `SymmTensor` | `data`, `allocator` | — |

### Module-level helpers

| API | Params | Optional |
|---|---|---|
| `compute_token_offsets()` | `routing`, `experts_per_rank`, `myrank`, `nranks` | — |
| `compute_combine_push_map()` | `token_offsets`, `experts_per_rank`, `myrank`, `nranks`, `capacity` | `capacity` |
| `compute_dispatch_topk_map()` | `token_offsets`, `topk_max` | — |
| `ops.allreduce()` | `tensor_in`, `dist_group`, `gamma`, `residual_in`, `residual_out`, `eps` | all except `tensor_in` |
| `ops.request_allocator()` | `max_tensor_size`, `dtype`, `dist_group` | — |
| `ops.get_sym_tensor()` | `shape`, `dtype`, `dist_group` | — |
| `ops.free_residual()` | `tensor_in` | — |
| `ops.restore()` | `tensor`, `dist_group` | — |
| `ops.mem_stats()` | — | — |
| `fused.allreduce_fused()` | `tensor_in`, `gamma`, `residual_in`, `residual_out`, `eps`, `dist_group` | `residual_out`, `eps`, `dist_group` |

### Meta APIs (this module)

| API | Params | Optional |
|---|---|---|
| `get_version()` | — | — |
| `query_api()` | `name`, `params` | `params` |
| `query_supported_hw()` | `api_name` | `api_name` |

---

## Change log (per-version)

### v0.01 — 2026-06-02

Initial registry snapshot. 42 `api_added` events bootstrap the full
surface listed above. All APIs are `nvlink`-only, SM `9.0a`+ minimum.

Future versions append new rows here. Example future entries:

> ### v0.02 (hypothetical)
> - `hw_added`: `SymmAllocator.alltoall` gains `roce` transport.
> - `param_added`: `SymmAllocator.alltoallv` adds optional `timeout_ms`.
>
> ### v0.03 (hypothetical, breaking)
> - `api_removed`: `SymmAllocator.combine_bf16_bf16` removed.
> - `param_made_required`: `SymmAllocator.create_tensor.dtype` is now required.

---

## How to keep this file accurate

Whenever you change a public API (add/remove/rename/move a parameter
or transport), do all of:

1. Append `Event(...)` entries to `API_LOG` in `ubx/_api_registry.py`.
   Use the appropriate `kind`: `api_added`, `api_removed`, `api_renamed`,
   `param_added`, `param_removed`, `param_renamed`, `param_made_optional`,
   `param_made_required`, `hw_added`, `hw_removed`, `sm_min_changed`,
   `description_changed`, or `behavior_changed`.
2. Bump `API_VERSION` per the scheme at the top of this file
   (`X.YZ → X.(YZ+1)` for additive; `(X+1).00` for breaking).
3. Add a per-version row at the bottom of THIS file describing the change.
4. Never edit past `Event` entries — they're history. The replayer
   computes current state by walking them in order; rewriting history
   silently changes derived state for older versions.

The registry is hand-maintained (not auto-introspected) so it captures
intent and supports the rename/remove tracking that lets `query_api()`
advise users when an API has moved.
