"""UB-X public-API registry and query interface.

**Design: the registry IS the change log.** `API_LOG` is an append-only,
chronological list of events. The current state of every API
(parameters, hardware support, etc.) is computed by replaying the log
from v0.01 to the latest version. This means:

  - There is no "current state" hand-maintained anywhere — it is derived.
  - First version (v0.01) is bootstrapped with many `api_added` events,
    one per existing public API.
  - Subsequent versions append events for whatever changed (could be
    zero events for a no-API-change release, or many for a refactor).
  - History is preserved automatically: `query_api()` can advise users
    on when a name disappeared / was renamed / had its signature change.

**Per-API hardware support.** Each API tracks its own `transports` and
`sm_min`. e.g. `alltoall` may support `nvlink + roce` while a fused-MoE
kernel like `a2av_token_bf16_bf16` is `nvlink` only.
`query_supported_hw(api_name)` returns the per-API capabilities;
`query_supported_hw()` returns the union (what the build supports
across any API).

**Maintainers: append events to `API_LOG`. Never edit past entries.**
Bump `API_VERSION` on each release that has events. The replayer
computes current state by walking events in order; rewriting history
silently changes derived state for older versions, which would defeat
the point of having a contract version at all.

See ``api.version.md`` § "How to keep this file accurate" for the full
checklist (event kinds, version-bump rules, doc updates).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List


# ----------------------------------------------------------------------
# Version
# ----------------------------------------------------------------------

# The API contract version. Bumped on every release that has changes
# in API_LOG below.
#
# Scheme (lightweight semver):
#   X.YZ → X.(YZ+1) for additive / backwards-compatible changes
#   (X+1).00 for removals, renames, or required-param additions
API_VERSION = "0.01"


# ----------------------------------------------------------------------
# Event types
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    """One atomic change to the API contract.

    Only fields relevant to ``kind`` need to be populated; the others
    default to None / empty. The replayer (`_compute_current_state`)
    pulls out the fields it expects per `kind`.

    Recognized kinds:
      - ``api_added``      — new API surface. Requires: api, description,
                             params, [optional_params], [sm_min],
                             [transports]
      - ``api_removed``    — API removed entirely. Requires: api.
      - ``api_renamed``    — Requires: api (old name), new_name.
      - ``param_added``    — Requires: api, param, [optional].
      - ``param_removed``  — Requires: api, param.
      - ``param_renamed``  — Requires: api, param (old), new_name.
      - ``param_made_optional`` — Requires: api, param.
      - ``param_made_required`` — Requires: api, param.
      - ``hw_added``       — Add transport support to api.
                             Requires: api, transport.
      - ``hw_removed``     — Requires: api, transport.
      - ``sm_min_changed`` — Requires: api, sm_min.
      - ``description_changed`` — Requires: api, description.
      - ``behavior_changed`` — semantics-only change (no derived-state
                               update). Requires: api, description.
    """

    version: str
    kind: str
    api: str
    description: str = ""
    new_name: Optional[str] = None
    param: Optional[str] = None
    params: tuple = ()
    optional_params: tuple = ()
    optional: bool = False
    transport: Optional[str] = None
    transports: tuple = ("nvlink",)
    sm_min: str = "9.0a"


# ----------------------------------------------------------------------
# API_LOG — append-only change history
# ----------------------------------------------------------------------
#
# Append new events at the bottom. Never edit past entries. Bump
# API_VERSION above on each release that adds events here.

API_LOG: List[Event] = [
    # ============================================================
    # v0.01 — initial registry snapshot
    # ============================================================

    # --- SymmAllocator (class + core methods) ---

    Event(version="0.01", kind="api_added", api="SymmAllocator",
          description="Symmetric memory pool allocator backed by NCCL's device "
                      "API. Manages a contiguous pool + REG0 (commbuff + flag "
                      "slots) + graph/non-graph sub-pools. Exposes collectives "
                      "(allreduce, alltoall, alltoallv, allgather, combine_*, "
                      "a2av_token_*) that operate on tensors backed by the pool.",
          params=("size_bytes", "device", "dist_group")),
    Event(version="0.01", kind="api_added", api="SymmAllocator.close",
          description="Release pool, window, devcomm, and bootstrapped "
                      "ncclComm. Idempotent.",
          params=()),
    Event(version="0.01", kind="api_added", api="SymmAllocator.create_tensor",
          description="Allocate a SymmTensor view of the pool.",
          params=("shape", "dtype", "blocked"),
          optional_params=("blocked",)),
    Event(version="0.01", kind="api_added", api="SymmAllocator.barrier",
          description="Cross-rank barrier on the EP group via a dummy UBX kernel.",
          params=("smlimit",),
          optional_params=("smlimit",)),

    # --- Allreduce ---

    Event(version="0.01", kind="api_added", api="SymmAllocator.allreduce",
          description="Automatic-kernel allreduce. <0.25 MB → Lamport, else MC.",
          params=("tensor_in", "tensor_out", "smlimit", "nthreads",
                  "gamma", "residual_in", "residual_out", "eps"),
          optional_params=("tensor_out", "smlimit", "nthreads",
                           "gamma", "residual_in", "residual_out", "eps")),
    Event(version="0.01", kind="api_added", api="SymmAllocator.allreduce_mc",
          description="Multicast allreduce. Best for large tensors.",
          params=("tensor_in", "tensor_out", "smlimit", "nthreads",
                  "gamma", "residual_in", "residual_out", "eps"),
          optional_params=("tensor_out", "smlimit", "nthreads",
                           "gamma", "residual_in", "residual_out", "eps")),
    Event(version="0.01", kind="api_added", api="SymmAllocator.allreduce_uc",
          description="Unicast software-reduction allreduce. Fallback when "
                      "multicast unavailable.",
          params=("tensor_in", "tensor_out", "smlimit", "nthreads"),
          optional_params=("tensor_out", "smlimit", "nthreads")),
    Event(version="0.01", kind="api_added", api="SymmAllocator.allreduce_lamport",
          description="MC reduce-scatter + Lamport poison polling allgather. "
                      "Best for small tensors.",
          params=("tensor_in", "tensor_out", "smlimit", "nthreads"),
          optional_params=("tensor_out", "smlimit", "nthreads")),

    # --- Allgather ---

    Event(version="0.01", kind="api_added", api="SymmAllocator.allgather",
          description="Multicast allgather.",
          params=("tensor_in", "smlimit", "nthreads"),
          optional_params=("smlimit", "nthreads")),
    Event(version="0.01", kind="api_added", api="SymmAllocator.allgather_uc",
          description="Unicast allgather (works when multicast is disabled).",
          params=("tensor_in", "smlimit", "nthreads"),
          optional_params=("smlimit", "nthreads")),

    # --- Alltoall (fixed-size) ---

    Event(version="0.01", kind="api_added", api="SymmAllocator.alltoall",
          description="Fixed-size all-to-all (UC + UC exit barrier).",
          params=("tensor_in", "smlimit", "nthreads"),
          optional_params=("smlimit", "nthreads")),
    Event(version="0.01", kind="api_added", api="SymmAllocator.alltoall_lamport",
          description="All-to-all with Lamport poison polling. 1.5-2x faster "
                      "than `alltoall` at small sizes.",
          params=("tensor_in", "smlimit", "nthreads"),
          optional_params=("smlimit", "nthreads")),
    Event(version="0.01", kind="api_added", api="SymmAllocator.alltoall_auto",
          description="Auto-select alltoall variant. <0.25 MB → Lamport, else "
                      "regular.",
          params=("tensor_in", "smlimit", "nthreads"),
          optional_params=("smlimit", "nthreads")),

    # --- Alltoallv ---

    Event(version="0.01", kind="api_added", api="SymmAllocator.alltoallv",
          description="Variable-length all-to-all (convenience wrapper). For "
                      "repeated calls with same splits, use prepare+run.",
          params=("tensor_in", "output_split_sizes", "input_split_sizes",
                  "smlimit"),
          optional_params=("smlimit",)),
    Event(version="0.01", kind="api_added", api="SymmAllocator.alltoallv_prepare",
          description="Precompute per-rank byte offsets for a fixed split "
                      "pattern.",
          params=("output_split_sizes", "input_split_sizes", "dtype")),
    Event(version="0.01", kind="api_added", api="SymmAllocator.alltoallv_run",
          description="Execute alltoallv using state from `alltoallv_prepare`.",
          params=("tensor_in", "state", "smlimit"),
          optional_params=("smlimit",)),

    # --- MoE fused dispatch ---

    Event(version="0.01", kind="api_added",
          api="SymmAllocator.a2av_token_bf16_bf16",
          description="Fused MoE dispatch: per-(token, expert) NVLink writes of "
                      "bf16 token data to peer pools.",
          params=("tokens_bf16", "token_offsets", "experts_per_rank", "output",
                  "smlimit", "sync", "expert_start", "expert_count"),
          optional_params=("output", "smlimit", "sync",
                           "expert_start", "expert_count")),
    Event(version="0.01", kind="api_added",
          api="SymmAllocator.a2av_token_bf16_bf16_topk",
          description="Top-K variant of `a2av_token_bf16_bf16`. Inner loop "
                      "iterates K routed entries per token instead of "
                      "total_experts.",
          params=("tokens_bf16", "topk_expert", "topk_slot", "experts_per_rank",
                  "output", "smlimit", "sync"),
          optional_params=("output", "smlimit", "sync")),
    Event(version="0.01", kind="api_added",
          api="SymmAllocator.a2av_token_bf16_mxfp8",
          description="Fused MoE dispatch with bf16 → mxfp8 quantization in-kernel.",
          params=("tokens_bf16", "token_offsets", "experts_per_rank", "output",
                  "smlimit", "sync", "expert_start", "expert_count"),
          optional_params=("smlimit", "sync", "expert_start", "expert_count")),
    Event(version="0.01", kind="api_added",
          api="SymmAllocator.a2av_token_bf16_mxfp8_persistent",
          description="Persistent-grid variant of `a2av_token_bf16_mxfp8`.",
          params=("tokens_bf16", "token_offsets", "experts_per_rank", "output",
                  "smlimit", "sync"),
          optional_params=("smlimit", "sync")),
    Event(version="0.01", kind="api_added", api="SymmAllocator.a2av_wait",
          description="Wait kernel paired with `a2av_token_*` when sync=False.",
          params=()),

    # --- MoE combine ---

    Event(version="0.01", kind="api_added",
          api="SymmAllocator.combine_push3_bf16_bf16",
          description="RECOMMENDED MoE combine. 3-kernel split PUSH "
                      "(Phase 1 writes, Phase 2 barrier, Phase 3 local sum). "
                      "Double-buffered dest_bufs. Validated at EP=32.",
          params=("expert_outputs", "inverse_map", "topk_idx", "experts_per_rank",
                  "max_tokens_per_rank", "gate_weights", "smlimit"),
          optional_params=("gate_weights", "smlimit")),
    Event(version="0.01", kind="api_added",
          api="SymmAllocator.combine_bf16_bf16_push",
          description="Single-kernel PUSH combine with MC-atomic barrier.",
          params=("expert_outputs", "inverse_map", "topk_idx", "experts_per_rank",
                  "max_tokens_per_rank", "gate_weights", "smlimit"),
          optional_params=("gate_weights", "smlimit")),
    Event(version="0.01", kind="api_added",
          api="SymmAllocator.combine_bf16_bf16_lamport_push",
          description="PUSH combine with Lamport polling (no barrier).",
          params=("expert_outputs", "inverse_map", "topk_idx", "experts_per_rank",
                  "max_tokens_per_rank", "gate_weights", "smlimit"),
          optional_params=("gate_weights", "smlimit")),
    Event(version="0.01", kind="api_added",
          api="SymmAllocator.combine_v2_bf16_bf16",
          description="2-kernel combine (E22): Phase 1 pure data copy, Phase 2 "
                      "atomics + fp32 sum.",
          params=("expert_outputs", "token_offsets", "experts_per_rank", "smlimit"),
          optional_params=("smlimit",)),
    Event(version="0.01", kind="api_added",
          api="SymmAllocator.combine_bf16_bf16",
          description="PULL barrier variant — NOT STABLE at EP≥32; use "
                      "combine_push3 instead. Kept for diagnostics.",
          params=("expert_outputs", "token_offsets", "experts_per_rank",
                  "combine_temp", "smlimit"),
          optional_params=("combine_temp", "smlimit")),
    Event(version="0.01", kind="api_added",
          api="SymmAllocator.combine_mxfp8_bf16",
          description="MoE combine that reads mxfp8 expert outputs and "
                      "accumulates in bf16.",
          params=("expert_outputs", "token_offsets", "experts_per_rank",
                  "combine_temp", "smlimit"),
          optional_params=("combine_temp", "smlimit")),
    Event(version="0.01", kind="api_added", api="SymmAllocator.combine_wait",
          description="Wait kernel for split combine variants.",
          params=()),

    # --- SymmTensor ---

    Event(version="0.01", kind="api_added", api="SymmTensor",
          description="torch.Tensor subclass backed by a slice of the symmetric "
                      "pool. Reference-counted; freed when last view drops.",
          params=("data", "allocator")),

    # --- Module-level helpers (ubx.*) ---

    Event(version="0.01", kind="api_added", api="compute_token_offsets",
          description="Per-(token, expert) destination slot indices from a "
                      "routing matrix.",
          params=("routing", "experts_per_rank", "myrank", "nranks")),
    Event(version="0.01", kind="api_added", api="compute_combine_push_map",
          description="Inverse map + topk indices for combine_push3 / "
                      "combine_*_push.",
          params=("token_offsets", "experts_per_rank", "myrank", "nranks",
                  "capacity"),
          optional_params=("capacity",)),
    Event(version="0.01", kind="api_added", api="compute_dispatch_topk_map",
          description="[ntokens, topk_max] LUT for `a2av_token_bf16_bf16_topk`.",
          params=("token_offsets", "topk_max")),

    # ubx.ops.* helpers

    Event(version="0.01", kind="api_added", api="ops.allreduce",
          description="Module-level allreduce convenience (manages allocator).",
          params=("tensor_in", "dist_group", "gamma", "residual_in",
                  "residual_out", "eps"),
          optional_params=("dist_group", "gamma", "residual_in",
                           "residual_out", "eps")),
    Event(version="0.01", kind="api_added", api="ops.request_allocator",
          description="Get-or-create a SymmAllocator for `(dist_group, dtype, "
                      "max_tensor_size)`.",
          params=("max_tensor_size", "dtype", "dist_group")),
    Event(version="0.01", kind="api_added", api="ops.get_sym_tensor",
          description="Allocate a SymmTensor from the module-level allocator "
                      "registry.",
          params=("shape", "dtype", "dist_group")),
    Event(version="0.01", kind="api_added", api="ops.free_residual",
          description="Return a residual tensor's pool slot.",
          params=("tensor_in",)),
    Event(version="0.01", kind="api_added", api="ops.restore",
          description="Copy a regular tensor into a SymmTensor.",
          params=("tensor", "dist_group")),
    Event(version="0.01", kind="api_added", api="ops.mem_stats",
          description="Per-allocator memory statistics.",
          params=()),

    # ubx.fused.*

    Event(version="0.01", kind="api_added", api="fused.allreduce_fused",
          description="Allreduce fused with residual-add + RMSNorm.",
          params=("tensor_in", "gamma", "residual_in", "residual_out",
                  "eps", "dist_group"),
          optional_params=("residual_out", "eps", "dist_group")),

    # --- Meta APIs (this module) ---

    Event(version="0.01", kind="api_added", api="get_version",
          description="Return the current API contract version (string).",
          params=()),
    Event(version="0.01", kind="api_added", api="query_api",
          description="Check whether an API and its named params exist in the "
                      "current contract. Returns advice if not.",
          params=("name", "params"),
          optional_params=("params",)),
    Event(version="0.01", kind="api_added", api="query_supported_hw",
          description="Return supported transports / SM archs / NVLink-domain "
                      "constraints, either per-API or global aggregate.",
          params=("api_name",),
          optional_params=("api_name",)),

    # ============================================================
    # Future versions append events below this line.
    # ============================================================
]


# ----------------------------------------------------------------------
# Replay — derive current state from API_LOG
# ----------------------------------------------------------------------


def _compute_current_state(log: List[Event]) -> dict:
    """Replay the API_LOG to derive the current-state dict for every API.

    Returns ``{api_name: {description, params, optional_params,
                          transports, sm_min, added_in, removed_in,
                          renamed_to}}``.

    Entries for removed/renamed APIs are kept (with ``removed_in`` /
    ``renamed_to`` set) so `query_api` can advise users on history.
    """
    state: dict = {}
    for ev in log:
        if ev.kind == "api_added":
            state[ev.api] = {
                "description": ev.description,
                "params": list(ev.params),
                "optional_params": list(ev.optional_params),
                "transports": list(ev.transports),
                "sm_min": ev.sm_min,
                "added_in": ev.version,
                "removed_in": None,
                "renamed_to": None,
            }
        elif ev.kind == "api_removed":
            if ev.api in state:
                state[ev.api]["removed_in"] = ev.version
        elif ev.kind == "api_renamed":
            if ev.api in state and ev.new_name:
                state[ev.api]["renamed_to"] = ev.new_name
                # Carry the spec over to the new name.
                state[ev.new_name] = dict(state[ev.api])
                state[ev.new_name]["added_in"] = ev.version
                state[ev.new_name]["renamed_to"] = None
        elif ev.kind == "param_added":
            spec = state.get(ev.api)
            if spec is not None and ev.param and ev.param not in spec["params"]:
                spec["params"].append(ev.param)
                if ev.optional:
                    spec["optional_params"].append(ev.param)
        elif ev.kind == "param_removed":
            spec = state.get(ev.api)
            if spec is not None and ev.param:
                if ev.param in spec["params"]:
                    spec["params"].remove(ev.param)
                if ev.param in spec["optional_params"]:
                    spec["optional_params"].remove(ev.param)
        elif ev.kind == "param_renamed":
            spec = state.get(ev.api)
            if spec is not None and ev.param and ev.new_name:
                if ev.param in spec["params"]:
                    i = spec["params"].index(ev.param)
                    spec["params"][i] = ev.new_name
                if ev.param in spec["optional_params"]:
                    i = spec["optional_params"].index(ev.param)
                    spec["optional_params"][i] = ev.new_name
        elif ev.kind == "param_made_optional":
            spec = state.get(ev.api)
            if spec is not None and ev.param and ev.param in spec["params"]:
                if ev.param not in spec["optional_params"]:
                    spec["optional_params"].append(ev.param)
        elif ev.kind == "param_made_required":
            spec = state.get(ev.api)
            if spec is not None and ev.param in spec.get("optional_params", []):
                spec["optional_params"].remove(ev.param)
        elif ev.kind == "hw_added":
            spec = state.get(ev.api)
            if spec is not None and ev.transport and ev.transport not in spec["transports"]:
                spec["transports"].append(ev.transport)
        elif ev.kind == "hw_removed":
            spec = state.get(ev.api)
            if spec is not None and ev.transport in spec.get("transports", []):
                spec["transports"].remove(ev.transport)
        elif ev.kind == "sm_min_changed":
            spec = state.get(ev.api)
            if spec is not None and ev.sm_min:
                spec["sm_min"] = ev.sm_min
        elif ev.kind == "description_changed":
            spec = state.get(ev.api)
            if spec is not None and ev.description:
                spec["description"] = ev.description
        elif ev.kind == "behavior_changed":
            # Log-only event; no derived-state mutation.
            pass
        else:
            # Unknown event kind — ignore so an older ubx can still load
            # a log written by a newer one (forward-compat).
            pass
    return state


# Compute once at module load.
_CURRENT: dict = _compute_current_state(API_LOG)


# ----------------------------------------------------------------------
# Public query interface
# ----------------------------------------------------------------------


def get_version() -> str:
    """Return the API contract version (e.g. ``'0.01'``)."""
    return API_VERSION


def query_api(name: str, params: Optional[list] = None) -> dict:
    """Check whether ``name`` is a current ubx public API and optionally
    whether the caller's parameter names are still accepted.

    The contract is lenient on additive change:
      - Caller passes a subset of current params → OK.
      - Caller passes a param NOT in the current signature → not OK
        (either the API was renamed, the param was removed, or a typo).

    Returns:
      dict with keys:
        ok                 (bool) overall verdict
        status             ("ok" | "not_found" | "removed" | "renamed" |
                            "params_changed")
        name               (str) echoed query
        current_version    (str)
        added_in           (str|None)
        removed_in         (str|None)
        renamed_to         (str|None)
        current_params     (list[str])
        current_optional   (list[str])
        user_unknown_params (list[str])
        transports         (list[str]) — what hardware this API supports
        sm_min             (str)
        advice             (str)
    """
    out = {
        "ok": False,
        "status": "not_found",
        "name": name,
        "current_version": API_VERSION,
        "added_in": None,
        "removed_in": None,
        "renamed_to": None,
        "current_params": [],
        "current_optional": [],
        "user_unknown_params": [],
        "transports": [],
        "sm_min": None,
        "advice": "",
    }

    spec = _CURRENT.get(name)
    if spec is None:
        out["advice"] = (
            f"'{name}' is not a known ubx API in v{API_VERSION}. "
            f"Check the spelling or the catalog (api.version.md)."
        )
        return out

    out["added_in"] = spec["added_in"]
    out["removed_in"] = spec["removed_in"]
    out["renamed_to"] = spec["renamed_to"]
    out["current_params"] = list(spec["params"])
    out["current_optional"] = list(spec["optional_params"])
    out["transports"] = list(spec["transports"])
    out["sm_min"] = spec["sm_min"]

    if spec["removed_in"] is not None:
        out["status"] = "removed"
        out["advice"] = (
            f"'{name}' was removed in API v{spec['removed_in']}. "
            f"Last available in versions before that — pin ubx to an "
            f"earlier release if you still need this API."
        )
        return out

    if spec["renamed_to"] is not None:
        out["status"] = "renamed"
        out["advice"] = (
            f"'{name}' was renamed to '{spec['renamed_to']}'. Update "
            f"the call site or pin ubx to a version where '{name}' "
            f"was still available."
        )
        return out

    if params is None:
        out["ok"] = True
        out["status"] = "ok"
        out["advice"] = f"'{name}' is present in API v{API_VERSION}."
        return out

    current_set = set(spec["params"])
    user_set = set(params)
    unknown = sorted(user_set - current_set)
    if not unknown:
        out["ok"] = True
        out["status"] = "ok"
        out["advice"] = (
            f"'{name}' present in API v{API_VERSION}; all of your "
            f"parameter names are still accepted."
        )
        return out

    # Try to find a per-param history hint in the log.
    hints = []
    for p in unknown:
        for ev in API_LOG:
            if ev.api != name:
                continue
            if ev.kind == "param_removed" and ev.param == p:
                hints.append(f"'{p}': removed in v{ev.version}")
                break
            if ev.kind == "param_renamed" and ev.param == p:
                hints.append(f"'{p}': renamed to '{ev.new_name}' in v{ev.version}")
                break
    out["status"] = "params_changed"
    out["user_unknown_params"] = unknown
    if hints:
        out["advice"] = (
            f"'{name}' is present in v{API_VERSION} but these params "
            f"are not in the current signature: {unknown}. "
            + "; ".join(hints)
            + f". Current params: {list(spec['params'])}."
        )
    else:
        out["advice"] = (
            f"'{name}' is present in v{API_VERSION} but these params "
            f"are not in the current signature: {unknown}. "
            f"Current params: {list(spec['params'])}. Update the call "
            f"site, or pin ubx to an earlier version if you need the "
            f"old signature."
        )
    return out


def query_supported_hw(api_name: Optional[str] = None) -> dict:
    """Return supported hardware.

    If ``api_name`` is None: return the global aggregate — the union of
    transports supported across any current API + the build's SM-arch
    capability + NVLink-domain constraints.

    If ``api_name`` is given: return that API's per-API capability
    (transports + sm_min) — useful for `'this API supports nvlink and
    roce; that one only nvlink'` distinctions. Returns ``{"error":
    ...}`` if the API is unknown.
    """
    if api_name is not None:
        spec = _CURRENT.get(api_name)
        if spec is None:
            return {"error": f"'{api_name}' is not a known ubx API in "
                             f"v{API_VERSION}."}
        return {
            "api": api_name,
            "transports": list(spec["transports"]),
            "sm_min": spec["sm_min"],
            "removed_in": spec["removed_in"],
        }
    # Aggregate over all current (not-removed, not-renamed-out) APIs.
    all_transports: set = set()
    sm_mins: set = set()
    for name, spec in _CURRENT.items():
        if spec["removed_in"] is not None:
            continue
        if spec["renamed_to"] is not None:
            continue
        for t in spec["transports"]:
            all_transports.add(t)
        sm_mins.add(spec["sm_min"])
    return {
        "transports": sorted(all_transports),
        "sm_archs": ["9.0", "9.0a", "10.0", "10.0a"],
        "sm_min": min(sm_mins) if sm_mins else "9.0a",
        "sm_recommended": "10.0a",
        "nvlink_domain_min": 2,
        # Whether UC kernel paths and MC kernel paths work when
        # multicast is disabled (e.g. NCCL_NVLS_ENABLE=0): UC kernels
        # do (they only need P2P), MC kernels don't (they need the
        # multicast hardware).
        "unicast_support": True,
        "multicast_support": False,
    }
