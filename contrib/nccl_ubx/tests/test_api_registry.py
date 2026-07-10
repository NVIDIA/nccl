"""Tests for the UB-X public-API registry and query interface.

Pure-Python tests — no GPU, no distributed setup. Exercises:
  - `ubx.version` / `ubx.get_version()` returns the contract version
  - `ubx.query_api(name, params=...)` for every API listed in v0.01
  - `ubx.query_api(name)` for several made-up names (negative cases)
  - `ubx.query_api(name, params=[bad])` (param drift case)
  - `ubx.query_supported_hw()` aggregate + per-API
  - `_compute_current_state` correctly replays the full v0.01 log
  - `_compute_current_state` correctly handles every documented event
    kind (synthetic future events appended to a copy of the log).

The list of v0.01 APIs lives in this file alongside the tests so that
adding a new API in `_api_registry.py` without updating this file
fails the regression — keeping the tests honest as the registry grows.
"""

import pytest

# Import directly from the registry module so the tests run on a host
# without torch installed.
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ubx"))
import _api_registry as r  # noqa: E402


# ----------------------------------------------------------------------
# Canonical v0.01 surface — every public API the registry must contain.
#
# Each entry: (name, expected_params_subset). The "subset" is what
# downstream code might realistically call with — we check that
# query_api accepts these as valid (not that the spec lists ONLY these).
# ----------------------------------------------------------------------

V_0_01_APIS = [
    # SymmAllocator class + core methods
    ("SymmAllocator", ["size_bytes", "device", "dist_group"]),
    ("SymmAllocator.close", []),
    ("SymmAllocator.create_tensor", ["shape", "dtype"]),
    ("SymmAllocator.barrier", []),

    # Allreduce
    ("SymmAllocator.allreduce", ["tensor_in"]),
    ("SymmAllocator.allreduce_mc", ["tensor_in"]),
    ("SymmAllocator.allreduce_uc", ["tensor_in"]),
    ("SymmAllocator.allreduce_lamport", ["tensor_in"]),

    # Allgather
    ("SymmAllocator.allgather", ["tensor_in"]),
    ("SymmAllocator.allgather_uc", ["tensor_in"]),

    # Alltoall (fixed-size)
    ("SymmAllocator.alltoall", ["tensor_in"]),
    ("SymmAllocator.alltoall_lamport", ["tensor_in"]),
    ("SymmAllocator.alltoall_auto", ["tensor_in"]),

    # Alltoallv
    ("SymmAllocator.alltoallv", ["tensor_in", "output_split_sizes",
                                  "input_split_sizes"]),
    ("SymmAllocator.alltoallv_prepare", ["output_split_sizes",
                                          "input_split_sizes", "dtype"]),
    ("SymmAllocator.alltoallv_run", ["tensor_in", "state"]),

    # MoE fused dispatch
    ("SymmAllocator.a2av_token_bf16_bf16",
     ["tokens_bf16", "token_offsets", "experts_per_rank"]),
    ("SymmAllocator.a2av_token_bf16_bf16_topk",
     ["tokens_bf16", "topk_expert", "topk_slot", "experts_per_rank"]),
    ("SymmAllocator.a2av_token_bf16_mxfp8",
     ["tokens_bf16", "token_offsets", "experts_per_rank", "output"]),
    ("SymmAllocator.a2av_token_bf16_mxfp8_persistent",
     ["tokens_bf16", "token_offsets", "experts_per_rank", "output"]),
    ("SymmAllocator.a2av_wait", []),

    # MoE combine
    ("SymmAllocator.combine_push3_bf16_bf16",
     ["expert_outputs", "inverse_map", "topk_idx", "experts_per_rank",
      "max_tokens_per_rank"]),
    ("SymmAllocator.combine_bf16_bf16_push",
     ["expert_outputs", "inverse_map", "topk_idx", "experts_per_rank",
      "max_tokens_per_rank"]),
    ("SymmAllocator.combine_bf16_bf16_lamport_push",
     ["expert_outputs", "inverse_map", "topk_idx", "experts_per_rank",
      "max_tokens_per_rank"]),
    ("SymmAllocator.combine_v2_bf16_bf16",
     ["expert_outputs", "token_offsets", "experts_per_rank"]),
    ("SymmAllocator.combine_bf16_bf16",
     ["expert_outputs", "token_offsets", "experts_per_rank"]),
    ("SymmAllocator.combine_mxfp8_bf16",
     ["expert_outputs", "token_offsets", "experts_per_rank"]),
    ("SymmAllocator.combine_wait", []),

    # SymmTensor
    ("SymmTensor", ["data", "allocator"]),

    # Module-level helpers
    ("compute_token_offsets",
     ["routing", "experts_per_rank", "myrank", "nranks"]),
    ("compute_combine_push_map",
     ["token_offsets", "experts_per_rank", "myrank", "nranks"]),
    ("compute_dispatch_topk_map", ["token_offsets", "topk_max"]),

    # ubx.ops.*
    ("ops.allreduce", ["tensor_in"]),
    ("ops.request_allocator", ["max_tensor_size", "dtype", "dist_group"]),
    ("ops.get_sym_tensor", ["shape", "dtype", "dist_group"]),
    ("ops.free_residual", ["tensor_in"]),
    ("ops.restore", ["tensor", "dist_group"]),
    ("ops.mem_stats", []),

    # ubx.fused.*
    ("fused.allreduce_fused", ["tensor_in", "gamma", "residual_in"]),

    # Meta APIs
    ("get_version", []),
    ("query_api", ["name"]),
    ("query_supported_hw", []),
]


# ----------------------------------------------------------------------
# Sanity: version + registry shape
# ----------------------------------------------------------------------


def test_api_version_format():
    """API_VERSION should be a non-empty string matching X.YZ shape."""
    assert isinstance(r.API_VERSION, str)
    assert r.API_VERSION  # non-empty
    parts = r.API_VERSION.split(".")
    assert len(parts) == 2, f"expected X.YZ, got {r.API_VERSION!r}"
    assert parts[0].isdigit() and parts[1].isdigit()


def test_get_version_matches_constant():
    assert r.get_version() == r.API_VERSION


def test_api_log_is_a_list_of_events():
    assert isinstance(r.API_LOG, list)
    assert len(r.API_LOG) > 0
    for ev in r.API_LOG:
        assert isinstance(ev, r.Event), f"non-Event in API_LOG: {ev!r}"


def test_current_state_replay_matches_v_0_01_snapshot():
    """Replaying API_LOG must produce one entry per v0.01 API listed
    in V_0_01_APIS. Catches accidental dropouts on either side."""
    current_names = set(r._CURRENT.keys())
    expected_names = {name for name, _ in V_0_01_APIS}
    missing = expected_names - current_names
    extra = current_names - expected_names
    assert not missing, (
        f"V_0_01_APIS lists names not in the registry: {sorted(missing)}"
    )
    assert not extra, (
        f"Registry has APIs not listed in V_0_01_APIS — add them to the "
        f"test or remove from registry: {sorted(extra)}"
    )


# ----------------------------------------------------------------------
# query_api — positive cases
# ----------------------------------------------------------------------


@pytest.mark.parametrize("api,params", V_0_01_APIS,
                         ids=[name for name, _ in V_0_01_APIS])
def test_query_api_returns_ok_for_every_v_0_01_api(api, params):
    """Every v0.01 API must return ok=True when called with a subset
    of its current parameters (or no params at all)."""
    res = r.query_api(api, params=params if params else None)
    assert res["ok"] is True, (
        f"query_api({api!r}, params={params!r}) returned not ok: {res}"
    )
    assert res["status"] == "ok"
    assert res["name"] == api
    assert res["added_in"] == "0.01"
    assert res["removed_in"] is None
    assert res["renamed_to"] is None
    assert res["sm_min"] == "9.0a"
    assert res["transports"] == ["nvlink"]
    # The full current_params should at least cover what the user passed.
    if params:
        assert set(params).issubset(set(res["current_params"])), (
            f"user params {params} not all present in current_params "
            f"{res['current_params']}"
        )


def test_query_api_no_params_returns_ok():
    """Passing params=None skips the per-param check."""
    res = r.query_api("SymmAllocator.alltoallv")
    assert res["ok"] is True
    assert res["status"] == "ok"
    assert "smlimit" in res["current_params"]


# ----------------------------------------------------------------------
# query_api — negative cases
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bogus_name", [
    "does_not_exist",
    "SymmAllocator.totally_made_up",
    "foo_bar_baz",
    "alltoall",  # missing SymmAllocator. prefix
    "ops.alltoallv",  # not in ops.* namespace
    "",
])
def test_query_api_unknown_name_returns_not_found(bogus_name):
    res = r.query_api(bogus_name)
    assert res["ok"] is False
    assert res["status"] == "not_found"
    assert res["added_in"] is None
    assert "not a known ubx API" in res["advice"]


def test_query_api_unknown_param_returns_params_changed():
    """User passes a param NOT in the current signature → flag it."""
    res = r.query_api(
        "SymmAllocator.alltoallv",
        params=["tensor_in", "totally_made_up_param"],
    )
    assert res["ok"] is False
    assert res["status"] == "params_changed"
    assert res["user_unknown_params"] == ["totally_made_up_param"]
    assert "totally_made_up_param" in res["advice"]
    # The advice should also include the current params so the user
    # can see what to migrate to.
    for p in ("tensor_in", "output_split_sizes",
              "input_split_sizes", "smlimit"):
        assert p in res["advice"]


def test_query_api_multiple_unknown_params_listed_sorted():
    res = r.query_api(
        "SymmAllocator.alltoall",
        params=["tensor_in", "zzz_param", "aaa_param"],
    )
    assert res["status"] == "params_changed"
    # Sorted alphabetically.
    assert res["user_unknown_params"] == ["aaa_param", "zzz_param"]


def test_query_api_extra_optional_params_are_ok():
    """User passing only `tensor_in` to alltoallv (omits the optional
    `smlimit` and the required splits — the latter is the user's
    problem at call time, not a registry concern)."""
    res = r.query_api(
        "SymmAllocator.alltoallv",
        params=["tensor_in"],
    )
    # Still ok because we're checking the user's subset is a subset
    # of the current params, not that they passed all required ones.
    assert res["ok"] is True


# ----------------------------------------------------------------------
# query_supported_hw
# ----------------------------------------------------------------------


def test_query_supported_hw_global_aggregate():
    hw = r.query_supported_hw()
    assert hw["transports"] == ["nvlink"]
    assert "9.0a" in hw["sm_archs"]
    assert "10.0a" in hw["sm_archs"]
    assert hw["sm_min"] == "9.0a"
    assert hw["sm_recommended"] == "10.0a"
    assert hw["unicast_support"] is True
    assert hw["multicast_support"] is False


def test_query_supported_hw_per_api():
    hw = r.query_supported_hw("SymmAllocator.alltoall")
    assert hw["api"] == "SymmAllocator.alltoall"
    assert hw["transports"] == ["nvlink"]
    assert hw["sm_min"] == "9.0a"
    assert hw["removed_in"] is None


def test_query_supported_hw_unknown_api_returns_error():
    hw = r.query_supported_hw("does_not_exist")
    assert "error" in hw
    assert "not a known ubx API" in hw["error"]


def test_query_supported_hw_aggregate_is_immutable():
    """Mutating the returned dict must NOT affect future calls."""
    hw = r.query_supported_hw()
    hw["transports"].append("hacked")
    hw2 = r.query_supported_hw()
    assert "hacked" not in hw2["transports"]


# ----------------------------------------------------------------------
# Replay tests — every documented event kind
# ----------------------------------------------------------------------


def _replay_with_extra(*events):
    """Helper: replay API_LOG + a synthetic future event list."""
    return r._compute_current_state(list(r.API_LOG) + list(events))


def test_replay_param_added_optional():
    state = _replay_with_extra(
        r.Event(version="0.02", kind="param_added",
                api="SymmAllocator.alltoallv",
                param="timeout_ms", optional=True),
    )
    assert "timeout_ms" in state["SymmAllocator.alltoallv"]["params"]
    assert "timeout_ms" in state["SymmAllocator.alltoallv"]["optional_params"]


def test_replay_param_added_required():
    state = _replay_with_extra(
        r.Event(version="0.02", kind="param_added",
                api="SymmAllocator.alltoallv",
                param="must_have", optional=False),
    )
    assert "must_have" in state["SymmAllocator.alltoallv"]["params"]
    assert "must_have" not in state["SymmAllocator.alltoallv"]["optional_params"]


def test_replay_param_removed():
    state = _replay_with_extra(
        r.Event(version="0.02", kind="param_removed",
                api="SymmAllocator.alltoallv",
                param="smlimit"),
    )
    assert "smlimit" not in state["SymmAllocator.alltoallv"]["params"]
    assert "smlimit" not in state["SymmAllocator.alltoallv"]["optional_params"]


def test_replay_param_renamed():
    state = _replay_with_extra(
        r.Event(version="0.02", kind="param_renamed",
                api="SymmAllocator.alltoallv",
                param="smlimit", new_name="sm_cap"),
    )
    params = state["SymmAllocator.alltoallv"]["params"]
    assert "smlimit" not in params
    assert "sm_cap" in params
    # Position should be preserved.
    assert params.index("sm_cap") == 3


def test_replay_param_made_optional():
    state = _replay_with_extra(
        r.Event(version="0.02", kind="param_made_optional",
                api="SymmAllocator.alltoallv",
                param="input_split_sizes"),
    )
    opt = state["SymmAllocator.alltoallv"]["optional_params"]
    assert "input_split_sizes" in opt


def test_replay_param_made_required():
    state = _replay_with_extra(
        r.Event(version="0.02", kind="param_made_required",
                api="SymmAllocator.alltoallv",
                param="smlimit"),
    )
    opt = state["SymmAllocator.alltoallv"]["optional_params"]
    assert "smlimit" not in opt


def test_replay_hw_added():
    state = _replay_with_extra(
        r.Event(version="0.02", kind="hw_added",
                api="SymmAllocator.alltoall",
                transport="roce"),
    )
    assert state["SymmAllocator.alltoall"]["transports"] == ["nvlink", "roce"]


def test_replay_hw_removed():
    state = _replay_with_extra(
        r.Event(version="0.02", kind="hw_added",
                api="SymmAllocator.alltoall", transport="roce"),
        r.Event(version="0.03", kind="hw_removed",
                api="SymmAllocator.alltoall", transport="nvlink"),
    )
    assert state["SymmAllocator.alltoall"]["transports"] == ["roce"]


def test_replay_sm_min_changed():
    state = _replay_with_extra(
        r.Event(version="0.02", kind="sm_min_changed",
                api="SymmAllocator.alltoall", sm_min="10.0a"),
    )
    assert state["SymmAllocator.alltoall"]["sm_min"] == "10.0a"


def test_replay_api_removed_sets_removed_in():
    state = _replay_with_extra(
        r.Event(version="0.02", kind="api_removed",
                api="SymmAllocator.combine_bf16_bf16"),
    )
    assert state["SymmAllocator.combine_bf16_bf16"]["removed_in"] == "0.02"


def test_replay_api_renamed_carries_spec_to_new_name():
    state = _replay_with_extra(
        r.Event(version="0.02", kind="api_renamed",
                api="SymmAllocator.alltoallv", new_name="SymmAllocator.av_all"),
    )
    assert state["SymmAllocator.alltoallv"]["renamed_to"] == "SymmAllocator.av_all"
    assert "SymmAllocator.av_all" in state
    assert state["SymmAllocator.av_all"]["added_in"] == "0.02"
    # Params should be carried over.
    assert "output_split_sizes" in state["SymmAllocator.av_all"]["params"]


def test_replay_description_changed():
    state = _replay_with_extra(
        r.Event(version="0.02", kind="description_changed",
                api="SymmAllocator.alltoall",
                description="NEW description for alltoall"),
    )
    assert state["SymmAllocator.alltoall"]["description"] == \
        "NEW description for alltoall"


def test_replay_unknown_kind_is_ignored_forward_compat():
    """An older replayer must silently skip unknown event kinds (so
    an older ubx can load a log written by a newer one without
    crashing)."""
    state = _replay_with_extra(
        r.Event(version="9.99", kind="totally_new_event_type_from_the_future",
                api="SymmAllocator.alltoall"),
    )
    # State should be identical to baseline.
    assert state["SymmAllocator.alltoall"] == \
        r._CURRENT["SymmAllocator.alltoall"]


# ----------------------------------------------------------------------
# Bonus: query_api advice after a future change has happened
# ----------------------------------------------------------------------


def test_query_api_advises_when_param_was_removed():
    """If the log contains a future param_removed, calling query_api
    with that param should surface the version hint."""
    # Build a temporarily-extended registry by monkey-patching _CURRENT.
    original_log = r.API_LOG
    original_state = r._CURRENT
    try:
        new_log = list(original_log) + [
            r.Event(version="0.07", kind="param_removed",
                    api="SymmAllocator.alltoallv", param="smlimit"),
        ]
        r.API_LOG = new_log
        r._CURRENT = r._compute_current_state(new_log)

        res = r.query_api(
            "SymmAllocator.alltoallv",
            params=["tensor_in", "smlimit"],
        )
        assert res["status"] == "params_changed"
        assert res["user_unknown_params"] == ["smlimit"]
        assert "removed in v0.07" in res["advice"]
    finally:
        r.API_LOG = original_log
        r._CURRENT = original_state
