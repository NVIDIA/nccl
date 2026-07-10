#!/usr/bin/env python3
"""
gen_shim.py — Generate C++ LD_PRELOAD shim wrappers for all NCCL functions
that involve ncclComm_t handles.

Usage:
    python3 gen_shim.py nccl.h [extra.h ...]                  # emit wrappers to stdout
    python3 gen_shim.py nccl.h [extra.h ...] --exports        # emit linker export map
    python3 gen_shim.py nccl.h [extra.h ...] --list           # print parsed functions + categories
    python3 gen_shim.py nccl.h [extra.h ...] --list --skipped # also show non-comm functions

Requires Python 3.7+. No external dependencies.

The generated file requires shim_core.h to provide:
    g_commHandles.toReal(...)
    g_regHandles.toReal(...)
    g_windowHandles.toReal(...)
    g_commHandles.remove(...)
"""

import re
import sys
import os
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Override tables — functions whose behaviour can't be inferred from the
# signature alone.
# ---------------------------------------------------------------------------

# Input comm handles that are consumed (destroyed) by a successful call.
DESTROY_PARAMS: Dict[str, Set[str]] = {
    "ncclCommDestroy": {"comm"},
    "ncclCommAbort":   {"comm"},
}

# Functions which this auto-generator will ignore.  Manually create shim in
# shim.cc.
MANUALLY_SHIMMED_FUNCS: Set[str] = {
    "ncclCommInitAll",
    "ncclCommInitRank",          # records CommInitParams for restore
    "ncclCommInitRankConfig",    # records CommInitParams for restore
    "ncclCommInitRankScalable",  # records CommInitParams for restore
    "ncclCommSplit",             # records split parent/color/key for restore
    "ncclCommShrink",            # records shrink parent/excluded ranks/flags for restore
    "ncclCommGrow",              # records CommInitParams for restore
    "ncclCommAbort",             # deregister_comm (keep bimap; drop from snapshot)
    "ncclCommDestroy",           # deregister_comm (keep bimap; drop from snapshot)
    "ncclCommFinalize",          # deregister_comm on success; handles ncclInProgress
    "ncclCommRegister",          # preserves synthetic registration handles
    "ncclCommDeregister",        # translates synthetic registration handles
    "ncclCommWindowRegister",    # preserves synthetic window handles
    "ncclCommWindowDeregister",  # translates synthetic window handles
}

EXTRA_EXPORTED_FUNCS: Set[str] = {
    "ncclCheckpointGetVersion",
    "ncclCheckpointPrepare",
    "ncclCheckpointRestore",
}

# These strings are transcribed directly into generated C++ string literals.
# Keep them escaped for C/C++ source.
RESTORE_UNSAFE_COMM_FUNCS: Dict[str, str] = {
    "ncclRedOpCreatePreMulSum": '"user-created reduction operators are not replayed after restore"',
    "ncclDevCommCreate": '"device communicators are not restored"',
    "ncclWinGetUserPtr": '"pre-checkpoint window user pointers are not valid after restore"',
}

RESTORE_UNSAFE_WINDOW_FUNCS: Dict[str, str] = {
    "ncclGetLsaMultimemDevicePointer": '"device window pointers are not valid after restore"',
    "ncclGetMultimemDevicePointer": '"device window pointers are not valid after restore"',
    "ncclGetLsaDevicePointer": '"device window pointers are not valid after restore"',
    "ncclGetPeerDevicePointer": '"device window pointers are not valid after restore"',
}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Param:
    raw: str           # original text, e.g. "const ncclComm_t comm"
    type_: str         # normalised type,  e.g. "const ncclComm_t"
    name: str          # parameter name,   e.g. "comm"
    is_in_comm: bool   # ncclComm_t by value (input handle)
    is_out_comm: bool  # ncclComm_t* (output handle — receives a new comm)
    is_in_window: bool # ncclWindow_t by value (input handle)
    is_reg_handle: bool # opaque void* registration handle input


@dataclass
class NcclFunction:
    name: str
    ret_type: str
    params: List[Param]

    @property
    def in_comms(self) -> List[Param]:
        return [p for p in self.params if p.is_in_comm]

    @property
    def out_comms(self) -> List[Param]:
        return [p for p in self.params if p.is_out_comm]

    @property
    def in_windows(self) -> List[Param]:
        return [p for p in self.params if p.is_in_window]

    @property
    def reg_handles(self) -> List[Param]:
        return [p for p in self.params if p.is_reg_handle]

    @property
    def skip(self) -> bool:
        has_translated_handle = any(
            p.is_in_comm or p.is_out_comm or p.is_in_window or p.is_reg_handle
            for p in self.params
        )
        return (
            (self.name in MANUALLY_SHIMMED_FUNCS)
            or not has_translated_handle
        )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*",  " ", text)
    return text


def normalize_type(raw: str) -> str:
    """Collapse whitespace and attach * directly to the left token."""
    t = raw.strip()
    t = re.sub(r"\s+", " ", t)          # collapse interior spaces
    t = re.sub(r"\s*\*\s*", "*", t)     # "type * " -> "type*"
    return t


def parse_param(raw: str) -> Optional[Param]:
    """
    Parse one C parameter declaration.
    Returns None for 'void', '...', or empty.
    """
    s = raw.strip()
    if not s or s in ("void", "..."):
        return None

    # The parameter name is the last identifier (optionally followed by []).
    m = re.search(r"\b(\w+)\s*(?:\[\w*\])?\s*$", s)
    if not m:
        return None

    name      = m.group(1)
    type_norm = normalize_type(s[: m.start()])

    # Input:  "ncclComm_t" or "const ncclComm_t"
    # Output: "ncclComm_t*" (pointer — caller receives a new comm handle)
    is_in_comm  = type_norm in ("ncclComm_t", "const ncclComm_t")
    is_out_comm = type_norm == "ncclComm_t*"
    is_in_window = type_norm in ("ncclWindow_t", "const ncclWindow_t")
    is_reg_handle = type_norm == "void*" and name == "handle"

    return Param(raw=s, type_=type_norm, name=name,
                 is_in_comm=is_in_comm, is_out_comm=is_out_comm,
                 is_in_window=is_in_window, is_reg_handle=is_reg_handle)


def parse_header(path: str) -> List[NcclFunction]:
    """
    Extract all ncclResult_t function declarations from the header.

    Handles:
    - Optional NCCL/NCCL device API macros before or after return type
    - Multi-line parameter lists
    - Duplicate declarations (keeps first occurrence)
    - pnccl* pointer variants are skipped entirely
    """
    with open(path) as fh:
        text = fh.read()
    text = strip_comments(text)

    qualifier = r"(?:\b[A-Z_][A-Z0-9_]*\b|__\w+__)"
    pattern = re.compile(
        rf"(?:{qualifier}\s+)*"       # optional macros before return type
        r"\bncclResult_t\b\s+"
        rf"(?:{qualifier}\s+)*"       # optional macros after return type
        r"(p?nccl\w+)"                # function name (nccl* or pnccl*)
        r"\s*\(([^)]*)\)\s*;",        # parameter list (no nested parens needed)
        re.DOTALL,
    )

    fns: List[NcclFunction] = []
    seen: Set[str] = set()

    for m in pattern.finditer(text):
        name = m.group(1)

        if name in seen or name.startswith("pnccl"):
            continue
        seen.add(name)

        raw_params = [p.strip() for p in m.group(2).split(",")]
        params: List[Param] = []
        for rp in raw_params:
            p = parse_param(rp)
            if p is not None:
                params.append(p)

        fns.append(NcclFunction(name=name, ret_type="ncclResult_t", params=params))

    return fns


def parse_headers(paths: List[str]) -> List[NcclFunction]:
    fns: List[NcclFunction] = []
    seen: Set[str] = set()
    for path in paths:
        for fn in parse_header(path):
            if fn.name in seen:
                continue
            seen.add(fn.name)
            fns.append(fn)
    return fns

# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

def fn_ptr_type(fn: NcclFunction) -> str:
    """
    Build a C++ function-pointer type for the *real* underlying function.

    Using an explicit typedef (rather than decltype(&fn)) avoids the
    self-referential lookup problem when the function is being defined.
    """
    types = ", ".join(p.type_ for p in fn.params) if fn.params else "void"
    return f"{fn.ret_type}(*)({types})"


def comm_label(fn: NcclFunction) -> str:
    """Human-readable per-parameter handle roles."""
    parts = []
    for p in fn.params:
        if p.is_in_comm:
            role = "DESTROY" if (fn.name in DESTROY_PARAMS and p.name in DESTROY_PARAMS[fn.name]) else "USE"
            parts.append(f"{role}:{p.name}")
        elif p.is_out_comm:
            parts.append(f"CREATE:{p.name}")
        elif p.is_in_window:
            parts.append(f"USE_WINDOW:{p.name}")
        elif p.is_reg_handle:
            parts.append(f"USE_REG:{p.name}")
    return ", ".join(parts) if parts else ""


def emit_function(fn: NcclFunction) -> str:
    """Return the C wrapper body for one function."""
    if fn.name in MANUALLY_SHIMMED_FUNCS:
        return ""
    lines  = [f"// {comm_label(fn)}"]

    # -- Signature (identical to original) -------------------------------------
    decl   = ", ".join(p.raw for p in fn.params) if fn.params else "void"
    lines += [f"{fn.ret_type} {fn.name}({decl}) {{"]

    # -- dlsym lookup ----------------------------------------------------------
    ptr_t = fn_ptr_type(fn)
    lines += [
        f"  using real_t = {ptr_t};",
        "  static real_t real_fn = nullptr;",
        f'  NCCLCHECK(resolveRealFunction("{fn.name}", &real_fn));',
    ]

    # -- Temporaries for output comm handles -----------------------------------
    for p in fn.out_comms:
        lines.append(f"  ncclComm_t real_{p.name} = nullptr;")

    # -- Translate input handles -----------------------------------------------
    lines.append("  ncclResult_t ret;")
    for p in fn.in_comms:
        lines += [
            f"  ncclComm_t real_{p.name} = {p.name};",
            f'  ret = g_commHandles.toReal({p.name}, &real_{p.name});',
            "  if (ret != ncclSuccess) return ret;",
        ]
    for p in fn.in_windows:
        lines += [
            f"  ncclWindow_t real_{p.name} = {p.name};",
            f'  ret = g_windowHandles.toReal({p.name}, &real_{p.name});',
            "  if (ret != ncclSuccess) return ret;",
        ]
    for p in fn.reg_handles:
        lines += [
            f"  void* real_{p.name} = {p.name};",
            f'  ret = g_regHandles.toReal({p.name}, &real_{p.name});',
            "  if (ret != ncclSuccess) return ret;",
        ]

    # -- Build translated call arguments ---------------------------------------
    call_args = []
    for p in fn.params:
        if p.is_in_comm:
            call_args.append(f"real_{p.name}")
        elif p.is_out_comm:
            call_args.append(f"&real_{p.name}")
        elif p.is_in_window:
            call_args.append(f"real_{p.name}")
        elif p.is_reg_handle:
            call_args.append(f"real_{p.name}")
        else:
            call_args.append(p.name)
    args = ", ".join(call_args)

    # -- Call ------------------------------------------------------------------
    # USE functions with no post-call work can return directly.
    need_ret = (
        bool(fn.out_comms)
        or fn.name in DESTROY_PARAMS
        or fn.name in RESTORE_UNSAFE_COMM_FUNCS
        or fn.name in RESTORE_UNSAFE_WINDOW_FUNCS
    )
    if need_ret:
        lines.append(f"  ret = real_fn({args});")
    else:
        lines += [f"  ret = real_fn({args});", "  return ret;", "}"]
        return "\n".join(lines)

    # -- Post-call: register new synthetic handles -----------------------------
    for p in fn.out_comms:
        lines += [
            f"  if (real_{p.name} != nullptr)",
            f"    *{p.name} = g_commHandles.makeSynthetic(real_{p.name});",
        ]

    # -- Post-call: mark restore-unsafe APIs ----------------------------------
    if fn.name in RESTORE_UNSAFE_COMM_FUNCS:
        reason = RESTORE_UNSAFE_COMM_FUNCS[fn.name]
        if not fn.in_comms:
            raise RuntimeError(f"{fn.name} is configured as comm-unsafe but has no comm parameter")
        comm = fn.in_comms[0].name
        lines += [
            f"  markCommRestoreUnsafe({comm}, {reason});",
        ]

    if fn.name in RESTORE_UNSAFE_WINDOW_FUNCS:
        reason = RESTORE_UNSAFE_WINDOW_FUNCS[fn.name]
        if not fn.in_windows:
            raise RuntimeError(f"{fn.name} is configured as window-unsafe but has no window parameter")
        window = fn.in_windows[0].name
        lines += [
            f"  markWindowCommRestoreUnsafe({window}, {reason});",
        ]

    # -- Post-call: remove destroyed comm mappings -----------------------------
    if fn.name in DESTROY_PARAMS:
        for pname in sorted(DESTROY_PARAMS[fn.name]):
            lines += [
                f"  if (ret == ncclSuccess)",
                f"    g_commHandles.remove({pname});",
            ]

    lines += ["  return ret;", "}"]
    return "\n".join(lines)


def emit_file(fns: List[NcclFunction], header_path: str) -> str:
    """Emit the complete generated .cpp file."""

    comm_fns = [f for f in fns if not f.skip]
    skipped  = [f for f in fns if f.skip]

    out: List[str] = [
        "// AUTO-GENERATED by gen_shim.py — DO NOT EDIT",
        "",
        '#include <nccl.h>',
        '#include "nccl_device/core.h"',
        '#include "shim_core.h"',
        "#include <dlfcn.h>",
        "",
        "using namespace nccl_checkpoint;",
        "",
        'extern "C" {',
        "",
    ]

    for fn in fns:
        if not fn.skip:
            out.append(emit_function(fn))
            out.append("")

    # Skipped functions are either manually shimmed elsewhere or do not carry
    # handles that this generator translates.
    if skipped:
        out.append("// Skipped by this generator:")
        for fn in skipped:
            out.append(f"//   {fn.ret_type} {fn.name}(...)")
        out.append("")

    out.append('} // extern "C"')
    out.append("")
    return "\n".join(out)


def emit_export_map(fns: List[NcclFunction]) -> str:
    exported = {
        fn.name
        for fn in fns
        if not fn.skip or fn.name in MANUALLY_SHIMMED_FUNCS
    }
    exported.update(EXTRA_EXPORTED_FUNCS)

    out = [
        "{",
        "  global:",
    ]
    out.extend(f"    {name};" for name in sorted(exported))
    out.extend([
        "  local:",
        "    *;",
        "};",
        "",
    ])
    return "\n".join(out)

# ---------------------------------------------------------------------------
# --list mode
# ---------------------------------------------------------------------------

def cmd_list(fns: List[NcclFunction], show_skipped: bool) -> None:
    skipped = 0
    for fn in fns:
        if fn.skip:
            skipped += 1
            if show_skipped:
                print(f"skipped  {fn.name}")
        else:
            print(f"{comm_label(fn):40s}  {fn.name}")
    if skipped and not show_skipped:
        print(f"\n{skipped} functions not shimmed")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    argv = sys.argv[1:]

    if not argv or "-h" in argv or "--help" in argv:
        print(__doc__)
        sys.exit(0 if argv else 1)

    list_mode    = "--list"     in argv
    show_skipped = "--skipped"  in argv
    export_mode  = "--exports"  in argv
    header_paths = [arg for arg in argv if not arg.startswith("--")]
    if not header_paths:
        print("error: at least one header path is required", file=sys.stderr)
        sys.exit(1)

    for header_path in header_paths:
        if not os.path.isfile(header_path):
            print(f"error: {header_path}: file not found", file=sys.stderr)
            sys.exit(1)

    fns = parse_headers(header_paths)

    if not fns:
        print(f"warning: no ncclResult_t functions found in {', '.join(header_paths)}", file=sys.stderr)
        print("         Verify the header uses the expected declaration style.", file=sys.stderr)
        sys.exit(1)

    comm_fns = [f for f in fns if not f.skip]
    print(
        f"// Parsed {len(fns)} functions: {len(comm_fns)} intercepted, "
        f"{len(fns)-len(comm_fns)} skipped",
        file=sys.stderr,
    )

    if export_mode:
        sys.stdout.write(emit_export_map(fns))
    elif list_mode or show_skipped:
        cmd_list(fns, show_skipped=show_skipped)
    else:
        sys.stdout.write(emit_file(fns, ", ".join(header_paths)))



if __name__ == "__main__":
    main()
