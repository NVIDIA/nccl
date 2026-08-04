#!/usr/bin/env python3
"""Generate per-instantiation .cu files for sg's scattergather_kernel_p2p
template so ninja compiles them in parallel.

Background: sg.cu's __global__ kernel template scattergather_kernel_p2p
is referenced by the host dispatcher (LAUNCH_SG_P2P_KERNEL + SWITCH_P2P_PAIR
+ SWITCH_P2P_KUNROLL + SWITCH_FLAT + SWITCH_F8MODE + P2P_L2) for every
(Nlr, kWP, kUnroll, kF8, kFlat, kUnifiedView, kLogZ) combination:

  3 (Nlr, kWP) pairs × 3 kUnroll × 2 kF8 × 4 kFlat × 2 kUnifiedView × 2 kLogZ
  = 288 instantiations, grouped into 32 .cu files (one per
  (kFlat, kF8, kUnifiedView, kLogZ) combo).

Compiling all of them in one TU (sg.cu) is slow — ptxas register
allocation dominates. This script emits one .cu file per
(kFlat, kF8, kUnifiedView, kLogZ) combo, each holding the 9
(Nlr, kWP, kUnroll) explicit instantiations, plus a decls header with
`extern template` declarations for the dispatcher.

Sibling to scripts/gen_ep_inst.py (same pattern, different kernel family).
Standalone — does not import gen_ep_inst.py, so the two generators can be
run/maintained independently.

Generated files (under <repo>/build/gen/sg/):
  sg_p2p_flat{F}_f8{0|1}_uv{0|1}_lz{0|1}.cu  — 32 files; each holds 9
                                                  instantiations.
  sg_p2p_extern_decls.cuh                   — all `extern template`
                                                  declarations.

Total: 32 .cu files. Grouped-by-(kFlat,kF8,kUnifiedView,kLogZ) granularity
keeps the file count under the ninja argv limit; ptxas still parallelizes
across the 32 files.

Manifest (under <repo>/build/gen/):
  sg_manifest.txt            — one path per generated .cu, for setup.py.
"""
import os
import sys
import io


# ---- Template-arg enumerations --------------------------------------------

# SWITCH_P2P_PAIR (sg.cu) enumerates these (Nlr, kWP) pairs. kNumThreads is
# derived as kWP * (2 * Nlr - 1) * 32. Only 3 valid pairs fit within the
# 1024-thread-per-block limit (invalid combos like (Nlr=8, kWP=10) → 4800
# threads are never compiled and don't trigger ptxas `maxntid` warnings).
SG_P2P_PAIRS = [
    (2, 10),  # kNumThreads = 10 * 3 * 32 = 960
    (4, 4),   # kNumThreads = 4  * 7 * 32 = 896
    (8, 2),   # kNumThreads = 2  * 15 * 32 = 960
]

# SWITCH_P2P_KUNROLL (sg.cu) enumerates these kUnroll values.
SG_P2P_KUNROLL = [2, 4, 8]

# SWITCH_FLAT (sg.cu) enumerates these kFlat values (0..3).
SG_P2P_FLAT = [0, 1, 2, 3]

# SWITCH_F8MODE (sg.cu) enumerates these kF8 values (false/true → 0/1).
SG_P2P_F8 = [False, True]

# P2P_L2 (sg.cu) enumerates these kUnifiedView and kLogZ values via outer
# if/else branches (host-side decision based on tensor count/Z power-of-2).
SG_P2P_UNIFIED_VIEW = [True, False]
SG_P2P_LOG_Z = [True, False]


# ---- Kernel param list (verbatim from sg_kernels.cuh, comments stripped) ----

# scattergather_kernel_p2p<Nlr, kWP, kNumThreads, kUnroll, kF8, kFlat,
# kUnifiedView, kLogZ> signature. kNumThreads is a derived template arg
# (= kWP * (2 * Nlr - 1) * 32); the generator emits the literal computed
# value, not the expression — same as EP's kNumThreads=1024 convention.
SG_P2P_PARAMS = (
    'uint64_t *args, size_t nargs, void *gin_win_ptr, void **p2p_ptrs, '
    'int *gmem_barrier, uint64_t *debug_buf, ncclWindow_t gin_win, '
    'const int nvl_ring, const int rdma_ring, const size_t rdma_unroll, '
    'int round_n, int rank, int num_ranks, ncclDevComm dev_comm, '
    'const int scatter_dim'
)


def _knumthreads(nlr, kwp):
    """Compute the derived kNumThreads template arg for a (Nlr, kWP) pair."""
    return kwp * (2 * nlr - 1) * 32


def _write_if_different(path, content):
    """Write `content` to `path` only if the existing file's content differs
    (or the file doesn't exist). Avoids bumping mtimes when the generator is
    re-run with unchanged template args, so ninja doesn't needlessly recompile
    the instantiation files."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, 'r') as f:
            old = f.read()
    except (OSError, IOError):
        old = None
    if old == content:
        return False
    with open(path, 'w') as f:
        f.write(content)
    return True


def _emit_inst_file_multi(out_path, kernel_name, args_list, params_macro,
                           header_include, namespace_path):
    """Write one .cu file with MULTIPLE explicit instantiations (all the
    (Nlr, kWP, kUnroll) combos for a fixed (kFlat, kF8, kUnifiedView, kLogZ)).
    Grouping keeps the generated-file count under the ninja argv limit; ptxas
    still parallelizes across files (one per (kFlat, kF8, kUnifiedView, kLogZ)
    combo)."""
    buf = io.StringIO()
    buf.write('// GENERATED — do not edit. Multiple explicit instantiations per file\n')
    buf.write('// (grouped by (kFlat, kF8, kUnifiedView, kLogZ)) to keep the\n')
    buf.write('// generated-file count under the ninja argv limit. ptxas still\n')
    buf.write('// parallelizes across files.\n')
    buf.write(f'#include "{header_include}"\n\n')
    for ns in namespace_path:
        buf.write(f'namespace {ns} {{ ')
    buf.write('\n')
    for args in args_list:
        buf.write(f'template __global__ void {kernel_name}<{args}>(\n')
        buf.write(f'    {params_macro}\n')
        buf.write(');\n')
    for ns in reversed(namespace_path):
        buf.write(f'}} // namespace {ns}\n')
    _write_if_different(out_path, buf.getvalue())


def _emit_extern_decls(out_path, combos, header_include):
    """Write the extern-template decls header included by the dispatcher TU.

    `combos` is a list of (kernel_name, args_cxx, params_macro) triples.
    """
    buf = io.StringIO()
    buf.write('// GENERATED — do not edit. extern template declarations for\n')
    buf.write('// scattergather_kernel_p2p so the\n')
    buf.write('// dispatcher TU (sg.cu) does not instantiate;\n')
    buf.write('// symbols come from the per-combination .cu files.\n')
    buf.write('#pragma once\n')
    buf.write(f'#include "{header_include}"\n\n')
    buf.write('namespace pace {\n')
    buf.write('\n// scattergather_kernel_p2p\n')
    for (_kname, args, params) in combos:
        buf.write(f'extern template __global__ void scattergather_kernel_p2p<{args}>(\n')
        buf.write(f'    {params}\n')
        buf.write(');\n')
    buf.write('}  // namespace pace\n')
    _write_if_different(out_path, buf.getvalue())


# ---- Generator ------------------------------------------------------------

def gen_sg(repo_root, gen_dir):
    """Generate sg p2p instantiation files + decls.

    Granularity: one .cu file PER (kFlat, kF8, kUnifiedView, kLogZ) combo,
    containing all (Nlr, kWP, kUnroll) explicit instantiations for that combo.
    Grouping by (kFlat, kF8, kUnifiedView, kLogZ) keeps the generated-file
    count low (4 * 2 * 2 * 2 = 32 files) so ninja's argv stays well under the
    OS limit; ptxas still parallelizes across the 32 files. Each file carries
    3 * 3 = 9 instantiations.

    Returns (files, decls_path) — files is the list of generated .cu paths
    (also written to the manifest for setup.py).
    """
    kernel_dir = os.path.join(gen_dir, 'sg')
    decls_path = os.path.join(kernel_dir, 'sg_p2p_extern_decls.cuh')
    header = '../../../csrc/collective/sg/sg_kernels.cuh'

    files = []
    all_decls = []  # list of (kernel_name, args_cxx, params_macro)

    for kflat in SG_P2P_FLAT:
        for kf8 in SG_P2P_F8:
            for kuv in SG_P2P_UNIFIED_VIEW:
                for klz in SG_P2P_LOG_Z:
                    args_list = []
                    for (nlr, kwp) in SG_P2P_PAIRS:
                        knumthreads = _knumthreads(nlr, kwp)
                        for kunroll in SG_P2P_KUNROLL:
                            args_cxx = (
                                f'{nlr}, {kwp}, {knumthreads}, {kunroll}, '
                                f'{1 if kf8 else 0}, {kflat}, '
                                f'{1 if kuv else 0}, {1 if klz else 0}'
                            )
                            args_list.append(args_cxx)
                            all_decls.append(
                                ('scattergather_kernel_p2p', args_cxx,
                                 SG_P2P_PARAMS)
                            )
                    fname = (
                        f'sg_p2p_flat{kflat}_f8{1 if kf8 else 0}'
                        f'_uv{1 if kuv else 0}_lz{1 if klz else 0}.cu'
                    )
                    fpath = os.path.join(kernel_dir, fname)
                    _emit_inst_file_multi(
                        fpath, 'scattergather_kernel_p2p', args_list,
                        SG_P2P_PARAMS, header, ['pace']
                    )
                    files.append(fpath)

    _emit_extern_decls(decls_path, all_decls, header)

    return files, decls_path


def main():
    repo_root = os.path.dirname(os.path.abspath(os.path.dirname(
        os.path.abspath(__file__))))
    gen_dir = os.path.join(repo_root, 'build', 'gen')
    files, _decls = gen_sg(repo_root, gen_dir)

    # Emit a manifest setup.py can read (write-if-different to avoid bumping
    # mtimes on no-op reruns).
    manifest = os.path.join(gen_dir, 'sg_manifest.txt')
    manifest_content = ''.join(p + '\n' for p in files)
    _write_if_different(manifest, manifest_content)
    print(f'Generated {len(files)} sg p2p instantiation files '
          f'under {gen_dir}/sg/')


if __name__ == '__main__':
    main()
