#!/usr/bin/env python3
"""Generate per-instantiation .cu files for rs's reduce_scatter template so
ninja compiles them in parallel.

Background: rs.cu's __global__ kernel template reduce_scatter is referenced
by the host dispatcher (SWITCH_TYPE + SWITCH_MUL + SWITCH_OUT_MODE +
SWITCH_ALIGN) for every (T, kMul, kOutMode, kAligned) combination:

  2 T (float, __nv_bfloat16) × 2 kMul × 4 kOutMode × 2 kAligned
  = 32 instantiations, grouped into 4 .cu files (one per kOutMode value).

Compiling all of them in one TU (rs.cu) is slow — ptxas register allocation
dominates. This script emits one .cu file per kOutMode value, each holding
the 2 * 2 * 2 = 8 (T, kMul, kAligned) explicit instantiations, plus a decls
header with `extern template` declarations for the dispatcher.

Sibling to scripts/gen_ep_inst.py (same pattern, different kernel family).
Standalone — does not import gen_ep_inst.py, so the two generators can be
run/maintained independently.

Generated files (under <repo>/build/gen/rs/):
  rs_outmode{N}.cu    — 4 files (N ∈ {0,1,2,3}); each holds 8 instantiations.
                          0=DIRECT, 1=AVG, 2=CAST_BF16, 3=AVG_CAST_BF16
                          (matches RS_OUT_MODE_* enum in rs_defs.cuh).
  rs_extern_decls.cuh — all `extern template` declarations.

Total: 4 .cu files. Grouped-by-kOutMode granularity keeps the file count
under the ninja argv limit; ptxas still parallelizes across the 4 files.

Manifest (under <repo>/build/gen/):
  rs_manifest.txt     — one path per generated .cu, for setup.py.
"""
import os
import sys
import io


# ---- Template-arg enumerations --------------------------------------------

# SWITCH_TYPE (rs_defs.cuh) enumerates these T values.
RS_TYPES = [
    ('float', 'float'),
    ('__nv_bfloat16', '__nv_bfloat16'),
]

# SWITCH_MUL (rs_defs.cuh) enumerates these kMul values (false/true → 0/1).
RS_MUL = [False, True]

# SWITCH_OUT_MODE (rs_defs.cuh) enumerates these kOutMode values.
# 0=RS_OUT_MODE_DIRECT, 1=RS_OUT_MODE_AVG, 2=RS_OUT_MODE_CAST_BF16,
# 3=RS_OUT_MODE_AVG_CAST_BF16 (rs_defs.cuh:11-14).
RS_OUT_MODES = [
    (0, 'DIRECT'),
    (1, 'AVG'),
    (2, 'CAST_BF16'),
    (3, 'AVG_CAST_BF16'),
]

# SWITCH_ALIGN (rs_defs.cuh) enumerates these kAligned values (false/true).
RS_ALIGNED = [False, True]


# ---- Kernel param list (verbatim from rs_kernels.cuh, comments stripped) ----

# reduce_scatter<T, kMul, kOutMode, kAligned> signature.
RS_PARAMS = (
    'const float extra_mul, const float extra_post_mul, '
    'uint64_t *args, size_t nargs, void* out_ptr, '
    'ncclWindow_t gin_win, void *gin_win_ptr, '
    'int *gmem_barrier, const int nvl_ring, const int rdma_ring, '
    'const size_t rdma_unroll, int round_n, const bool use_wg, '
    'int rank, int num_local_ranks, int num_ranks, ncclDevComm devComm, '
    'int arrival_sig_base, int cuda_device_id'
)


# ---- Helpers (mirrors gen_ep_inst.py — kept standalone to avoid cross-file coupling) ----

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
    (T, kMul, kAligned) combos for a fixed kOutMode)."""
    buf = io.StringIO()
    buf.write('// GENERATED — do not edit. Multiple explicit instantiations per file\n')
    buf.write('// (grouped by kOutMode) to keep the generated-file count\n')
    buf.write('// under the ninja argv limit. ptxas still parallelizes across files.\n')
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


def _emit_extern_decls(out_path, combos, header_include, namespace_path):
    """Write the extern-template decls header included by the dispatcher TU."""
    buf = io.StringIO()
    buf.write('// GENERATED — do not edit. extern template declarations for\n')
    buf.write('// reduce_scatter so the\n')
    buf.write('// dispatcher TU (rs.cu) does not instantiate;\n')
    buf.write('// symbols come from the per-combination .cu files.\n')
    buf.write('#pragma once\n')
    buf.write(f'#include "{header_include}"\n\n')
    for ns in namespace_path:
        buf.write(f'namespace {ns} {{\n')
    buf.write('\n// reduce_scatter\n')
    for (_kname, args, params) in combos:
        buf.write(f'extern template __global__ void reduce_scatter<{args}>(\n')
        buf.write(f'    {params}\n')
        buf.write(');\n')
    for ns in reversed(namespace_path):
        buf.write(f'}} // namespace {ns}\n')
    _write_if_different(out_path, buf.getvalue())


# ---- Generator ------------------------------------------------------------

def gen_rs(repo_root, gen_dir):
    """Generate rs instantiation files + decls.

    Granularity: one .cu file PER kOutMode value, containing all (T, kMul,
    kAligned) explicit instantiations for that mode. Grouping by kOutMode
    keeps the generated-file count low (4 files) so ninja's argv stays well
    under the OS limit; ptxas still parallelizes across the 4 files. Each
    file carries 2 * 2 * 2 = 8 instantiations.

    Returns (files, decls_path) — files is the list of generated .cu paths
    (also written to the manifest for setup.py).
    """
    kernel_dir = os.path.join(gen_dir, 'rs')
    decls_path = os.path.join(kernel_dir, 'rs_extern_decls.cuh')
    header = '../../../csrc/collective/rs/rs_kernels.cuh'

    files = []
    all_decls = []  # list of (kernel_name, args_cxx, params_macro)

    for (out_mode, _name) in RS_OUT_MODES:
        args_list = []
        for (_t_name, t_cxx) in RS_TYPES:
            for k_mul in RS_MUL:
                for k_aligned in RS_ALIGNED:
                    args_cxx = (
                        f'{t_cxx}, {1 if k_mul else 0}, {out_mode}, '
                        f'{1 if k_aligned else 0}'
                    )
                    args_list.append(args_cxx)
                    all_decls.append(
                        ('reduce_scatter', args_cxx, RS_PARAMS)
                    )
        fname = f'rs_outmode{out_mode}.cu'
        fpath = os.path.join(kernel_dir, fname)
        _emit_inst_file_multi(
            fpath, 'reduce_scatter', args_list,
            RS_PARAMS, header, ['pace']
        )
        files.append(fpath)

    _emit_extern_decls(decls_path, all_decls, header, ['pace'])

    return files, decls_path


def main():
    repo_root = os.path.dirname(os.path.abspath(os.path.dirname(
        os.path.abspath(__file__))))
    gen_dir = os.path.join(repo_root, 'build', 'gen')
    files, _decls = gen_rs(repo_root, gen_dir)

    # Emit a manifest setup.py can read (write-if-different to avoid bumping
    # mtimes on no-op reruns).
    manifest = os.path.join(gen_dir, 'rs_manifest.txt')
    manifest_content = ''.join(p + '\n' for p in files)
    _write_if_different(manifest, manifest_content)
    print(f'Generated {len(files)} rs instantiation files '
          f'under {gen_dir}/rs/')


if __name__ == '__main__':
    main()
