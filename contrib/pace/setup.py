import os
import subprocess
import setuptools
import importlib

from pathlib import Path
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import ctypes


def get_nccl_version(so_path):
    """
    Get the NCCL library version

    Args:
        so_path: path to the NCCL shared library, e.g. '/usr/lib/x86_64-linux-gnu/libnccl.so.2'

    Returns:
        tuple: (major_version, minor_version, patch_version)
    """
    try:
        # Load the NCCL library
        nccl_lib = ctypes.CDLL(so_path)

        # Define the function prototype
        nccl_lib.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        nccl_lib.ncclGetVersion.restype = ctypes.c_int

        # Prepare the argument
        version_ptr = ctypes.c_int()

        # Call the function
        result = nccl_lib.ncclGetVersion(ctypes.byref(version_ptr))

        if result != 0:
            print(f"Error calling ncclGetVersion, return code: {result}")
            return None

        # Parse the version number
        version = version_ptr.value
        major = version // 10000
        minor = (version - major * 10000) // 100
        patch = version % 100
        return major, minor, patch

    except Exception as e:
        print(f"Error loading NCCL library: {e}")
        return None


def get_extension_gin_cpp():

    nccl_dir = os.environ.get('NCCL_DIR', None)
    nccl_header_dir = None
    nccl_so_path = None
    nccl_a_path = None
    if nccl_dir is None:
        nccl_so_path = subprocess.check_output("ldconfig -p | grep libnccl.so", shell=True, text=True).splitlines()
        if len(nccl_so_path) == 0:
            raise Exception('libnccl.so is not found in default library path')
        nccl_so_path = nccl_so_path[0].split()[3]
        if not os.path.exists(nccl_so_path):
            raise Exception(f'ldconfig tells libnccl.so is on {nccl_so_path}, but it does not exist')
    else:
        nccl_header_dir = os.path.join(nccl_dir, 'include')
        nccl_a_path = os.path.join(nccl_dir, 'lib/libnccl_static.a')
        nccl_so_path = os.path.join(nccl_dir, 'lib/libnccl.so')
        assert os.path.exists(nccl_header_dir), f'{nccl_header_dir} not found'
        assert os.path.exists(nccl_a_path), f'{nccl_a_path} not found'
    assert os.path.exists(nccl_so_path)

    major, minor, patch = get_nccl_version(nccl_so_path)
    if major * 100 + minor < 230:
        raise Exception(f'requested nccl library (at {nccl_so_path}) version is '
                        f'{major}.{minor}.{patch}, while pace requires >= 2.30')

    cxx_flags = ['-O3', '-Wno-deprecated-declarations', '-Wno-unused-variable',
                 '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes']
    nvcc_flags = ['-O3', '-Xcompiler', '-O3']
    if os.environ.get('PACE_KERNEL_DEBUG', None) in ['1', 'true', 'True']:  # default off
        nvcc_flags.extend(['-lineinfo', '-Xptxas=-warn-spills'])
    if os.environ.get('PACE_TIMEOUT_DEBUG', '1') not in ['0', 'false', 'False']:  # default on
        nvcc_flags.extend(['-DPACE_TIMEOUT_DEBUG'])
        cxx_flags.extend(['-DPACE_TIMEOUT_DEBUG'])
    if os.environ.get('PACE_FAST_DEBUG', None) in ['1', 'true', 'True']:  # default off
        nvcc_flags.extend(['-DPACE_FAST_DEBUG'])

    sources = [
        'csrc/bindings.cpp',
        'csrc/services/event.cpp',
        'csrc/runtime/commoncomm.cpp',
        'csrc/services/nccl_comm_cache.cpp',
        'csrc/collective/sg/sgcomm.cpp',
        'csrc/collective/rs/rscomm.cpp',
        'csrc/collective/ag/agcomm.cpp',
        'csrc/device/sync.cu',
        'csrc/collective/rs/rs.cu',
        'csrc/collective/ag/ag_zero_sm.cu',
        'csrc/collective/ag/ag_ring.cu',
        'csrc/collective/sg/sg.cu',
    ]
    include_dirs = [os.path.abspath('csrc')]  # absolute so root-relative includes resolve regardless of ninja cwd

    # ---- Per-instantiation split for sg's scattergather_kernel_p2p ----
    # sg.cu's __global__ kernel template scattergather_kernel_p2p is
    # referenced by the host dispatcher (LAUNCH_SG_P2P_KERNEL + SWITCH_P2P_*
    # macros) for 288 (Nlr, kWP, kUnroll, kF8, kFlat, kUnifiedView, kLogZ)
    # combinations. Compiling all in one TU is slow (ptxas register allocation
    # dominates). scripts/gen_sg_inst.py emits one .cu file per
    # (kFlat, kF8, kUnifiedView, kLogZ) combo (32 files, 9 inst each) so
    # ninja compiles them in parallel; the dispatcher TU uses extern template
    # (generated sg_p2p_extern_decls.cuh) so it doesn't instantiate.
    repo_root = os.path.dirname(os.path.abspath(__file__))
    gen_dir = os.path.join(repo_root, 'build', 'gen')
    import subprocess as _sp
    _gen_script = os.path.join(repo_root, 'scripts', 'gen_sg_inst.py')
    if os.path.exists(_gen_script):
        _sp.check_call(['python3', _gen_script], cwd=repo_root)
    else:
        print(f'  WARNING: gen script {_gen_script} not found; skipping sg split')
    _manifest = os.path.join(gen_dir, 'sg_manifest.txt')
    if os.path.exists(_manifest):
        with open(_manifest) as f:
            for line in f:
                line = line.strip()
                if line:
                    sources.append(line)
    _sg_gen_dir = os.path.join(gen_dir, 'sg')
    if os.path.isdir(_sg_gen_dir):
        include_dirs.append(_sg_gen_dir)

    # ---- Per-instantiation split for rs's reduce_scatter ----
    # rs.cu's __global__ kernel template reduce_scatter is referenced by the
    # host dispatcher (SWITCH_TYPE + SWITCH_MUL + SWITCH_OUT_MODE +
    # SWITCH_ALIGN) for 32 (T, kMul, kOutMode, kAligned) combinations.
    # Compiling all in one TU is slow. scripts/gen_rs_inst.py emits one .cu
    # file per kOutMode (4 files, 8 inst each) so ninja compiles them in
    # parallel; the dispatcher TU uses extern template (generated
    # rs_extern_decls.cuh) so it doesn't instantiate.
    _gen_script = os.path.join(repo_root, 'scripts', 'gen_rs_inst.py')
    if os.path.exists(_gen_script):
        _sp.check_call(['python3', _gen_script], cwd=repo_root)
    else:
        print(f'  WARNING: gen script {_gen_script} not found; skipping rs split')
    _manifest = os.path.join(gen_dir, 'rs_manifest.txt')
    if os.path.exists(_manifest):
        with open(_manifest) as f:
            for line in f:
                line = line.strip()
                if line:
                    sources.append(line)
    _rs_gen_dir = os.path.join(gen_dir, 'rs')
    if os.path.isdir(_rs_gen_dir):
        include_dirs.append(_rs_gen_dir)

    library_dirs = []
    nvcc_dlink = []
    extra_link_args = []

    if nccl_header_dir is not None:
        include_dirs.extend([f'{nccl_header_dir}'])
    nvcc_dlink.extend(['-dlink'])
    # pace_cpp calls the CUDA driver API (cuGraphLaunch / cuStreamBatchMemOp /
    # cuMem*), so libcuda must be a hard runtime dependency. On driver-less
    # build hosts (e.g. a devbox with a 0-byte libcuda.so.1 stub), the linker's
    # default --as-needed drops -lcuda from DT_NEEDED, and the built .so then
    # fails to load on real GPUs with "undefined symbol: cuGraphLaunch". Force
    # it back on (harmless on hosts with a real driver).
    extra_link_args.extend(([] if nccl_a_path is None else [f'{nccl_a_path}']) + ['-Wl,--no-as-needed', '-lcuda', '-Wl,--as-needed', '-lnuma'])

    # CUDA 12 flags
    nvcc_flags.extend(['-rdc=true', '--ptxas-options=--register-usage-level=10'])

    # Ensure device linking and CUDA device runtime when RDC is enabled
    if '-rdc=true' in nvcc_flags and '-dlink' not in nvcc_dlink:
        nvcc_dlink.append('-dlink')

    # Keep the statically-embedded NCCL copies local to this module so calls
    # inside pace_cpp resolve to the GIN-capable NCCL it was built against,
    # rather than being preempted by a (possibly older) libnccl.so.2 that
    # PyTorch already loaded into the process.
    if nccl_a_path is not None:
        extra_link_args.extend(['-Wl,--exclude-libs,ALL'])

    # Put them together
    extra_compile_args = {
        'cxx': cxx_flags,
        'nvcc': nvcc_flags,
    }
    if len(nvcc_dlink) > 0:
        extra_compile_args['nvcc_dlink'] = nvcc_dlink

    # Summary
    print(f'Build summary:')
    print(f' > Sources: {sources}')
    print(f' > Includes: {include_dirs}')
    print(f' > Libraries: {library_dirs}')
    print(f' > Compilation flags: {extra_compile_args}')
    print(f' > Link flags: {extra_link_args}')
    print(f' > NCCL dir: {nccl_dir}')

    extension_gin_cpp = CUDAExtension(
        name='pace_cpp',
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        sources=sources,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
    )

    return extension_gin_cpp


if __name__ == '__main__':
    setuptools.setup(
        name='pace',
        version='0.1.0',
        packages=setuptools.find_packages(
            include=['pace', 'pace.sg', 'pace.rs', 'pace.ag']
        ),
        ext_modules=[
            get_extension_gin_cpp(),
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
