#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

import os
import sys


def paste(sep, *args):
  return sep.join(args)


indents = 0
def emitln(f, lines):
  global indents
  for ln in ((lines,) if isinstance(lines, str) else lines):
    f.write('  '*indents + ln + '\n')


class Rec(object):
  def __init__(me, **kw):
    me.__dict__.update(kw)
  def __eq__(x, y):
    return x.__dict__ == y.__dict__
  def __hash__(me):
    h = 0
    for k in me.__dict__:
      h += hash((k, me.__dict__[k]))
    return h


reductions = ["Reduce", "AllReduce", "ReduceScatter"]
all_reds = ["sum", "prod", "minmax"]
all_tys = ["i8", "u8", "i32", "u32", "i64", "u64", "f16", "f32", "f64", "bf16", "f8e4m3", "f8e5m2"]

coll_to_lower = {
  "Broadcast": "broadcast",
  "Reduce": "reduce",
  "AllGather": "all_gather",
  "AllReduce": "all_reduce",
  "ReduceScatter": "reduce_scatter"
}

red_to_Func = {
  "sum": "FuncSum",
  "prod": "FuncProd",
  "minmax": "FuncMinMax"
}
red_to_ncclDevRedOp = {
  "sum": "ncclDevSum",
  "prod": "ncclDevProd",
  "minmax": "ncclDevMinMax"
}

ty_to_cxxtype = {
  "i8": "int8_t",
  "u8": "uint8_t",
  "i32": "int32_t",
  "u32": "uint32_t",
  "i64": "int64_t",
  "u64": "uint64_t",
  "f32": "float",
  "f64": "double",
  "f16": "half",
  "bf16": "__nv_bfloat16",
  "f8e4m3": "__nv_fp8_e4m3",
  "f8e5m2": "__nv_fp8_e5m2",
}
ty_to_ncclDataType = {
  "i8": "ncclInt8",
  "u8": "ncclUint8",
  "i32": "ncclInt32",
  "u32": "ncclUint32",
  "i64": "ncclInt64",
  "u64": "ncclUint64",
  "f32": "ncclFloat32",
  "f64": "ncclFloat64",
  "f16": "ncclFloat16",
  "bf16": "ncclBfloat16",
  "f8e4m3": "ncclFloat8e4m3",
  "f8e5m2": "ncclFloat8e5m2",
}


def enumerate_kernels():
  for algo in ["Ring_Simple", "Ring_LL", "Ring_LL128"]:
    yield kernel(coll="Broadcast", algo=algo)
    yield kernel(coll="AllGather", algo=algo)
  for red in all_reds:
    for ty in all_tys:
      for algo in ["Ring_Simple", "Ring_LL", "Ring_LL128"]:
        yield kernel(coll="Reduce", algo=algo, red=red, ty=ty)
        yield kernel(coll="AllReduce", algo=algo, red=red, ty=ty)
        yield kernel(coll="ReduceScatter", algo=algo, red=red, ty=ty)
      for algo in ["Tree_Simple", "Tree_LL", "Tree_LL128"]:
        yield kernel(coll="AllReduce", algo=algo, red=red, ty=ty)


def required_cuda(k):
  cudart = 0
  arch = k.arch
  if k.coll in reductions:
    if k.ty == "bf16":
      cudart = 11000
    if k.ty.startswith("f8"):
      cudart = 11080
      # FP8 reductions in reduce_kernel.h require SM90.
      arch = max(arch, 900)
  return cudart, arch


def kernel_fbase(k):
  return coll_to_lower[k.coll]


def kernel_fname(k):
  parts = [coll_to_lower[k.coll]]
  if k.coll in reductions:
    parts += [k.red, k.ty]
  return paste('_', *parts) + '.cu'


def profile_fname(fname):
  assert fname.endswith('.cu')
  return fname[:-len('.cu')] + '_profile.cu'


def kernel_cname(k):
  if k.coll in reductions:
    return paste("_", "ncclGenkDevKernel", k.coll, k.algo, k.red, k.ty)
  return paste("_", "ncclGenkDevKernel", k.coll, k.algo)


def prototype(k):
  cudart, _ = required_cuda(k)
  out = []
  for cname in (kernel_cname(k), kernel_cname(k) + "_profile"):
    form = (
      "#if CUDART_VERSION >= {cudart}\n"
      "  __global__ void {cname}(ncclGenkDevWorkArgs4K const);\n"
      "#else\n"
      "  constexpr void* {cname} = nullptr;\n"
      "#endif"
    )
    out.append(form.format(cname=cname, cudart=cudart))
  return "\n".join(out)


def kernel(**kw):
  return Rec(arch=900, **kw)


def instantiate(k, profile):
  cudart, arch = required_cuda(k)
  id = k.coll + '_' + k.algo
  cname = kernel_cname(k) + ('_profile' if profile else '')
  prof = 'true' if profile else 'false'
  if k.coll in reductions:
    targs = '<{prof}, {red}, {ty}>'.format(prof=prof, red=red_to_Func[k.red], ty=ty_to_cxxtype[k.ty])
  else:
    targs = '<{prof}>'.format(prof=prof)

  start = '    ncclSymkProfilerStart(&args4K.args);\n' if profile else ''
  stop = '    ncclSymkProfilerStop(&args4K.args);\n' if profile else ''
  form = (
    "#if CUDART_VERSION >= {cudart}\n"
    "  __global__ void {cname}(ncclGenkDevWorkArgs4K NCCL_GRID_CONSTANT const args4K) {{\n"
    "    #if CUDART_VERSION >= 12030 && __CUDA_ARCH__ >= 900\n"
    "      cudaGridDependencySynchronize();\n"
    "    #endif\n"
    "{start}"
    "    #if __CUDA_ARCH__ >= {arch}\n"
    "      ncclGenkRun_{id}{targs}(&args4K.args);\n"
    "    #endif\n"
    "{stop}"
    "  }}\n"
    "#endif"
  )
  return form.format(cudart=cudart, arch=arch, cname=cname, id=id, targs=targs,
                     start=start, stop=stop)


def partition(vals, keyfn):
  ans = {}
  for x in vals:
    k = keyfn(x)
    if k not in ans:
      ans[k] = []
    ans[k].append(x)
  return ans


def generate(gensrc):
  global indents
  if os.path.exists(gensrc):
    for name in os.listdir(gensrc):
      path = os.path.join(gensrc, name)
      if os.path.isfile(path):
        os.remove(path)
  else:
    os.mkdir(gensrc)

  kernels = list(enumerate_kernels())
  kernels_by_file = partition(kernels, lambda k: (kernel_fname(k), kernel_fbase(k)))

  # Generate dependency-only sources for collectives whose instantiations are
  # split by reduction operation and datatype.
  for fbase in set(kernel_fbase(k) for k in kernels):
    fname = fbase + '.cu'
    if (fname, fbase) not in kernels_by_file:
      kernels_by_file[fname, fbase] = []

  files_to_print = ""
  for (fname, fbase), ks in kernels_by_file.items():
    files_to_print += fname + ";"
    with open(os.path.join(gensrc, fname), "w") as f:
      emitln(f, '#include "sym_kernels.h"')
      emitln(f, '#include "general/kernel.cuh"')
      emitln(f, '#include "general/{fbase}.cuh"'.format(fbase=fbase))
      for k in ks:
        emitln(f, instantiate(k, profile=False))
    if ks:
      pfname = profile_fname(fname)
      files_to_print += pfname + ";"
      with open(os.path.join(gensrc, pfname), "w") as f:
        emitln(f, '#include "sym_kernels.h"')
        emitln(f, '#include "general/kernel.cuh"')
        emitln(f, '#include "general/{fbase}.cuh"'.format(fbase=fbase))
        for k in ks:
          emitln(f, instantiate(k, profile=True))

  with open(os.path.join(gensrc, "gen_kernels_host.cc"), "w") as f:
    emitln(f, '#include "sym_kernels.h"')
    emitln(f, '#include "device.h"')
    emitln(f, '#include "debug.h"')
    emitln(f, '')

    for k in kernels:
      emitln(f, prototype(k))
    emitln(f, '')

    emitln(f, 'extern int const ncclGenkKernelCount = %d;' % len(kernels))
    emitln(f, 'void* ncclGenkKernelList[] = {')
    for k in kernels:
      emitln(f, '(void*){cname},'.format(cname=kernel_cname(k)))
    emitln(f, 'nullptr};')
    emitln(f, '')

    emitln(f, 'void* ncclGenkKernelListProfile[] = {')
    for k in kernels:
      emitln(f, '(void*){cname}_profile,'.format(cname=kernel_cname(k)))
    emitln(f, 'nullptr};')
    emitln(f, '')

    emitln(f, 'int ncclGenkKernelRequirements[] = {')
    for index,k in enumerate(kernels):
      cudart, _ = required_cuda(k)
      emitln(f, '  %7d, /*%4d %s*/' % (cudart, index, kernel_cname(k)))
    emitln(f, '};')
    emitln(f, '')

    emitln(f, 'int ncclGenkKernelMaxDynamicSmem[%d];' % len(kernels))
    emitln(f, '')

    emitln(f, 'int ncclGenkGetKernelIndex(ncclSymkKernelId id, int red, ncclDataType_t ty) {')
    indents += 1
    emitln(f, 'switch (id) {')
    emitln(f, 'default: WARN("ncclGenkGetKernelIndex: unknown kernel id %d", (int)id); return -1;')
    for (coll, algo), coll_algo_ks in partition(kernels, lambda k: (k.coll, k.algo)).items():
      emitln(f, 'case ncclSymkKernelId_'+coll+'_'+algo+':')
      indents += 1
      if len(coll_algo_ks) == 1:
        emitln(f, 'return %d;' % kernels.index(coll_algo_ks[0]))
      else:
        emitln(f, 'switch ((ncclDevRedOp_t)red) {')
        emitln(f, 'default: WARN("ncclGenkGetKernelIndex: unknown red op %d for id %d", red, (int)id); return -1;')
        for red, coll_algo_red_ks in partition(coll_algo_ks, lambda k: k.red).items():
          emitln(f, 'case '+red_to_ncclDevRedOp[red]+':')
          indents += 1
          emitln(f, 'switch (ty) {')
          emitln(f, 'default: WARN("ncclGenkGetKernelIndex: unknown type %d for id %d red %d", '
                    '(int)ty, (int)id, red); return -1;')
          for k in coll_algo_red_ks:
            emitln(f, 'case %s: return %d;' % (ty_to_ncclDataType[k.ty], kernels.index(k)))
          emitln(f, '}')
          indents -= 1
        emitln(f, '}')
      indents -= 1
    emitln(f, '}')
    indents -= 1
    emitln(f, '}')

  files_to_print += "gen_kernels_host.cc;"
  if os.environ.get("NCCL_USE_CMAKE", "0") == "1":
    print(files_to_print)
  else:
    with open(os.path.join(gensrc, "rules.mk"), "w") as f:
      inst_names = sorted(set(kernel_fname(k) for k in kernels))
      names = inst_names + [profile_fname(n) for n in inst_names] + ["gen_kernels_host.cc"]
      f.write("LIB_OBJS_GENK_GEN = $(patsubst %,$(OBJDIR)/genobj/general/%.o,{names})\n"
              .format(names=" ".join(names)))
      f.write("\n")

      inst_names = sorted(set((kernel_fname(k), kernel_fbase(k)) for k in kernels))
      for fname, fbase in inst_names:
        for src in (fname, profile_fname(fname)):
          f.write(
            "$(OBJDIR)/genobj/general/{src}.o: $(OBJDIR)/gensrc/general "
            "$(OBJDIR)/genobj/general/{fbase}.cu.d\n"
            "\t$(call COMPILE_GENK,$@,$(OBJDIR)/gensrc/general/{src},$(NVCC_GENCODE))\n"
            "\n"
            .format(src=src, fbase=fbase)
          )


if __name__ == "__main__":
  generate(sys.argv[1])
