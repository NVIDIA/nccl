#!/usr/bin/env python3
import os
import sys
import shutil

################################################################################
# The first command line argument is the path to the directory to generate and
# populate.

gensrc = sys.argv[1]

if os.path.exists(gensrc):
  for name in os.listdir(gensrc):
    path = os.path.join(gensrc, name)
    if os.path.isfile(path):
      os.remove(path)
    elif os.path.isdir(path):
      shutil.rmtree(path)
else:
  os.mkdir(gensrc)

def paste(sep, *args):
  return sep.join(args)

indents = 0
def emitln(f, lines):
  global indents
  for ln in ((lines,) if isinstance(lines, str) else lines):
    f.write('  '*indents + ln + '\n')

def indent(s):
  endl = '\n' if s.endswith('\n') else ''
  return '\n'.join('  '+l for l in s.splitlines()) + endl

class Rec(object):
  def __init__(me, **kw):
    me.__dict__.update(kw)
  def __eq__(x, y):
    if len(x) != len(y): return False
    for k in x:
      if k not in y: return False
      if x[k] != y[k]: return False
    return True
  def __hash__(me):
    h = 0
    for k in me.__dict__:
      h += hash((k, me.__dict__[k]))
    return h

################################################################################
# Edit this region for introducing new algos etc

reductions = ["AllReduce","ReduceScatter"]
all_reds = ["sum"]
all_tys = ["f32","f16","bf16","f8e4m3","f8e5m2"]
gin_algos = ["GinHier_MCRing"]

nvls_algos_by_coll = {
  "AllReduce": ["AGxLLMC_R","RSxLDMC_AGxSTMC"],
  "ReduceScatter": ["LDMC"]
}
ldmc_algos = ["RSxLDMC_AGxSTMC", "LDMC"]

coll_to_lower = {
  "AllGather": "all_gather",
  "AllReduce": "all_reduce",
  "ReduceScatter": "reduce_scatter"
}

red_to_ncclDevRedOp = {
  "sum": "ncclDevSum"
}
red_to_Func = {
  "sum": "FuncSum"
}

ty_to_ncclDataType = {
  "f32": "ncclFloat32",
  "f16": "ncclFloat16",
  "bf16": "ncclBfloat16",
  "f8e4m3": "ncclFloat8e4m3",
  "f8e5m2": "ncclFloat8e5m2"
}
ty_to_cxxtype = {
  "f32": "float",
  "f16": "half",
  "bf16": "__nv_bfloat16",
  "f8e4m3": "__nv_fp8_e4m3",
  "f8e5m2": "__nv_fp8_e5m2"
}

def enumerate_kernels():
  for algo in ["LL","LLMC","ST","STMC","GinHier_MCRing"]:
    yield Rec(coll="AllGather", algo=algo)
  for red in all_reds:
    for ty in all_tys:
      for algo in ["AGxLL_R","AGxLLMC_R","RSxLD_AGxST","RSxLDMC_AGxSTMC"]:
        yield Rec(coll="AllReduce", algo=algo, red=red, ty=ty)
      for algo in ["LL","LD","LDMC"]:
        yield Rec(coll="ReduceScatter", algo=algo, red=red, ty=ty)

def required_cuda(k):
  cudart, arch, specific_sms  = 0, 600, None
  is_nvls = k.algo in nvls_algos_by_coll.get(k.coll, [])
  if is_nvls:
    cudart = max(cudart, 12010)
    arch = 900
  if k.coll in reductions:
    if k.ty == "bf16":
      cudart = max(cudart, 11000)
    if k.ty.startswith("f8"):
      cudart = max(cudart, 11080)
      arch = 900
      if k.algo in ldmc_algos:
        cudart = 12070
        arch = None
        specific_sms = ["100a", "101a", "100f", "101f", "120a", "121a"]
  return (cudart, arch, specific_sms)

################################################################################

def kernel_fbase(k):
  return coll_to_lower[k.coll] + ("_gin" if k.algo in gin_algos else "")

def kernel_fname(k):
  parts = [coll_to_lower[k.coll]]
  if k.algo in gin_algos: parts += ['gin']
  if k.coll in reductions:
    if k.algo in ldmc_algos and k.ty.startswith('f8'):
      parts += [k.red, k.ty, k.algo]
    else:
      parts += [k.red, k.ty]
  return paste('_', *parts) + '.cu'

def kernel_gencode(k):
  if k.coll in reductions and k.algo in ldmc_algos and k.ty.startswith('f8'):
    return "$(NVCC_GENCODE_LDMC_FP8)"
  else:
    return "$(NVCC_GENCODE)"

def kernel_cname(k):
  if k.coll in reductions:
    return paste("_", "ncclSymkDevKernel", k.coll, k.algo, k.red, k.ty)
  else:
    return paste("_", "ncclSymkDevKernel", k.coll, k.algo)

def kernel_conds(k):
  cudart, arch, specific_sms = required_cuda(k)
  if cudart == 0 and arch == 0: return (None, None)

  cudart_cond = "CUDART_VERSION >= %d"%cudart
  if not specific_sms:
    arch_cond = "__CUDA_ARCH__ >= %d"%arch
  else:
    arch_cond = " || ".join(["0"] + ["NCCL_CUDA_ARCH_%sSPECIFIC==%d"%("FAMILY_" if sm[-1] == "f" else "", 10*int(sm.replace('a', '').replace('f', ''))) for sm in specific_sms])
  return cudart_cond, arch_cond

def instantiate(k):
  cudart_cond, arch_cond = kernel_conds(k)
  if (cudart_cond, arch_cond) == (None, None):
    form_red_ty = (
      "__global__ void {cname}(ncclSymkDevWorkArgs4K NCCL_GRID_CONSTANT const args4K) {{\n"
      "  ncclSymkRun_{id}<{red}, {ty}>(&args4K.args);\n"
      "}}"
    )
    form = (
      "__global__ void {cname}(ncclSymkDevWorkArgs4K NCCL_GRID_CONSTANT const args4K) {{\n"
      "  ncclSymkRun_{id}(&args4K.args);\n"
      "}}"
    )
  else:
    form_red_ty = (
      "#if {cudart_cond}\n"
      "  __global__ void {cname}(ncclSymkDevWorkArgs4K NCCL_GRID_CONSTANT const args4K) {{\n"
      "    #if {arch_cond}\n"
      "      ncclSymkRun_{id}<{red}, {ty}>(&args4K.args);\n"
      "    #endif\n"
      "  }}\n"
      "#endif"
    )
    form = (
      "#if {cudart_cond}\n"
      "  __global__ void {cname}(ncclSymkDevWorkArgs4K NCCL_GRID_CONSTANT const args4K) {{\n"
      "    #if {arch_cond}\n"
      "      ncclSymkRun_{id}(&args4K.args);\n"
      "    #endif\n"
      "  }}\n"
      "#endif"
    )

  id = k.coll+'_'+k.algo
  cname = kernel_cname(k)
  if k.coll in reductions:
    inst = form_red_ty.format(cname=cname, id=id, red=red_to_Func[k.red], ty=ty_to_cxxtype[k.ty], cudart_cond=cudart_cond, arch_cond=arch_cond)
  else:
    inst = form.format(cname=cname, id=id, cudart_cond=cudart_cond, arch_cond=arch_cond)
  return inst

def prototype(k):
  cudart_cond, arch_cond = kernel_conds(k)
  if cudart_cond is None:
    form = "__global__ void {cname}(ncclSymkDevWorkArgs4K const);"
  else:
    form = (
      "#if {cudart_cond}\n"
      "  __global__ void {cname}(ncclSymkDevWorkArgs4K const);\n"
      "#else\n"
      "  constexpr void* {cname} = nullptr;\n"
      "#endif"
    )
  return form.format(cname=kernel_cname(k), cudart_cond=cudart_cond)

################################################################################

def partition(vals, keyfn):
  ans = {}
  for x in vals:
    k = keyfn(x)
    if k not in ans:
      ans[k] = []
    ans[k].append(x)
  return ans


kernels_by_file = partition(enumerate_kernels(), lambda k: (kernel_fname(k), kernel_fbase(k)))

# Add dependency only files (e.g. allreduce.cu)
for fbase in set(kernel_fbase(k) for k in enumerate_kernels()):
  fname = fbase + '.cu'
  if (fname, fbase) not in kernels_by_file:
    kernels_by_file[fname, fbase] = []

files_to_print = ""
# Generate each kernel instantiation file
for (fname, fbase), ks in kernels_by_file.items():
  files_to_print += fname + ";"
  with open(os.path.join(gensrc, fname), "w") as f:
    emitln(f, '#include "sym_kernels.h"')
    emitln(f, '#include "symmetric/kernel.cuh"')
    emitln(f, '#include "symmetric/{fbase}.cuh"'.format(fbase=fbase))
    for k in ks:
      emitln(f, instantiate(k))

# Generate <gensrc>/sym_kernels_host.cc
with open(os.path.join(gensrc, "sym_kernels_host.cc"), "w") as f:
  emitln(f, '#include "sym_kernels.h"')
  emitln(f, '#include "device.h"')
  emitln(f, '')

  for k in enumerate_kernels():
    emitln(f, prototype(k))
  emitln(f, '')

  emitln(f, 'extern int const ncclSymkKernelCount = %d;' % len(list(enumerate_kernels())))
  emitln(f, 'void* ncclSymkKernelList[] = {')
  for k in enumerate_kernels():
    emitln(f, '(void*){cname},'.format(cname=kernel_cname(k)))
  emitln(f, 'nullptr};')
  emitln(f, '')

  emitln(f, 'int ncclSymkKernelRequirements[] = {')
  for index,k in enumerate(enumerate_kernels()):
    cudart, _, _ = required_cuda(k)
    sym = kernel_cname(k)
    emitln(f, '  %7d, /*%4d %s*/' % (cudart or 0, index, sym));
  emitln(f, '};')
  emitln(f, '')

  emitln(f, 'void* ncclSymkGetKernelPtr(ncclSymkKernelId id, int red, ncclDataType_t ty) {')
  indents += 1
  emitln(f, 'switch (id) {')
  emitln(f, 'default: return nullptr;')
  for (coll, algo), coll_algo_ks in partition(enumerate_kernels(), lambda k: (k.coll, k.algo)).items():
    emitln(f, 'case ncclSymkKernelId_'+coll+'_'+algo+':')
    indents += 1
    if len(coll_algo_ks) == 1:
      emitln(f, 'return (void*)&'+kernel_cname(coll_algo_ks[0])+';')
    else:
      emitln(f, 'switch ((ncclDevRedOp_t)red) {')
      emitln(f, 'default: return nullptr;')
      for red, coll_algo_red_ks in partition(coll_algo_ks, lambda k: k.red).items():
        emitln(f, 'case '+red_to_ncclDevRedOp[red]+':')
        indents += 1
        emitln(f, 'switch (ty) {')
        emitln(f, 'default: return nullptr;')
        for k in coll_algo_red_ks:
          emitln(f, 'case '+ty_to_ncclDataType[k.ty]+': return (void*)'+kernel_cname(k)+';')
        emitln(f, '}')
        indents -= 1
      emitln(f, '}')
    indents -=1
  emitln(f, '}')
  indents -= 1
  emitln(f, '}')

# Output file list for CMake (excludes rules.mk since it's not generated for CMake)
files_to_print += "sym_kernels_host.cc;"
if os.environ.get("NCCL_USE_CMAKE", "0") == "1":
  print(files_to_print)

# Generate <gensrc>/rules.mk (only needed for Makefile builds, not CMake)
if os.environ.get("NCCL_USE_CMAKE", "0") != "1":
  with open(os.path.join(gensrc, "rules.mk"), "w") as f:
    inst_names = sorted(set(kernel_fname(k) for k in enumerate_kernels()))
    names = inst_names + ["sym_kernels_host.cc"]
    f.write("LIB_OBJS_SYM_GEN = $(patsubst %,$(OBJDIR)/genobj/symmetric/%.o,{names})\n"
            .format(names=" ".join(names)))
    f.write("\n")

    inst_names = sorted(set((kernel_fname(k), kernel_fbase(k), kernel_gencode(k)) for k in enumerate_kernels()))
    for fname, fbase, gencode in inst_names:
      f.write(
        "$(OBJDIR)/genobj/symmetric/{fname}.o: $(OBJDIR)/gensrc/symmetric $(OBJDIR)/genobj/symmetric/{fbase}.cu.d\n"
        "\t" "$(call COMPILE_SYM,$@,$(OBJDIR)/gensrc/symmetric/{fname},{gencode})\n"
        "\n"
        .format(fname=fname, fbase=fbase, gencode=gencode)
      )

