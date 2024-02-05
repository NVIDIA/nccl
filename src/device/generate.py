#!/usr/bin/env python3
import os
import sys

# Order of redops, tys, protos, algos must match src/include/device.h
all_colls =  ["Broadcast","Reduce","AllGather","ReduceScatter","AllReduce","SendRecv"]
all_redops = ["Sum","Prod","MinMax","PreMulSum","SumPostDiv"]
all_tys =    ["i8","u8","i32","u32","i64","u64","f16","f32","f64","bf16"]
all_protos = ["LL","LL128","SIMPLE"]
all_algos =  ["TREE","RING","COLLNET_DIRECT","COLLNET_CHAIN","NVLS","NVLS_TREE"]

################################################################################
# The first command line argument is the path to the directory to generate and
# populate.

gensrc = sys.argv[1]

if os.path.exists(gensrc):
  for name in os.listdir(gensrc):
    os.remove(os.path.join(gensrc, name))
    #os.truncate(os.path.join(gensrc, name), 0)
else:
  os.mkdir(gensrc)

################################################################################
# The second  command line argument is used as a regex to filter the functions
# which make it into libnccl. This is helpful for reducing the binary when
# developing device code. The regex supports non-space containing globs '*',
# parentheses '(x)', and union 'a|b'. The string representing the function has
# one of the forms:
#
# SendRecv
# (AllGather|Broadcast) <algo> <proto>
# (AlLReduce|Reduce|ReduceScatter) <redop> <type> <algo> <proto>
#
# The possible values for redop, type, algo, proto can be found in the all_<foo>
# lists at the top of this file.
#
# Since the Makefile forwards this from the ONLY_FUNCS variable, useful command
# line examples are given:
"""
# Only send/recv:
make ONLY_FUNCS="SendRecv"

# Only non-reductions:
make ONLY_FUNCS="AllGather * *|Broadcast * *|SendRecv"

# Only AllReduce sum f32 (but all algos, protos)
make ONLY_FUNCS="AllReduce Sum f32 * *"

# Only AllReduce minmax i32 NVLS (but all protos)
make ONLY_FUNCS="AllReduce MinMax i32 NVLS *"

# AllReduce sum <all floats> RING LL128
make ONLY_FUNCS="AllReduce Sum f32 RING LL128"
"""

# Paste all non-None arguments together with `sep`.
def paste(sep, *args):
  return sep.join(x for x in args if x is not None)

func_pattern = sys.argv[2:3]
if func_pattern and func_pattern[0]:
  import re
  func_pattern = func_pattern[0]
  func_pattern = func_pattern.replace("*", "[^ ]*")
  func_pattern += "$"
  def func_filter(*fn):
    return None is not re.match(func_pattern, paste(" ", *fn), flags=re.IGNORECASE)
else:
  def func_filter(coll, redop, ty, algo, proto):
    return True

################################################################################

algos_of_coll = {
  "AllGather":     ["RING","COLLNET_DIRECT","NVLS"],
  "AllReduce":     all_algos,
  "Broadcast":     ["RING"],
  "Reduce":        ["RING"],
  "ReduceScatter": ["RING","COLLNET_DIRECT","NVLS"],
  "SendRecv":      [None]
}

coll_camel_to_lower = {
  "AllGather":     "all_gather",
  "AllReduce":     "all_reduce",
  "Broadcast":     "broadcast",
  "Reduce":        "reduce",
  "ReduceScatter": "reduce_scatter",
  "SendRecv":      "sendrecv"
}
coll_lower_to_camel = {coll_camel_to_lower[x]: x for x in coll_camel_to_lower}

################################################################################

# Returns pair of minimum required values for (CUDART_VERSION, __CUDA_ARCH__)
# or None if function is never supported. Note that (0, 0) encodes universal
# support.
def required_cuda(coll, redop, ty, algo, proto):
  cudart, arch = 0, 0
  # kernels mapped to by coll="Nop" functions have coll="Generic"
  if coll in ("SendRecv", "Generic", "Nop"): return (cudart, arch)

  if proto!="SIMPLE" and algo not in ("RING","TREE"): return None

  if coll in ("AllReduce","Reduce","ReduceScatter"):
    if redop=="SumPostDiv" and ty[0] not in ("i","u"): return None
    if ty=="bf16": cudart = max(cudart, 11000)

  if "NVLS" in algo:
    if coll in ("AllReduce","Reduce","ReduceScatter"):
      # Must match ncclNvlsSupported() in src/include/device.h
      nvls_ok = ((ty in ("i32","u32","i64","u64") and redop in ("Sum","MinMax")) or
                 (ty in ("f32","f64") and redop=="Sum") or
                 (ty in ("f16","bf16") and redop in ("Sum","MinMax")))
      if not nvls_ok: return None
    cudart = max(cudart, 12010)
    arch = max(arch, 900)

  return (cudart, arch)

# Maps functions to the chosen representative for the equivalence class it
# belongs to. For instance (sum, signed int) maps to (sum, unsigned int).
def equivalent_primary(coll, redop, ty, algo, proto):
  if coll in ("AllReduce", "Reduce", "ReduceScatter"):
    # map signed integer sum/prod to unsigned
    if redop in ("Sum","Prod","PreMulSum") and ty[0]=="i":
      return (coll, redop, "u"+ty[1:], algo, proto)
    # map signed integer min/max to unsigned for non-NVLS
    if redop=="MinMax" and ty[0]=="i" and ("NVLS" not in algo):
      return (coll, redop, "u"+ty[1:], algo, proto)
  return (coll, redop, ty, algo, proto)

# Map to another func representing the best kernel to use. Every distinct value
# returned will instantiate a ncclDevKernel specialized to run this func
# without function call overhead.
def best_kernel(coll, redop, ty, algo, proto):
  def best(coll, redop, ty, algo, proto):
    # Modify this logic to control how many kernels are specialized.
    if coll=="Nop": return ("Generic", None, None, None, None)
    if coll=="SendRecv": return ("SendRecv", None, None, None, None)
    if coll in ("AllGather","Broadcast"): return (coll, None, None, "RING", "LL")
    return (coll, "Sum", ty, ("TREE" if algo=="TREE" else "RING"), "LL")
  # Need to ensure kernel is specialize for a primary function
  kfn = equivalent_primary(*best(coll, redop, ty, algo, proto))
  # And isn't filtered out.
  if not func_filter(*kfn): return ("Generic", None, None, None, None)
  return kfn

# Order rows are enumerated must match formula of `ncclDevFuncId()`:
def enumerate_func_rows():
  yield ("SendRecv", None, None, None, None)
  for coll in ("AllGather", "Broadcast"):
    algos = algos_of_coll[coll]
    for algo in algos:
      for proto in all_protos:
        yield (coll, None, None, algo, proto)
  for coll in ("AllReduce", "Reduce", "ReduceScatter"):
    algos = algos_of_coll[coll]
    for redop in all_redops:
      for ty in all_tys:
        for algo in algos:
          for proto in all_protos:
            yield (coll, redop, ty, algo, proto)

################################################################################

def is_built(coll, redop, ty, algo, proto):
  built = required_cuda(coll, redop, ty, algo, proto)
  built = built and func_filter(coll, redop, ty, algo, proto)
  return built

# Returns None if required_cuda(...) is None.
# Returns the coll="Nop" function if developer has filtered it out.
# Otherwise just returns func it was given.
def validate(coll, redop, ty, algo, proto):
  valid = required_cuda(coll, redop, ty, algo, proto)
  built = valid and func_filter(coll, redop, ty, algo, proto)
  if built: return (coll, redop, ty, algo, proto)
  if valid: return ("Nop", None, None, None, None)
  return None

# Corresponds to ncclDevFuncRowToId[]
func_rows = [validate(*fn) for fn in enumerate_func_rows()]

# Corresponds to ncclDevFuncTable[]
primary_funcs = sorted(set(equivalent_primary(*fn) for fn in func_rows if fn is not None))

# primary_to_index[primary_funcs[i]] == i
primary_to_index = {fn: i for (i,fn) in zip(range(len(primary_funcs)), primary_funcs)}

kernel_funcs = sorted(set(best_kernel(*fn) for fn in primary_funcs))

################################################################################

# Generate <gensrc>/device_table.cu
with open(os.path.join(gensrc, "device_table.cu"), "w") as f:
  out = f.write
  out('#include "common.h"\n')
  out("\n")

  for fn in primary_funcs:
    sym = paste("_", "ncclDevFunc", *fn)
    cudart, arch = required_cuda(*fn)
    if (cudart, arch) != (0, 0):
      out("#if CUDART_VERSION >= %d && __CUDA_ARCH__ >= %d\n" % (cudart, arch))
    out("__device__ void %s();\n" % sym)
    if (cudart, arch) != (0, 0):
      out("#endif\n")
  out("\n")

  out("__device__ ncclDevFuncPtr_t const ncclDevFuncTable[] = {\n");
  index = 0
  for fn in primary_funcs:
    sym = paste("_", "ncclDevFunc", *fn)
    cudart, arch = required_cuda(*fn)
    if (cudart, arch) != (0, 0):
      out("#if CUDART_VERSION >= %d && __CUDA_ARCH__ >= %d\n" % (cudart ,arch))
    out("/*%4d*/ %s,\n" % (index, sym))
    if (cudart, arch) != (0, 0):
      out("#else\n" "/*%4d*/ nullptr,\n" "#endif\n" % index)
    index += 1
  out("nullptr};\n")
  out("\n")

  out("// Workaround for https://reviews.llvm.org/D55580\n"
      "__device__ void ncclWorkaroundClangD55580() {}\n")

# Generate <gensrc>/host_table.cc
with open(os.path.join(gensrc, "host_table.cc"), "w") as f:
  out = f.write
  out('#include "device.h"\n')
  out("\n")

  # The mapping from function rows to valid primary function ids.
  out("extern int const ncclDevFuncRowToId[] = {\n")
  index = 0
  for fn in func_rows:
    fn_id, comment = -1, ""
    if fn is not None:
      fn_id = primary_to_index[equivalent_primary(*fn)]
      comment = " // " + paste(" ", *fn)
    out("/*%4d*/ %d,%s\n" % (index, fn_id, comment))
    index += 1
  out("-1};\n")
  out("\n")

  # Forward declarations of kernels.
  for kfn in kernel_funcs:
    cudart, _ = required_cuda(*kfn)
    sym = paste("_", "ncclDevKernel", *kfn)
    if cudart != 0: out("#if CUDART_VERSION >= %d\n" % cudart)
    out("__global__ void %s(struct ncclDevComm*, uint64_t, struct ncclWork*);\n" % sym)
    if cudart != 0: out("#endif\n")
  out("\n")

  # List of all kernel function pointers.
  out("extern int const ncclDevKernelCount = %d;\n" % len(kernel_funcs))
  out("extern void* const ncclDevKernelList[] = {\n")
  index = 0
  for kfn in kernel_funcs:
    cudart, _ = required_cuda(*kfn)
    sym = paste("_", "ncclDevKernel", *kfn)
    if cudart != 0: out("#if CUDART_VERSION >= %d\n" % cudart)
    out("/*%4d*/ (void*)%s,\n" % (index, sym));
    if cudart != 0: out("#else\n" "/*%4d*/ nullptr,\n" "#endif\n" % index)
    index += 1
  out("nullptr};\n")
  out("\n")

  # Maps primary id to kernel function pointer.
  out("extern void* const ncclDevKernelForFunc[] = {\n")
  index = 0
  for fn in primary_funcs:
    kfn = best_kernel(*fn)
    sym = paste("_", "ncclDevKernel", *kfn)
    cudart, _ = required_cuda(*kfn)
    if cudart != 0: out("#if CUDART_VERSION >= %d\n" % cudart)
    out("/*%4d*/ (void*)%s,\n" % (index, sym))
    if cudart != 0: out("#else\n" "/*%4d*/ nullptr,\n" "#endif\n" % index)
    index += 1
  out("nullptr};\n")
  out("\n")

  # Does the prior map use an explicitly specialized kernel.
  out("extern bool const ncclDevKernelForFuncIsSpecialized[] = {\n")
  index = 0
  for fn in primary_funcs:
    kfn = best_kernel(*fn)
    specialized = "1" if fn == kfn else "0"
    out("/*%4d*/ %s,\n" % (index, specialized))
    index += 1
  out("0};\n")

# Maps to .cu filename which implements this func. The only constraint is that
# "coll" is reflected in the name: formally that no two funcs having different
# coll's map to the same filename.
def impl_filename(coll, redop, ty, algo, proto):
  return "%s.cu" % paste("_", coll_camel_to_lower[coll], redop and redop.lower(), ty)

# Partition the functions and kernels to the .cu filenames. The partition is
# a dictionary mapping filename to (coll, func-tuple list)
def partition_by_name(fns):
  ans = {}
  for fn in fns:
    name = impl_filename(*fn)
    coll = fn[0]
    if name not in ans:
      ans[name] = (coll, [])
    ans[name][1].append(fn)
  return ans

name_to_funcs = partition_by_name(fn for fn in primary_funcs if fn[0]!="Nop")
name_to_kernels = partition_by_name(kfn for kfn in kernel_funcs if kfn[0]!="Generic")

# Generate <gensrc>/rules.mk
with open(os.path.join(gensrc, "rules.mk"), "w") as f:
  out = f.write
  impl_names = sorted(name_to_funcs.keys())
  names = impl_names + ["host_table.cc", "device_table.cu"]
  out("LIB_OBJS_GEN = $(patsubst %, $(OBJDIR)/genobj/%.o, {names})\n"
      .format(names=" ".join(names)))
  out("\n")

  # For each <coll>_<op>_<ty>.cu compile to a .cu.o file. Notice the dependencies
  # come from the suffix-erased file (e.g. 'gensrc/all_reduce.cu')
  for name in impl_names:
    coll = name_to_funcs[name][0]
    out(
      "$(OBJDIR)/genobj/{name}.o: $(OBJDIR)/gensrc $(OBJDIR)/genobj/{lower_coll}.cu.d\n"
      "\t" "$(call COMPILE,$@,$(OBJDIR)/gensrc/{name})\n"
      "\n"
      .format(name=name, lower_coll=coll_camel_to_lower[coll])
    )

# Add the suffix-erased .cu's which are used only for dependency scraping.
for coll in set(coll for (coll,_,_,_,_) in primary_funcs if coll!="Nop"):
  name = impl_filename(coll, None, None, None, None)
  if name not in name_to_funcs:
    name_to_funcs[name] = (coll, [])

redop_to_cxx = {
  None: "FuncCopy",
  "Sum": "FuncSum",
  "Prod": "FuncProd",
  "MinMax": "FuncMinMax",
  "PreMulSum": "FuncPreMulSum",
  "SumPostDiv": "FuncSumPostDiv"
}

ty_to_cxx = {
  None: "int8_t",
  "i8": "int8_t",
  "u8": "uint8_t",
  "i32": "int32_t",
  "u32": "uint32_t",
  "i64": "int64_t",
  "u64": "uint64_t",
  "f16": "half",
  "f32": "float",
  "f64": "double",
  "bf16": "__nv_bfloat16"
}

# Generate each <gensrc>/<impl>.cu:
for name in name_to_funcs.keys():
  (coll, fns) = name_to_funcs[name]
  with open(os.path.join(gensrc, name), "w") as f:
    out = f.write
    out(
      '#include "common.h"\n'
      '#include "{lower_coll}.h"\n'
      .format(lower_coll=coll_camel_to_lower[coll])
    )

    (_, kfns) = name_to_kernels.get(name) or (None, [])
    for kfn in kfns:
      (coll, redop, ty, algo, proto) = kfn
      sym = paste("_", coll, redop, ty, algo, proto)
      fn_id = primary_to_index[kfn]
      cudart, arch = required_cuda(*kfn)
      if (cudart, arch) != (0, 0):
        out("#if CUDART_VERSION >= %d && __CUDA_ARCH__ >= %d\n" % (cudart, arch))
      out(
        "DEFINE_ncclDevKernel({sym}, ncclFunc{coll}, {redop_cxx}, {ty_cxx}, NCCL_ALGO_{algo}, NCCL_PROTO_{proto}, {fn_id})\n"
        .format(sym=sym, coll=coll, redop_cxx=redop_to_cxx[redop], ty_cxx=ty_to_cxx[ty],
                algo=(algo or "RING"), proto=(proto or "SIMPLE"), fn_id=fn_id)
      )
      if (cudart, arch) != (0, 0):
        out("#endif\n")

    for fn in fns:
      (coll, redop, ty, algo, proto) = fn
      sym = paste("_", coll, redop, ty, algo, proto)
      cudart, arch = required_cuda(*fn)
      if (cudart, arch) != (0, 0):
        out("#if CUDART_VERSION >= %d && __CUDA_ARCH__ >= %d\n" % (cudart, arch))
      out(
        "DEFINE_ncclDevFunc({sym}, ncclFunc{coll}, {redop_cxx}, {ty_cxx}, NCCL_ALGO_{algo}, NCCL_PROTO_{proto})\n"
        .format(sym=sym, coll=coll, redop_cxx=redop_to_cxx[redop], ty_cxx=ty_to_cxx[ty],
                algo=(algo or "RING"), proto=(proto or "SIMPLE"))
      )
      if (cudart, arch) != (0, 0):
        out("#endif\n")
