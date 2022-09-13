#!/bin/bash
#
# Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

dir=$1

datatypes="i8 u8 i32 u32 i64 u64 f16 f32 f64"
if [ "$CUDA_MAJOR" -ge 11 ]
then
    datatypes+=" bf16"
fi

echo "CURDIR := \$(dir \$(realpath \$(word \$(words \$(math\$(MAKEFILE_LIST))-1), \$(MAKEFILE_LIST))))"
+echo ""

targets="GENOBJS := \\\\\n"

for base in sendrecv all_reduce all_gather broadcast reduce reduce_scatter; do
  opn=0
  for op in sum prod min max premulsum sumpostdiv; do
    dtn=0
    # Order must match that of the ncclDataType_t enum
    for dt in ${datatypes}; do
      # Generate a unique filename for each compilation unit,
      # otherwise the __nv_module_id may conflict at link time
      echo "${dir}/${base}_${opn}_${dtn}.cu :"
      echo "	echo \"#include \\\"\$(CURDIR)${base}.cu\\\"\" > \$@"
      echo ""
      # Compile the file
      echo "${dir}/${base}_${op}_${dt}.o : ${dir}/${base}_${opn}_${dtn}.cu ${base}.cu ${dir}/${base}.dep"

      echo "	@printf \"Compiling  %-35s > %s\\\\n\" ${base}.cu ${dir}/${base}_${op}_${dt}.o"
      echo "	mkdir -p ${dir}"
      echo "	\${NVCC} -DNCCL_OP=${opn} -DNCCL_TYPE=${dtn} \${NVCUFLAGS} -dc \$< -o \$@"
      echo ""
      targets="$targets\t${dir}/${base}_${op}_${dt}.o \\\\\n"
      dtn=$(($dtn + 1))
    done
    opn=$(($opn + 1))
  done
done
echo -e "$targets"
