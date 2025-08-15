#!/bin/bash

set -euo pipefail
# This script accepts a Pytorch NGC package suffix (i.e. 23.12, 24.01, 24.08, etc), a git branch/ref, and an optional NCCL repo URL (default: https://gitlab-master.nvidia.com/ai-efficiency/nccl.git). It will attempt to build NCCL in that environment, then extract the built libraries back to the host filesystem

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Accept NCCL repo URL as third argument, or use default
NCCL_REPO_URL="https://gitlab-master.nvidia.com/ai-efficiency/nccl.git"
if [ -n "$3" ]; then
    NCCL_REPO_URL="$3"
fi

sanitized_ref="${2//[\//]/_}"

TMPDIR=$(mktemp -d -t "build-${1}-${sanitized_ref}.XXXX" --tmpdir=.)

CONTAINER_NAME=nvcr.io/nvidia/pytorch:${1}-py3
CU_ARCH_TARGETS=arch=compute_90,code=sm_90
DEPLOYMENT=${4:-}

# Use NCCL_REPO_URL in Dockerfile

echo "FROM ${CONTAINER_NAME}

RUN rm -f /usr/include/nccl*.h
RUN find /usr/lib -name libnccl.* -delete

COPY ./gen-package-data.sh /opt/
RUN git clone ${NCCL_REPO_URL} --branch ${2} --single-branch /opt/nccl
ENV NVCC_GENCODE=-gencode=${CU_ARCH_TARGETS}

WORKDIR /opt/nccl

RUN bash /opt/gen-package-data.sh ${CONTAINER_NAME} ${CU_ARCH_TARGETS} ${DEPLOYMENT} > /opt/package.json

RUN make clean
RUN make -j 16 DEBUG=0 MPI=1 CUDA_HOME=/usr/local/cuda/" > ${TMPDIR}/Dockerfile

ID=$(uuidgen)
docker build --no-cache -t ${ID} -f ${TMPDIR}/Dockerfile .

docker run --rm -it -v "$(readlink -f ${TMPDIR}):/package/:rw" ${ID} sh -c "cp /opt/package.json /package/package.json"

TAG=$(jq -r .nccl.version < "${TMPDIR}/package.json")
COMMIT_HASH=$(jq -r .nccl.commit < "${TMPDIR}/package.json")
DIRTY_BOOL=$(jq -r .nccl.dirty_repo < "${TMPDIR}/package.json")

if [ "${DIRTY_BOOL}" = "true" ]
then
    DIRTY="-dirty"
else
    DIRTY=""
fi

GIT_INFO=${TAG}-${COMMIT_HASH}${DIRTY}
OUTPUT_DIR="container-libs/ngc-pytorch-${1}-${GIT_INFO}"
mkdir ${OUTPUT_DIR}
docker run --rm -it -v "$(readlink -f ${OUTPUT_DIR}):/${OUTPUT_DIR}:rw" ${ID} sh -c "cp /opt/nccl/build/lib/libnccl* /${OUTPUT_DIR}"

cp ${TMPDIR}/package.json ${OUTPUT_DIR}/package.json

echo "created ${OUTPUT_DIR}"
cat ${OUTPUT_DIR}/package.json

rm -rf "${TMPDIR}"
