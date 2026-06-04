# custom_algos: MC AllReduce benchmark
#
# Build with:
#   make -C contrib/custom_algos/allreduce MPI=1
#
# Requires NCCL to be built first:
#   make src.build
#
# Optional variables:
#   MPI_HOME    — path to MPI installation (default: detected from mpicc)
#   CUDA_HOME   — path to CUDA installation (default: /usr/local/cuda)
#   DEBUG=1     — enable debug build (-O0 -G -g)

include ../../../makefiles/common.mk

BUILDDIR  ?= $(abspath ../../../build)
DST_DIR   := $(BUILDDIR)/test/custom_algos/allreduce

# Include paths:
#   - NCCL public headers (nccl.h, nccl_device.h, etc.) from build/include
#   - NCCL device API inline implementations from build/include/nccl_device
#     and src/include (needed transitively by nccl_device.h)
#   - Current directory (kernel headers)
INCFLAGS := \
    -I. \
    -I$(BUILDDIR)/include \
    -I$(BUILDDIR)/include/nccl_device \
    -I../../../src/include

NVCUFLAGS := $(NVCC_GENCODE) $(INCFLAGS) \
    --expt-relaxed-constexpr \
    --compiler-options "-fPIC" \
    -ccbin $(CXX) $(CXXSTD)

.PHONY: all clean

ifeq ($(MPI), 1)
MPI_HOME    ?= $(shell mpicc --showme:prefix 2>/dev/null || echo /usr/local/mpi)
MPI_CFLAGS  := -I$(MPI_HOME)/include
MPI_LDFLAGS := -L$(MPI_HOME)/lib -lmpi

LDFLAGS := -L$(BUILDDIR)/lib -lnccl $(MPI_LDFLAGS) -lpthread -lrt -ldl -lcuda

all: $(DST_DIR)/allreduce

$(DST_DIR)/allreduce: main.cu \
    oneshot_mc_allreduce_kernel.cuh \
    twoshot_mc_simple_allreduce_kernel.cuh \
    twoshot_mc_poison_allreduce_kernel.cuh
	@mkdir -p $(DST_DIR)
	@printf "Compiling+linking allreduce\n"
	$(NVCC) $(NVCUFLAGS) $(MPI_CFLAGS) $< -o $@ $(LDFLAGS)
	@printf "Built: $@\n"

else
all:
	@echo "Skipping allreduce (MPI not enabled). Build with: make MPI=1"
endif

clean:
	rm -rf $(DST_DIR)
