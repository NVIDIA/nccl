# NIIN perftest Makefile
#
# Usage:
#   make NCCL_HOME=<path-to-nccl-build> MPI_HOME=<path-to-mpi> ARCH=sm_89
#   mpirun -np 2 ./build/shmem_put_latency
#
# All tests require exactly 2 PEs (mpirun -np 2).
#
# shmem_atomic_bw links the NIIN GPUNetIO atomic provider, which `make -C ..`
# builds by default; it is skipped when that archive is absent. Match the
# provider's GPUNETIO_CUDA_ARCHITECTURES to the ARCH used here (89-real and
# sm_89 on L40S).

NCCL_HOME ?= $(shell cd ../../.. && pwd)
MPI_HOME  ?= /usr/lib/x86_64-linux-gnu/openmpi
ARCH      ?= sm_89
# Set NCCL_SOURCE_ROOT when NCCL_HOME is a build/install prefix rather than the
# matching source checkout that carries the vendored GPUNetIO sources.
NCCL_SOURCE_ROOT ?= $(NCCL_HOME)
GPUNETIO_HOME ?= $(NCCL_SOURCE_ROOT)/src/transport/net_ib/gdaki/doca-gpunetio
GPUNETIO_ATOMICS_LIB ?= ../lib/libniin_gpunetio_atomics.a
IBVERBS_LIBRARY ?= -libverbs

# An absolute verbs library needs a matching runtime search path as well.
ifneq ($(filter /%,$(IBVERBS_LIBRARY)),)
IBVERBS_RPATH := -Xlinker -rpath,$(dir $(IBVERBS_LIBRARY))
endif

NVCC      := nvcc
NVCUFLAGS := --expt-relaxed-constexpr -std=c++17 -arch=$(ARCH) -DNIIN_HAS_MPI
INCLUDES  := -I common -I ../include -I $(NCCL_HOME)/build/include -I $(MPI_HOME)/include
LDFLAGS   := -L $(NCCL_HOME)/build/lib -lnccl -L $(MPI_HOME)/lib -lmpi -lcuda -lcudart -lrt \
             -Xlinker -rpath,$(NCCL_HOME)/build/lib -Xlinker -rpath,$(MPI_HOME)/lib

BUILDDIR  := build
SRCDIR    := device/pt-to-pt

TESTS := shmem_put_latency shmem_put_bw shmem_p_latency shmem_get_latency shmem_put_signal_latency

TARGETS := $(addprefix $(BUILDDIR)/,$(TESTS))
ATOMIC_TARGET := $(BUILDDIR)/shmem_atomic_bw

.PHONY: all atomic atomic-if-available clean

all: $(TARGETS) atomic-if-available

# The provider archive comes from `make -C ..`. Skip the test rather than fail
# the whole perftest build when NIIN was built without it.
atomic-if-available:
	@if [ -f "$(GPUNETIO_ATOMICS_LIB)" ] && [ -d "$(GPUNETIO_HOME)" ]; then \
	  $(MAKE) atomic; \
	else \
	  echo "Skipping shmem_atomic_bw: no GPUNetIO atomic provider at $(GPUNETIO_ATOMICS_LIB)"; \
	fi

# Link the provider archive after the CUDA object so its private GPUNetIO,
# ibverbs, CUDA-driver, dl, and pthread dependencies resolve normally.
atomic: $(ATOMIC_TARGET)

$(GPUNETIO_ATOMICS_LIB):
	@if [ ! -f "$@" ]; then \
	  echo "GPUNetIO atomic provider archive not found: $@"; \
	  echo "Build it first with: make -C .. gpunetio-atomics"; \
	  exit 1; \
	fi

$(ATOMIC_TARGET): $(SRCDIR)/shmem_atomic_bw.cu common/perftest.h $(GPUNETIO_ATOMICS_LIB) | $(BUILDDIR)
	@if [ ! -d "$(GPUNETIO_HOME)" ]; then \
	  echo "GPUNetIO sources not found: $(GPUNETIO_HOME)"; \
	  echo "Pass GPUNETIO_HOME=/path/to/unmodified/gpunetio-open"; \
	  exit 1; \
	fi
	$(NVCC) $(NVCUFLAGS) -DNIIN_GPUNETIO_ENABLE=1 $(INCLUDES) -I $(GPUNETIO_HOME)/include $< -o $@ \
	  $(GPUNETIO_ATOMICS_LIB) $(LDFLAGS) $(IBVERBS_LIBRARY) $(IBVERBS_RPATH) -ldl -lpthread

$(BUILDDIR)/%: $(SRCDIR)/%.cu common/perftest.h | $(BUILDDIR)
	$(NVCC) $(NVCUFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR)
