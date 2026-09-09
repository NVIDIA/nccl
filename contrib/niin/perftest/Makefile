# NIIN perftest Makefile
#
# Usage:
#   make NCCL_HOME=<path-to-nccl-build> MPI_HOME=<path-to-mpi> ARCH=sm_89
#   mpirun -np 2 ./build/shmem_put_latency
#
# All tests require exactly 2 PEs (mpirun -np 2).

NCCL_HOME ?= $(shell cd ../../.. && pwd)
MPI_HOME  ?= /usr/lib/x86_64-linux-gnu/openmpi
ARCH      ?= sm_89

NVCC      := nvcc
NVCUFLAGS := --expt-relaxed-constexpr -std=c++17 -arch=$(ARCH) -DNIIN_HAS_MPI
INCLUDES  := -I common -I ../include -I $(NCCL_HOME)/build/include -I $(MPI_HOME)/include
LDFLAGS   := -L $(NCCL_HOME)/build/lib -lnccl -L $(MPI_HOME)/lib -lmpi -lcuda -lcudart -lrt \
             -Xlinker -rpath,$(NCCL_HOME)/build/lib -Xlinker -rpath,$(MPI_HOME)/lib

BUILDDIR  := build
SRCDIR    := device/pt-to-pt

TESTS := shmem_put_latency shmem_put_bw shmem_p_latency shmem_get_latency shmem_put_signal_latency

TARGETS := $(addprefix $(BUILDDIR)/,$(TESTS))

.PHONY: all clean

all: $(TARGETS)

$(BUILDDIR)/%: $(SRCDIR)/%.cu common/perftest.h | $(BUILDDIR)
	$(NVCC) $(NVCUFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR)
