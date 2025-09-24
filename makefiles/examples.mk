#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

# Make sure NCCL headers are found and libraries are linked
ifneq ($(NCCL_HOME), "")
NVCUFLAGS += -I$(NCCL_HOME)/include/
NVLDFLAGS += -L$(NCCL_HOME)/lib
endif

# Build configuration
INCLUDES = -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include
LIBRARIES = -L$(CUDA_HOME)/lib64 -L$(NCCL_HOME)/lib
LDFLAGS = -lcudart -lnccl -Wl,-rpath,$(NCCL_HOME)/lib


# MPI configuration
ifeq ($(MPI), 1)

ifdef MPI_HOME
MPICXX ?= $(MPI_HOME)/bin/mpicxx
MPIRUN ?= $(MPI_HOME)/bin/mpirun
else
MPICXX ?= mpicxx
MPIRUN ?= mpirun
endif

CXXFLAGS += -DMPI_SUPPORT
endif
