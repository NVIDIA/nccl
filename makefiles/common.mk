#
# Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

CUDA_HOME ?= /usr/local/cuda
PREFIX ?= /usr/local
VERBOSE ?= 0
KEEP ?= 0
DEBUG ?= 0
ASAN ?= 0
UBSAN ?= 0
TRACE ?= 0
WERROR ?= 0
PROFAPI ?= 1
NVTX ?= 1
RDMA_CORE ?= 0
NET_PROFILER ?= 0
MLX5DV ?= 0
MAX_EXT_NET_PLUGINS ?= 0

NVCC = $(CUDA_HOME)/bin/nvcc

CUDA_LIB ?= $(CUDA_HOME)/lib64
CUDA_INC ?= $(CUDA_HOME)/include
CUDA_VERSION = $(strip $(shell which $(NVCC) >/dev/null && $(NVCC) --version | grep release | sed 's/.*release //' | sed 's/\,.*//'))
#CUDA_VERSION ?= $(shell ls $(CUDA_LIB)/libcudart.so.* | head -1 | rev | cut -d "." -f -2 | rev)
CUDA_MAJOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 1)
CUDA_MINOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 2)
#$(info CUDA_VERSION ${CUDA_MAJOR}.${CUDA_MINOR})

# You should define NVCC_GENCODE in your environment to the minimal set
# of archs to reduce compile time.
CUDA8_GENCODE = -gencode=arch=compute_50,code=sm_50 \
                -gencode=arch=compute_60,code=sm_60 \
                -gencode=arch=compute_61,code=sm_61
ifeq ($(shell test "0$(CUDA_MAJOR)" -lt 12; echo $$?),0)
# SM35 is deprecated from CUDA12.0 onwards
CUDA8_GENCODE += -gencode=arch=compute_35,code=sm_35
endif
CUDA9_GENCODE = -gencode=arch=compute_70,code=sm_70
CUDA11_GENCODE = -gencode=arch=compute_80,code=sm_80
CUDA12_GENCODE = -gencode=arch=compute_90,code=sm_90
CUDA13_GENCODE = -gencode=arch=compute_100,code=sm_100 \
                 -gencode=arch=compute_120,code=sm_120

CUDA8_PTX     = -gencode=arch=compute_61,code=compute_61
CUDA9_PTX     = -gencode=arch=compute_70,code=compute_70
CUDA11_PTX    = -gencode=arch=compute_80,code=compute_80
CUDA12_PTX    = -gencode=arch=compute_90,code=compute_90
CUDA13_PTX    = -gencode=arch=compute_120,code=compute_120

ifeq ($(shell test "0$(CUDA_MAJOR)" -ge 13; echo $$?),0)
# Prior to SM75 is deprecated from CUDA13.0 onwards
  NVCC_GENCODE ?= $(CUDA11_GENCODE) $(CUDA12_GENCODE) $(CUDA13_GENCODE) $(CUDA13_PTX)
else ifeq ($(shell test "0$(CUDA_MAJOR)" -eq 12 -a "0$(CUDA_MINOR)" -ge 8; echo $$?),0)
# Include Blackwell support if we're using CUDA12.8 or above
  NVCC_GENCODE ?= $(CUDA8_GENCODE) $(CUDA9_GENCODE) $(CUDA11_GENCODE) $(CUDA12_GENCODE) $(CUDA13_GENCODE) $(CUDA13_PTX)
else ifeq ($(shell test "0$(CUDA_MAJOR)" -eq 11 -a "0$(CUDA_MINOR)" -ge 8 -o "0$(CUDA_MAJOR)" -gt 11; echo $$?),0)
# Include Hopper support if we're using CUDA11.8 or above
  NVCC_GENCODE ?= $(CUDA8_GENCODE) $(CUDA9_GENCODE) $(CUDA11_GENCODE) $(CUDA12_GENCODE) $(CUDA12_PTX)
else ifeq ($(shell test "0$(CUDA_MAJOR)" -ge 11; echo $$?),0)
  NVCC_GENCODE ?= $(CUDA8_GENCODE) $(CUDA9_GENCODE) $(CUDA11_GENCODE) $(CUDA11_PTX)
# Include Volta support if we're using CUDA9 or above
else ifeq ($(shell test "0$(CUDA_MAJOR)" -ge 9; echo $$?),0)
  NVCC_GENCODE ?= $(CUDA8_GENCODE) $(CUDA9_GENCODE) $(CUDA9_PTX)
else
  NVCC_GENCODE ?= $(CUDA8_GENCODE) $(CUDA8_PTX)
endif
$(info NVCC_GENCODE is ${NVCC_GENCODE})

# CUDA 13.0 requires c++17
ifeq ($(shell test "0$(CUDA_MAJOR)" -ge 13; echo $$?),0)
  CXXSTD ?= -std=c++17
else
  CXXSTD ?= -std=c++11
endif

CXXFLAGS   := -DCUDA_MAJOR=$(CUDA_MAJOR) -DCUDA_MINOR=$(CUDA_MINOR) -fPIC -fvisibility=hidden \
              -Wall -Wno-unused-function -Wno-sign-compare $(CXXSTD) -Wvla \
              -I $(CUDA_INC) -I $(CUDA_INC)/cccl \
              $(CXXFLAGS)
# Maxrregcount needs to be set accordingly to NCCL_MAX_NTHREADS (otherwise it will cause kernel launch errors)
# 512 : 120, 640 : 96, 768 : 80, 1024 : 60
# We would not have to set this if we used __launch_bounds__, but this only works on kernels, not on functions.
NVCUFLAGS  := -ccbin $(CXX) $(NVCC_GENCODE) $(CXXSTD) --expt-extended-lambda -Xptxas -maxrregcount=96 -Xfatbin -compress-all
# Use addprefix so that we can specify more than one path
NVLDFLAGS  := -L${CUDA_LIB} -lcudart -lrt

########## GCOV ##########
GCOV ?= 0 # disable by default.
GCOV_FLAGS := $(if $(filter 0,${GCOV} ${DEBUG}),,--coverage) # only gcov=1 and debug =1
CXXFLAGS  += ${GCOV_FLAGS}
NVCUFLAGS += ${GCOV_FLAGS:%=-Xcompiler %}
LDFLAGS   += ${GCOV_FLAGS}
NVLDFLAGS   += ${GCOV_FLAGS:%=-Xcompiler %}
# $(warning GCOV_FLAGS=${GCOV_FLAGS})
########## GCOV ##########

ifeq ($(DEBUG), 0)
NVCUFLAGS += -O3
CXXFLAGS  += -O3 -g
else
NVCUFLAGS += -O0 -G -g
CXXFLAGS  += -O0 -g -ggdb3
endif

# Make sure to run with ASAN_OPTIONS=protect_shadow_gap=0 otherwise CUDA will fail with OOM
ifneq ($(ASAN), 0)
CXXFLAGS += -fsanitize=address
LDFLAGS += -fsanitize=address -static-libasan
NVLDFLAGS += -Xcompiler -fsanitize=address,-static-libasan
endif

ifneq ($(UBSAN), 0)
CXXFLAGS += -fsanitize=undefined
LDFLAGS += -fsanitize=undefined -static-libubsan
NVLDFLAGS += -Xcompiler -fsanitize=undefined,-static-libubsan
endif

ifneq ($(VERBOSE), 0)
NVCUFLAGS += -Xptxas -v -Xcompiler -Wall,-Wextra,-Wno-unused-parameter
CXXFLAGS  += -Wall -Wextra
else
.SILENT:
endif

ifneq ($(TRACE), 0)
CXXFLAGS  += -DENABLE_TRACE
endif

ifeq ($(NVTX), 0)
CXXFLAGS  += -DNVTX_DISABLE
endif

ifneq ($(WERROR), 0)
CXXFLAGS  += -Werror
endif

ifneq ($(KEEP), 0)
NVCUFLAGS += -keep
endif

ifneq ($(PROFAPI), 0)
CXXFLAGS += -DPROFAPI
endif

ifneq ($(RDMA_CORE), 0)
CXXFLAGS += -DNCCL_BUILD_RDMA_CORE=1 -libverbs
endif

ifneq ($(MLX5DV), 0)
CXXFLAGS += -DNCCL_BUILD_MLX5DV=1 -lmlx5
endif

ifneq ($(NET_PROFILER), 0)
CXXFLAGS += -DNCCL_ENABLE_NET_PROFILING=1
endif

ifneq ($(MAX_EXT_NET_PLUGINS), 0)
CXXFLAGS += -DNCCL_NET_MAX_PLUGINS=$(MAX_EXT_NET_PLUGINS)
endif
