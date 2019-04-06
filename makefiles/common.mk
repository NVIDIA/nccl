#
# Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

CUDA_HOME ?= /usr/local/cuda
PREFIX ?= /usr/local
VERBOSE ?= 0
KEEP ?= 0
DEBUG ?= 0
TRACE ?= 0
PROFAPI ?= 0

NVCC = $(CUDA_HOME)/bin/nvcc

CUDA_LIB ?= $(CUDA_HOME)/lib64
CUDA_INC ?= $(CUDA_HOME)/include
CUDA_VERSION = $(strip $(shell which $(NVCC) >/dev/null && $(NVCC) --version | grep release | sed 's/.*release //' | sed 's/\,.*//'))
#CUDA_VERSION ?= $(shell ls $(CUDA_LIB)/libcudart.so.* | head -1 | rev | cut -d "." -f -2 | rev)
CUDA_MAJOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 1)
CUDA_MINOR = $(shell echo $(CUDA_VERSION) | cut -d "." -f 2)
#$(info CUDA_VERSION ${CUDA_MAJOR}.${CUDA_MINOR})


# Better define NVCC_GENCODE in your environment to the minimal set
# of archs to reduce compile time.
CUDA8_GENCODE = -gencode=arch=compute_30,code=sm_30 \
                -gencode=arch=compute_35,code=sm_35 \
                -gencode=arch=compute_50,code=sm_50 \
                -gencode=arch=compute_60,code=sm_60 \
                -gencode=arch=compute_61,code=sm_61
CUDA9_GENCODE = -gencode=arch=compute_70,code=sm_70

CUDA8_PTX     = -gencode=arch=compute_61,code=compute_61
CUDA9_PTX     = -gencode=arch=compute_70,code=compute_70

# Include Volta support if we're using CUDA9 or above
ifeq ($(shell test "0$(CUDA_MAJOR)" -gt 8; echo $$?),0)
  NVCC_GENCODE ?= $(CUDA8_GENCODE) $(CUDA9_GENCODE) $(CUDA9_PTX)
else
  NVCC_GENCODE ?= $(CUDA8_GENCODE) $(CUDA8_PTX)
endif
#$(info NVCC_GENCODE is ${NVCC_GENCODE})

CXXFLAGS   := -DCUDA_MAJOR=$(CUDA_MAJOR) -DCUDA_MINOR=$(CUDA_MINOR) -fPIC -fvisibility=hidden
CXXFLAGS   += -Wall -Wno-unused-function -Wno-sign-compare -std=c++11 -Wvla
CXXFLAGS   += -I $(CUDA_INC)
NVCUFLAGS  := -ccbin $(CXX) $(NVCC_GENCODE) -lineinfo -std=c++11 -Xptxas -maxrregcount=96 -Xfatbin -compress-all
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

ifneq ($(VERBOSE), 0)
NVCUFLAGS += -Xptxas -v -Xcompiler -Wall,-Wextra,-Wno-unused-parameter
CXXFLAGS  += -Wall -Wextra
else
.SILENT:
endif

ifneq ($(TRACE), 0)
CXXFLAGS  += -DENABLE_TRACE
endif

ifneq ($(KEEP), 0)
NVCUFLAGS += -keep
endif

ifneq ($(PROFAPI), 0)
CXXFLAGS += -DPROFAPI
endif
