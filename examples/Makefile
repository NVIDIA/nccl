CUDA_HOME ?= /usr/local/cuda
BUILDDIR ?= ../build

CUDA_LIB ?= $(CUDA_HOME)/lib64
CUDA_INC ?= $(CUDA_HOME)/include
NVCC = $(CUDA_HOME)/bin/nvcc
NCCL_HOME := $(abspath ../build/)

NVCUFLAGS +=  -I. -I$(NCCL_HOME)/include
NVLDFLAGS += -L$(NCCL_HOME)/lib -lnccl 

SRC_FILES := $(wildcard *.cc)
DST_DIR := $(BUILDDIR)/examples
OBJ_FILES := $(SRC_FILES:%.cc=$(DST_DIR)/%.o)
BIN_FILES := $(SRC_FILES:%.cc=$(DST_DIR)/%)

all: $(BIN_FILES)

clean:
	rm -rf $(DST_DIR)

${DST_DIR}/%.o: %.cc
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(NVCC) -o $@ $(NVCUFLAGS) -c $<

${DST_DIR}/%: $(DST_DIR)/%.o
	@printf "Linking  %-35s > %s\n" $< $@
	@mkdir -p $(DST_DIR)
	$(NVCC) -o $@ $(NVCUFLAGS) $^ $(NVLDFLAGS)
