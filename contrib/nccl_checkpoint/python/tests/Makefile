NCCL_SRC ?= ../../../../src
NCCL_HOME ?= $(error set NCCL_HOME to your NCCL install/build prefix)
CUDA_HOME ?= $(error set CUDA_HOME to your CUDA install prefix)

CXX ?= g++
CXXFLAGS ?= -std=c++17 -ffunction-sections -fdata-sections
TARGET := comm_hash_two_local_ranks
SRC := comm_hash_two_local_ranks.c

CPPFLAGS := \
	-DNCCL_OS_LINUX=1 \
	-I$(CUDA_HOME)/include \
	-I$(CUDA_HOME)/include/cccl \
	-I$(NCCL_SRC)/../build/include \
	-I$(NCCL_SRC)/include \
	-I$(NCCL_SRC)/include/plugin

LDFLAGS := \
	-Wl,--gc-sections \
	-L$(CUDA_HOME)/lib64 \
	-L$(NCCL_HOME)/lib

LDLIBS := -lnccl -lcudart

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) -x c++ $(CXXFLAGS) $(CPPFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGET)
