#
# Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#
NCCL_HOME:=../../build/
# For #include nccl_profiler.h (not exported yet)
NCCL_SRC:=../../src/
CUDA_HOME:=/usr/local/cuda
INC:= -I$(NCCL_HOME)/include -I$(CUDA_HOME)/include -I$(NCCL_SRC)/include
PLUGIN_SO:=libnccl-profiler.so

default: $(PLUGIN_SO)

$(PLUGIN_SO): profiler.cc
	$(CC) $(INC) -fPIC -shared -o $@ -Wl,-soname,$(PLUGIN_SO) $^

clean:
	rm -f $(PLUGIN_SO)
