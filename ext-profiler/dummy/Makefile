#
# Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#
NCCL_HOME:=../../build/
CUDA_HOME:=/usr/local/cuda
INC:= -I$(NCCL_HOME)/include -I$(CUDA_HOME)/include
PLUGIN_SO:=libnccl-profiler.so

default: $(PLUGIN_SO)

$(PLUGIN_SO): dummy_profiler.cc
	$(CC) $(INC) -fPIC -shared -o $@ -Wl,-soname,$(PLUGIN_SO) $^

clean:
	rm -f $(PLUGIN_SO)
