#
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#
NCCL_HOME := ../../build
INC := -I$(NCCL_HOME)/include -I$(CUDA_HOME)/include -Inccl
PLUGIN_SO := libnccl-profiler.so

default: $(PLUGIN_SO)

$(PLUGIN_SO): plugin.c event.c print_event.c
	$(CXX) $(INC) -g -fPIC -shared -o $@ -Wl,-soname,$(PLUGIN_SO) $^

clean:
	rm -f $(PLUGIN_SO)
