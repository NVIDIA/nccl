#
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#
.PHONY=test

NCCL_SRC:=../..
NCCL_HOME:=../../build
CUDA_HOME:=/usr/local/cuda
INC:= -I$(NCCL_HOME)/include -I$(CUDA_HOME)/include -I$(NCCL_SRC)/ext-net/example/nccl -I$(NCCL_SRC)/ext-tuner/example/nccl
PLUGIN_SO:=libnccl-mixed.so

default: $(PLUGIN_SO)

$(PLUGIN_SO): plugin.c
	$(CC) $(INC) -fPIC -shared -o $@ -Wl,-soname,$(PLUGIN_SO) $^

test: $(PLUGIN_SO)
	@bash test.sh

clean:
	rm -f $(PLUGIN_SO)
