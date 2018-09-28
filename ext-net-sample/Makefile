#
# Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#
default: build

NCCL_HOME:=../
NCCL_INCD:=$(NCCL_HOME)build/include/
INC:=-I$(NCCL_INCD) -I/usr/local/cuda/include
PLUGIN_SO:=libnccl-net.so

build: $(PLUGIN_SO)

$(PLUGIN_SO): plugin.c | $(NCCL_INCD)nccl_net.h
	gcc -ggdb $(INC) -fPIC -shared -o $@ -Wl,-soname,$(PLUGIN_SO) $^
