#
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

# Variables
NCCL_HOME := ../../build
INC := -I$(NCCL_HOME)/include -I$(CUDA_HOME)/include -Inccl
PLUGIN_SO := libnccl-profiler-inspector.so
VERSION_FILE := version.cc

# Compiler and flags
CXX := g++
CXXFLAGS := -g -O3 -fPIC -shared -march=native -DNDEBUG -Wall -Wextra

ifeq ($(DEBUG), 1)
CXXFLAGS += -g2 -ggdb3 -rdynamic -funwind-tables -fno-omit-frame-pointer
endif

ifeq ($(ASAN), 1)
CXXFLAGS += -fsanitize=address
LDFLAGS += -fsanitize=address -static-libasan
NVLDFLAGS += -Xcompiler -fsanitize=address,-static-libasan
endif

ifeq ($(UBSAN), 1)
CXXFLAGS += -fsanitize=undefined
LDFLAGS += -fsanitize=undefined -static-libubsan
NVLDFLAGS += -Xcompiler -fsanitize=undefined,-static-libubsan
endif

# Source files
SOURCES := inspector_plugin.cc inspector.cc json.cc

# Default target
all: $(PLUGIN_SO)

# Rule to build the plugin
$(PLUGIN_SO): $(VERSION_FILE) $(SOURCES)
	@echo "Compiling to create $@ from $^"
	$(CXX) $(INC) $(CXXFLAGS) -o $@ -Wl,-soname,$(PLUGIN_SO) $^

# Rule to generate version.cc
$(VERSION_FILE):
	@GIT_INFO=$$(./utils/extract_git_version.sh); \
	echo '#include "version.h"' > $(VERSION_FILE).tmp; \
	echo 'const char* get_git_version_info() { return "'$$GIT_INFO'"; }' >> $(VERSION_FILE).tmp; \
	if ! cmp $(VERSION_FILE).tmp $(VERSION_FILE); then \
		echo "updating ${VERSION_FILE} file -> $$GIT_INFO"; \
		mv $(VERSION_FILE).tmp $(VERSION_FILE); \
	else \
		echo "${VERSION_FILE} up to date -> $$GIT_INFO"; \
		rm $(VERSION_FILE).tmp; \
	fi

# Clean target
clean:
	rm -f $(VERSION_FILE) $(PLUGIN_SO)

# Phony targets
.PHONY: all clean
