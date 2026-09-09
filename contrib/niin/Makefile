# NIIN build helper.
#
# NIIN's device implementation lives in headers. This Makefile prepares the
# app-facing NIIN_HOME tree so applications can use:
#   -I$(NIIN_HOME)/include -L$(NIIN_HOME)/lib -lnvshmem_host

NCCL_HOME ?= ../..
PREFIX ?= $(CURDIR)

INCDIR := $(PREFIX)/include
LIBDIR := $(PREFIX)/lib
NCCL_INCLUDE := $(abspath $(NCCL_HOME)/build/include)
NCCL_LIB := $(abspath $(NCCL_HOME)/build/lib/libnccl.so)
NCCL_RUNTIME_LIB := $(LIBDIR)/libnccl.so.2
NVSHMEM_HOST_LIB := $(LIBDIR)/libnvshmem_host.so
NVSHMEM_LIB := $(LIBDIR)/libnvshmem.so
NCCL_HEADER_LINKS := $(INCDIR)/nccl.h $(INCDIR)/nccl_device.h $(INCDIR)/nccl_device

.PHONY: all clean

all: $(NCCL_HEADER_LINKS) $(NVSHMEM_HOST_LIB) $(NVSHMEM_LIB) $(NCCL_RUNTIME_LIB)

$(INCDIR)/nccl.h:
	@if [ ! -e "$(NCCL_INCLUDE)/nccl.h" ]; then \
	  echo "NCCL header not found: $(NCCL_INCLUDE)/nccl.h"; \
	  echo "Build NCCL first or pass NCCL_HOME=/path/to/nccl"; \
	  exit 1; \
	fi
	ln -sfn "$(NCCL_INCLUDE)/nccl.h" "$@"

$(INCDIR)/nccl_device.h:
	@if [ ! -e "$(NCCL_INCLUDE)/nccl_device.h" ]; then \
	  echo "NCCL header not found: $(NCCL_INCLUDE)/nccl_device.h"; \
	  echo "Build NCCL first or pass NCCL_HOME=/path/to/nccl"; \
	  exit 1; \
	fi
	ln -sfn "$(NCCL_INCLUDE)/nccl_device.h" "$@"

$(INCDIR)/nccl_device:
	@if [ ! -d "$(NCCL_INCLUDE)/nccl_device" ]; then \
	  echo "NCCL device header directory not found: $(NCCL_INCLUDE)/nccl_device"; \
	  echo "Build NCCL first or pass NCCL_HOME=/path/to/nccl"; \
	  exit 1; \
	fi
	ln -sfn "$(NCCL_INCLUDE)/nccl_device" "$@"

$(NVSHMEM_HOST_LIB) $(NCCL_RUNTIME_LIB):
	@if [ ! -e "$(NCCL_LIB)" ]; then \
	  echo "NCCL library not found: $(NCCL_LIB)"; \
	  echo "Build NCCL first or pass NCCL_HOME=/path/to/nccl"; \
	  exit 1; \
	fi
	@mkdir -p "$(LIBDIR)"
	ln -sfn "$(NCCL_LIB)" "$@"

$(NVSHMEM_LIB): $(NVSHMEM_HOST_LIB)
	ln -sfn "$(notdir $(NVSHMEM_HOST_LIB))" "$@"

clean:
	rm -f "$(NVSHMEM_HOST_LIB)" "$(NVSHMEM_LIB)" "$(NCCL_RUNTIME_LIB)" $(NCCL_HEADER_LINKS)
