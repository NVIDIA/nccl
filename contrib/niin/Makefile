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
GPUNETIO_ATOMICS_LIB := $(LIBDIR)/libniin_gpunetio_atomics.a
NCCL_HEADER_LINKS := $(INCDIR)/nccl.h $(INCDIR)/nccl_device.h $(INCDIR)/nccl_device
GPUNETIO_HEADER_LINK := $(INCDIR)/gpunetio
# Required when GPUNETIO_HOME names an NCCL-vendored provider but NCCL_HOME is
# an install/build prefix rather than the matching source checkout.
NCCL_SOURCE_ROOT ?= $(NCCL_HOME)
# NCCL vendors the GPUNetIO sources GDAKI is built against, so the atomic
# provider builds out of the box. Point this elsewhere to use another
# unmodified GPUNetIO tree.
GPUNETIO_HOME ?= $(NCCL_SOURCE_ROOT)/src/transport/net_ib/gdaki/doca-gpunetio
GPUNETIO_BUILD_DIR ?= $(CURDIR)/gpunetio/build
# Empty retains CMake's default. Set a target-specific value such as `89-real`
# when the resulting CUDA artifacts will run only on L40S GPUs.
GPUNETIO_CUDA_ARCHITECTURES ?=
NIIN_GPUNETIO_ENABLE_SRQ_PROXY_ATOMICS ?= OFF
GDRCOPY_INCLUDE_DIR ?=
IBVERBS_INCLUDE_DIR ?=
IBVERBS_LIBRARY ?=

ifneq ($(strip $(GDRCOPY_INCLUDE_DIR)),)
GDRCOPY_CMAKE_ARG := -DGDRCOPY_INCLUDE_DIR="$(abspath $(GDRCOPY_INCLUDE_DIR))"
endif
ifneq ($(strip $(IBVERBS_INCLUDE_DIR)),)
IBVERBS_INCLUDE_CMAKE_ARG := -DIBVERBS_INCLUDE_DIR="$(abspath $(IBVERBS_INCLUDE_DIR))"
endif
ifneq ($(strip $(IBVERBS_LIBRARY)),)
IBVERBS_LIBRARY_CMAKE_ARG := -DIBVERBS_LIBRARY="$(abspath $(IBVERBS_LIBRARY))"
endif
ifneq ($(strip $(GPUNETIO_CUDA_ARCHITECTURES)),)
GPUNETIO_CUDA_ARCHITECTURES_CMAKE_ARG := -DCMAKE_CUDA_ARCHITECTURES="$(GPUNETIO_CUDA_ARCHITECTURES)"
endif

.PHONY: all clean gpunetio-atomics

all: $(NCCL_HEADER_LINKS) $(GPUNETIO_HEADER_LINK) $(NVSHMEM_HOST_LIB) $(NVSHMEM_LIB) \
     $(NCCL_RUNTIME_LIB) gpunetio-atomics

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

# The device side of the atomic provider is header-only. Linking the GPUNetIO
# include tree into NIIN's own include/ means `-I$(NIIN_HOME)/include` is enough
# for a CUDA consumer to get it, with no extra include path and no define.
#
# NCCL ships those sources, so a missing tree means NCCL_HOME/NCCL_SOURCE_ROOT
# points somewhere that is not an NCCL source checkout. That is a build
# configuration error, not a supported configuration: failing here beats
# shipping a NIIN whose network atomics are quietly unimplemented.
$(GPUNETIO_HEADER_LINK):
	@if [ ! -d "$(GPUNETIO_HOME)/include" ]; then \
	  echo "GPUNetIO sources not found: $(GPUNETIO_HOME)/include"; \
	  echo "NIIN builds from an NCCL source checkout. Pass NCCL_SOURCE_ROOT=/path/to/nccl"; \
	  echo "(or GPUNETIO_HOME=/path/to/gpunetio-open) when NCCL_HOME is a build or install prefix."; \
	  exit 1; \
	fi
	@mkdir -p "$(INCDIR)"
	ln -sfn "$(abspath $(GPUNETIO_HOME)/include)" "$@"
	@echo "NIIN GPUNetIO device headers: $@"

gpunetio-atomics:
	@if [ ! -d "$(GPUNETIO_HOME)" ]; then \
	  echo "GPUNetIO sources not found: $(GPUNETIO_HOME)"; \
	  echo "Pass GPUNETIO_HOME=/path/to/unmodified/gpunetio-open"; \
	  exit 1; \
	fi
	cmake -S "$(CURDIR)/gpunetio" -B "$(GPUNETIO_BUILD_DIR)" \
	  -DNCCL_HOME="$(abspath $(NCCL_HOME))" \
	  -DNCCL_SOURCE_ROOT="$(abspath $(NCCL_SOURCE_ROOT))" \
	  -DGPUNETIO_HOME="$(abspath $(GPUNETIO_HOME))" \
	  -DNIIN_GPUNETIO_ENABLE_SRQ_PROXY_ATOMICS="$(NIIN_GPUNETIO_ENABLE_SRQ_PROXY_ATOMICS)" $(GPUNETIO_CUDA_ARCHITECTURES_CMAKE_ARG) $(GDRCOPY_CMAKE_ARG) $(IBVERBS_INCLUDE_CMAKE_ARG) $(IBVERBS_LIBRARY_CMAKE_ARG)
	cmake --build "$(GPUNETIO_BUILD_DIR)" --target niin_gpunetio_compile_check
	@mkdir -p "$(LIBDIR)"
	ln -sfn "$(abspath $(GPUNETIO_BUILD_DIR))/libniin_gpunetio_atomics.a" "$(GPUNETIO_ATOMICS_LIB)"
	@echo "NIIN GPUNetIO atomics archive: $(GPUNETIO_ATOMICS_LIB)"

clean:
	rm -f "$(NVSHMEM_HOST_LIB)" "$(NVSHMEM_LIB)" "$(GPUNETIO_ATOMICS_LIB)" \
	  "$(NCCL_RUNTIME_LIB)" "$(GPUNETIO_HEADER_LINK)" $(NCCL_HEADER_LINKS)
