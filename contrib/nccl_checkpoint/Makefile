CXXSTD := -std=c++17

.PHONY: all install clean

NCCL_SRC ?= ../../src
NCCL_SRC := $(abspath $(NCCL_SRC))

include $(NCCL_SRC)/../makefiles/common.mk
include $(NCCL_SRC)/../makefiles/version.mk

OG ?= 0
ifneq ($(OG), 0)
CXXFLAGS := $(filter-out -O0 -O3,$(CXXFLAGS)) -Og -g
endif

BUILDDIR ?= $(abspath ./build)
BUILDDIR := $(abspath $(BUILDDIR))
OBJDIR := $(BUILDDIR)/obj/nccl_checkpoint
LIBDIR := $(BUILDDIR)/lib
GENDIR := $(BUILDDIR)/include/nccl_checkpoint
BINDIR := $(BUILDDIR)/bin
PREFIX_LIBDIR := $(PREFIX)/lib
PREFIX_INCLUDEDIR := $(PREFIX)/include

ifdef HIREDIS_HOME
  HIREDIS_CFLAGS := -I$(HIREDIS_HOME)/include
  HIREDIS_LIBS := $(HIREDIS_HOME)/lib/libhiredis.a
else
  HIREDIS_CFLAGS := $(shell pkg-config --cflags hiredis 2>/dev/null)
  HIREDIS_LIBS := $(shell pkg-config --libs hiredis 2>/dev/null)
endif

CUDALIBS := -L$(CUDA_LIB) -lcudart

SHIM_SO := $(LIBDIR)/libnccl-checkpoint-shim.so
PUBLIC_HEADER := nccl_checkpoint.h
NCCL_HEADER := $(GENDIR)/nccl.h
SHIM_HEADERS := $(NCCL_HEADER) $(NCCL_SRC)/include/nccl_device/core.h
SHIM_AUTO := $(GENDIR)/shim_auto.cc
SHIM_EXPORTS := $(GENDIR)/shim_exports.map

INCFLAGS = -I. -I$(GENDIR) -I$(NCCL_SRC)/include -I$(NCCL_SRC)/include/plugin $(HIREDIS_CFLAGS)
CXXFLAGS += $(INCFLAGS) -fPIC -fvisibility=default
# Dependency generation flags for automatic header tracking
DEPFLAGS = -MMD -MP

KV_TEST := $(BINDIR)/kv_sanity_test

all: $(SHIM_SO) $(KV_TEST)

$(NCCL_HEADER): $(NCCL_SRC)/nccl.h.in $(NCCL_SRC)/../makefiles/version.mk
	@$(eval NCCL_VERSION := $(shell printf "%d%02d%02d" $(NCCL_MAJOR) $(NCCL_MINOR) $(NCCL_PATCH)))
	@mkdir -p $(dir $@)
	@printf "Generating %-35s > %s\n" $< $@
	sed -e "s/\$${nccl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${nccl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${nccl:Patch}/$(NCCL_PATCH)/g" \
	    -e "s/\$${nccl:Suffix}/$(NCCL_SUFFIX)/g" \
	    -e "s/\$${nccl:Version}/$(NCCL_VERSION)/g" \
	    $< > $@

$(SHIM_AUTO): $(SHIM_HEADERS) gen_shim.py
	@mkdir -p $(dir $@)
	python3 gen_shim.py $(SHIM_HEADERS) >$@
$(SHIM_EXPORTS): $(SHIM_HEADERS) gen_shim.py
	@mkdir -p $(dir $@)
	python3 gen_shim.py $(SHIM_HEADERS) --exports >$@
$(OBJDIR)/shim.o: shim.cc $(SHIM_AUTO)

HOST_SRC := shim_core.cc shim.cc shim_checkpoint.cc kv_store_client.cc
HOST_OBJ := $(HOST_SRC:%.cc=$(OBJDIR)/%.o)
$(HOST_OBJ): $(NCCL_HEADER)

# Compile host code
$(OBJDIR)/%.o: %.cc
	@echo "Compiling $<"
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

$(SHIM_SO): $(HOST_OBJ) $(SHIM_EXPORTS)
	@echo "Linking $@"
	@mkdir -p $(dir $@)
	$(CXX) -shared -fPIC $(CXXFLAGS) $(LFLAGS) $(HOST_OBJ) -o $@ -Wl,--version-script=$(SHIM_EXPORTS) -ldl $(CUDALIBS) $(HIREDIS_LIBS)
$(KV_TEST): kv_sanity_test.cc kv_store_client.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXSTD) $(INCFLAGS) $^ -o $@ $(HIREDIS_LIBS)

install: $(SHIM_SO) $(PUBLIC_HEADER)
	mkdir -p $(PREFIX_LIBDIR)
	install -m 755 $(SHIM_SO) $(PREFIX_LIBDIR)/
	mkdir -p $(PREFIX_INCLUDEDIR)
	install -m 644 $(PUBLIC_HEADER) $(PREFIX_INCLUDEDIR)/

# Include auto-generated header dependencies (produced by -MMD -MP above)
-include $(HOST_OBJ:.o=.d)

clean:
	rm -rf $(OBJDIR) $(GENDIR) $(SHIM_SO) $(KV_TEST) $(BUILDDIR)
