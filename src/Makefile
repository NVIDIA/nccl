#
# Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#
include ../makefiles/common.mk
include ../makefiles/version.mk

CONDA_INCLUDE_DIR := /engshare/conda/torchcomms/include
CONDA_LIB_DIR     := /engshare/conda/torchcomms/lib
CUDA_HOME         := /usr/local/cuda

CXXFLAGS += -DOTEL_SCUBA_DATA -std=c++20 -Wno-deprecated-declarations -Wno-class-memaccess

##### src files
INCEXPORTS  := nccl.h
LIBSRCFILES := \
	bootstrap.cc channel.cc collectives.cc commDesc.cc debug.cc enqueue.cc group.cc \
	init.cc init_nvtx.cc proxy.cc transport.cc mnnvl.cc allocator.cc symmetric.cc \
	$(wildcard graph/*.cc) \
	$(wildcard misc/*.cc) \
	$(wildcard transport/*.cc) \
	$(wildcard register/*.cc) \
	$(wildcard plugin/*.cc) \
	$(wildcard plugin/net/*.cc) \
	$(wildcard plugin/tuner/*.cc) \
	$(wildcard plugin/profiler/*.cc) \
	$(filter-out ras/client.cc,$(wildcard ras/*.cc))
LIBSRCFILES += $(wildcard ../meta/*.cc)
LIBSRCFILES += $(wildcard ../meta/colltrace/*.cc)
LIBSRCFILES += $(wildcard ../meta/logger/*.cc)
LIBSRCFILES += $(wildcard ../meta/cvars/*.cc)
INCLUDES := -Iinclude
INCLUDES += -I../
INCLUDES += -I../../
INCLUDES += -I../meta
INCLUDES += -I../meta/logger -Idevice
INCLUDES += -I$(CONDA_INCLUDE_DIR)

BINSRCFILES := ras/client.cc

##### lib files
LIBNAME     := libnccl.so
STATICLIBNAME := libnccl_static.a
##### binaries
BINNAME := ncclras
##### pkgconfig files
PKGCONFIGFILE := nccl.pc
##### dirs
BUILDDIR ?= $(abspath ../build)
INCDIR := $(BUILDDIR)/include
LIBDIR := $(BUILDDIR)/lib
OBJDIR := $(BUILDDIR)/obj
PKGDIR := $(BUILDDIR)/lib/pkgconfig
BINDIR := $(BUILDDIR)/bin
##### target files
CUDARTLIB  ?= cudart_static

# Use compatibility shim only with static cudart; see https://github.com/NVIDIA/nccl/issues/658
ifeq ($(CUDARTLIB), cudart_static)
	LIBSRCFILES += enhcompat.cc
endif

THIRD_PARTY_LDFLAGS ?= -L${CONDA_LIB_DIR} \
                 -l:libfolly.a \
                 -l:libfolly_exception_tracer.a \
                 -l:libfolly_exception_tracer_base.a \
                 -l:libglog.a \
                 -l:libgflags.a \
                 -l:libfmt.a \
                 -l:libboost_context.a \
                 -l:libdouble-conversion.a \
                 -l:libevent.a \
                 -l:libsnappy.a \
                 -l:libsodium.a \
                 -l:libopentelemetry_exporter_otlp_http_log.a \
                 -l:libopentelemetry_otlp_recordable.a \
                 -l:libopentelemetry_exporter_otlp_http_client.a \
                 -l:libopentelemetry_http_client_curl.a \
                 -l:libopentelemetry_proto.a \
                 -l:libopentelemetry_logs.a \
                 -l:libopentelemetry_trace.a \
                 -l:libopentelemetry_metrics.a \
                 -l:libopentelemetry_resources.a \
                 -l:libopentelemetry_common.a \
                 -l:libprotobuf.a \
                 -l:libglog.a \
                 -l:libgflags.a \
                 -l:libfmt.a \
                 -l:libcurl.a \
                 -l:libssl.a \
                 -l:libcrypto.a \
                 -l:libzstd.a \
                 -l:libpsl.a \
                 -l:libz.a \
                 -l:libabsl_log_internal_check_op.a \
                 -l:libabsl_die_if_null.a \
                 -l:libabsl_log_internal_conditions.a \
                 -l:libabsl_log_internal_message.a \
                 -l:libabsl_log_internal_nullguard.a \
                 -l:libabsl_examine_stack.a \
                 -l:libabsl_log_internal_format.a \
                 -l:libabsl_log_internal_structured_proto.a \
                 -l:libabsl_log_internal_log_sink_set.a \
                 -l:libabsl_log_sink.a \
                 -l:libabsl_log_entry.a \
                 -l:libabsl_log_internal_proto.a \
                 -l:libabsl_flags_internal.a \
                 -l:libabsl_flags_marshalling.a \
                 -l:libabsl_flags_reflection.a \
                 -l:libabsl_flags_config.a \
                 -l:libabsl_flags_program_name.a \
                 -l:libabsl_flags_private_handle_accessor.a \
                 -l:libabsl_flags_commandlineflag.a \
                 -l:libabsl_flags_commandlineflag_internal.a \
                 -l:libabsl_log_initialize.a \
                 -l:libabsl_log_internal_globals.a \
                 -l:libabsl_log_globals.a \
                 -l:libabsl_vlog_config_internal.a \
                 -l:libabsl_log_internal_fnmatch.a \
                 -l:libabsl_raw_hash_set.a \
                 -l:libabsl_hashtablez_sampler.a \
                 -l:libabsl_random_distributions.a \
                 -l:libabsl_random_seed_sequences.a \
                 -l:libabsl_random_internal_entropy_pool.a \
                 -l:libabsl_random_internal_randen.a \
                 -l:libabsl_random_internal_randen_hwaes.a \
                 -l:libabsl_random_internal_randen_hwaes_impl.a \
                 -l:libabsl_random_internal_randen_slow.a \
                 -l:libabsl_random_internal_platform.a \
                 -l:libabsl_random_internal_seed_material.a \
                 -l:libabsl_random_seed_gen_exception.a \
                 -l:libabsl_statusor.a \
                 -l:libabsl_status.a \
                 -l:libabsl_cord.a \
                 -l:libabsl_cordz_info.a \
                 -l:libabsl_cord_internal.a \
                 -l:libabsl_hash.a \
                 -l:libabsl_city.a \
                 -l:libabsl_cordz_functions.a \
                 -l:libabsl_exponential_biased.a \
                 -l:libabsl_cordz_handle.a \
                 -l:libabsl_crc_cord_state.a \
                 -l:libabsl_crc32c.a \
                 -l:libabsl_crc_internal.a \
                 -l:libabsl_crc_cpu_detect.a \
                 -l:libabsl_leak_check.a \
                 -l:libabsl_strerror.a \
                 -l:libabsl_str_format_internal.a \
                 -l:libabsl_synchronization.a \
                 -l:libabsl_stacktrace.a \
                 -l:libabsl_symbolize.a \
                 -l:libabsl_debugging_internal.a \
                 -l:libabsl_demangle_internal.a \
                 -l:libabsl_demangle_rust.a \
                 -l:libabsl_decode_rust_punycode.a \
                 -l:libabsl_utf8_for_code_point.a \
                 -l:libabsl_graphcycles_internal.a \
                 -l:libabsl_kernel_timeout_internal.a \
                 -l:libabsl_malloc_internal.a \
                 -l:libabsl_tracing_internal.a \
                 -l:libabsl_time.a \
                 -l:libabsl_civil_time.a \
                 -l:libabsl_time_zone.a \
                 -l:libutf8_validity.a \
                 -l:libabsl_strings.a \
                 -l:libabsl_int128.a \
                 -l:libabsl_strings_internal.a \
                 -l:libabsl_string_view.a \
                 -l:libabsl_base.a \
                 -l:libabsl_spinlock_wait.a \
                 -l:libabsl_throw_delegate.a \
                 -l:libabsl_raw_logging_internal.a \
                 -l:libabsl_log_severity.a \
                 -l:libssh2.a \
                 -l:libnghttp2.a \
                 -lidn2 \
                 -lunistring \
                 -l:libiconv.a


INCTARGETS := $(INCEXPORTS:%=$(INCDIR)/%)
LIBSONAME  := $(LIBNAME:%=%.$(NCCL_MAJOR))
LIBTARGET  := $(LIBNAME:%=%.$(NCCL_MAJOR).$(NCCL_MINOR).$(NCCL_PATCH))
STATICLIBTARGET := $(STATICLIBNAME)
PKGTARGET  := $(PKGCONFIGFILE)
LIBOBJ     := $(LIBSRCFILES:%.cc=$(OBJDIR)/%.o)
BINOBJ     := $(BINSRCFILES:%.cc=$(OBJDIR)/%.o)
DEPFILES   := $(LIBOBJ:%.o=%.d) $(BINOBJ:%.o=%.d)
LDFLAGS    += -Wl,--no-undefined
LDFLAGS    += -L${CUDA_LIB} -L${BUILDDIR} -l$(CUDARTLIB) -lpthread -lm -lrt -ldl
LDFLAGS    += ${THIRD_PARTY_LDFLAGS}


ifeq ($(CUDARTLIB), cudarthook)
  # Ensure that cudart symbols are resolved within the module itself.
  LDFLAGS += -Wl,-Bsymbolic
endif

INCPLUGIN  := include/plugin

DEVMANIFEST := $(BUILDDIR)/obj/device/manifest

##### rules
build : lib staticlib binary

lib : $(INCTARGETS) $(LIBDIR)/$(LIBTARGET) $(PKGDIR)/$(PKGTARGET)

staticlib : $(LIBDIR)/$(STATICLIBTARGET)

binary : $(BINDIR)/$(BINNAME)

$(DEVMANIFEST): ALWAYS_REBUILD $(INCTARGETS)
	$(MAKE) -C ./device CONDA_INCLUDE_DIR=$(CONDA_INCLUDE_DIR)

# Empty target to force rebuild
ALWAYS_REBUILD:

-include $(DEPFILES)
$(LIBDIR)/$(LIBTARGET) $(LIBDIR)/$(STATICLIBTARGET) : $(LIBOBJ)

$(INCDIR)/nccl.h : nccl.h.in ../makefiles/version.mk
# NCCL_VERSION(X,Y,Z) ((X) * 10000 + (Y) * 100 + (Z))
	@$(eval NCCL_VERSION := $(shell printf "%d%02d%02d" $(NCCL_MAJOR) $(NCCL_MINOR) $(NCCL_PATCH)))
	mkdir -p $(INCDIR)
	@printf "Generating %-35s > %s\n" $< $@
	sed -e "s/\$${nccl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${nccl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${nccl:Patch}/$(NCCL_PATCH)/g" \
	    -e "s/\$${nccl:Suffix}/$(NCCL_SUFFIX)/g" \
	    -e "s/\$${nccl:Version}/$(NCCL_VERSION)/g" \
	    $< > $@

$(LIBDIR)/$(LIBTARGET): $(LIBOBJ) $(DEVMANIFEST)
	@printf "Linking    %-35s > %s\n" $(LIBTARGET) $@
	mkdir -p $(LIBDIR)
	$(CXX) $(CXXFLAGS) -shared -Wl,--no-as-needed -Wl,-soname,$(LIBSONAME) -o $@ $(LIBOBJ) $$(cat $(DEVMANIFEST)) $(LDFLAGS)
	ln -sf $(LIBSONAME) $(LIBDIR)/$(LIBNAME)
	ln -sf $(LIBTARGET) $(LIBDIR)/$(LIBSONAME)

$(LIBDIR)/$(STATICLIBTARGET): $(LIBOBJ) $(DEVMANIFEST)
	@printf "Archiving  %-35s > %s\n" $(STATICLIBTARGET) $@
	mkdir -p $(LIBDIR)
	ar cr $@ $(LIBOBJ) $$(cat $(DEVMANIFEST))

$(BINDIR)/$(BINNAME): $(BINOBJ)
	@printf "Linking    %-35s > %s\n" $(BINNAME) $@
	mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(PKGDIR)/nccl.pc : nccl.pc.in
	mkdir -p $(PKGDIR)
	@printf "Generating %-35s > %s\n" $< $@
	sed -e 's|$${nccl:Prefix}|\$(PREFIX)|g' \
	    -e "s/\$${nccl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${nccl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${nccl:Patch}/$(NCCL_PATCH)/g" \
	    $< > $@

$(INCDIR)/%.h : %.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)
	install -m 644 $< $@

$(INCDIR)/nccl_%.h : include/nccl_%.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)
	install -m 644 $< $@

$(PKGDIR)/%.pc : %.pc
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(PKGDIR)
	install -m 644 $< $@

$(OBJDIR)/%.o : %.cc $(INCTARGETS)
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(CXX) -I. -I$(INCDIR) $(CXXFLAGS) $(INCLUDES) -I$(INCPLUGIN) -c $< -o $@
	@$(CXX) -I. -I$(INCDIR) $(CXXFLAGS) $(INCLUDES) -I$(INCPLUGIN) -M $< > $(@:%.o=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%.o=%.d.tmp) > $(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%.o=%.d.tmp) | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%.o=%.d)
	@rm -f $(@:%.o=%.d.tmp)

clean :
	$(MAKE) -C device clean
	rm -rf ${BINDIR} ${INCDIR} ${LIBDIR} ${PKGDIR} ${OBJDIR}

install : build
	mkdir -p $(PREFIX)/lib
	mkdir -p $(PREFIX)/lib/pkgconfig
	mkdir -p $(PREFIX)/include
	mkdir -p $(PREFIX)/bin
	cp -P -v $(BUILDDIR)/lib/lib* $(PREFIX)/lib/
	cp -P -v $(BUILDDIR)/lib/pkgconfig/* $(PREFIX)/lib/pkgconfig/
	cp -v $(BUILDDIR)/include/* $(PREFIX)/include/
	cp -v $(BUILDDIR)/bin/ncclras $(PREFIX)/bin/

FILESTOFORMAT := $(shell find . -name ".\#*" -prune -o \( -name "*.cc" -o -name "*.h" \) -print | grep -v -E 'ibvwrap.h|nvmlwrap.h|gdrwrap.h|nccl.h')
# Note that formatting.mk defines a new target so in order to not overwrite the default target,
# it shouldn't be included at the top. Also, it uses the above definition of FILESTOFORMAT as well
# as the BUILDDIR variable.
include ../makefiles/formatting.mk
