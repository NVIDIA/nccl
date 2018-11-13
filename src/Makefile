#
# Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

include ../makefiles/common.mk
include ../makefiles/version.mk

##### src files
INCEXPORTS  := nccl.h nccl_net.h
LIBSRCFILES := init.cu ring.cu bootstrap.cu transport.cu misc/group.cu \
		misc/nvmlwrap.cu misc/ibvwrap.cu misc/rings.cu misc/utils.cu misc/enqueue.cu \
		transport/p2p.cu transport/shm.cu transport/net.cu transport/net_socket.cu transport/net_ib.cu \
                collectives/all_reduce.cu collectives/all_gather.cu collectives/broadcast.cu collectives/reduce.cu collectives/reduce_scatter.cu

##### lib files
LIBNAME     := libnccl.so
STATICLIBNAME := libnccl_static.a
##### dirs
BUILDDIR ?= $(abspath ../build)
INCDIR := $(BUILDDIR)/include
LIBDIR := $(BUILDDIR)/lib
OBJDIR := $(BUILDDIR)/obj
##### target files
INCTARGETS := $(INCEXPORTS:%=$(INCDIR)/%)
LIBSONAME  := $(LIBNAME:%=%.$(NCCL_MAJOR))
LIBTARGET  := $(LIBNAME:%=%.$(NCCL_MAJOR).$(NCCL_MINOR).$(NCCL_PATCH))
STATICLIBTARGET := $(STATICLIBNAME)
LIBOBJ     := $(LIBSRCFILES:%.cu=$(OBJDIR)/%.o)
DEPFILES   := $(LIBOBJ:%.o=%.d)
LDFLAGS    += -L${CUDA_LIB} -lcudart_static -lrt

DEVICELIB  := $(BUILDDIR)/obj/collectives/device/colldevice.a


##### rules
build : lib staticlib

lib : $(INCTARGETS) $(LIBDIR)/$(LIBTARGET)

staticlib : $(LIBDIR)/$(STATICLIBTARGET)

devicelib: $(INCDIR)/nccl.h
	$(MAKE) -C collectives/device

-include $(DEPFILES)
$(LIBDIR)/$(LIBTARGET) $(LIBDIR)/$(STATICLIBTARGET) : $(LIBOBJ)

$(INCDIR)/nccl.h : nccl.h.in
# NCCL_VERSION(X,Y,Z) ((X) * 1000 + (Y) * 100 + (Z))
	@$(eval NCCL_VERSION := $(shell printf "%d%d%02d" $(NCCL_MAJOR) $(NCCL_MINOR) $(NCCL_PATCH)))
	mkdir -p $(INCDIR)
	printf "Generating %-35s > %s\n" $< $@
	sed -e "s/\$${nccl:Major}/$(NCCL_MAJOR)/g" \
	    -e "s/\$${nccl:Minor}/$(NCCL_MINOR)/g" \
	    -e "s/\$${nccl:Patch}/$(NCCL_PATCH)/g" \
	    -e "s/\$${nccl:Suffix}/$(NCCL_SUFFIX)/g" \
	    -e "s/\$${nccl:Version}/$(NCCL_VERSION)/g" \
	    $< > $@

$(LIBDIR)/$(LIBTARGET): $(LIBOBJ) devicelib
	@printf "Linking    %-35s > %s\n" $(LIBTARGET) $@
	mkdir -p $(LIBDIR)
	$(CXX) $(CXXFLAGS) -shared -Wl,--no-as-needed -Wl,-soname,$(LIBSONAME) -o $@ $(LIBOBJ) $(DEVICELIB) $(LDFLAGS)
	ln -sf $(LIBSONAME) $(LIBDIR)/$(LIBNAME)
	ln -sf $(LIBTARGET) $(LIBDIR)/$(LIBSONAME)

$(LIBDIR)/$(STATICLIBTARGET): $(LIBOBJ) devicelib
	@printf "Archiving  %-35s > %s\n" $(STATICLIBTARGET) $@
	mkdir -p $(LIBDIR)
	$(eval TMP := $(shell mktemp -d))
	cp $(LIBOBJ) $(TMP)
	cd $(TMP) && ar x $(DEVICELIB) && cd -
	ar cr $@ $(LIBOBJ) $(TMP)/*.o
	rm -Rf $(TMP)

$(INCDIR)/%.h : %.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)
	cp -f $< $@

$(INCDIR)/nccl_%.h : include/nccl_%.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)
	cp -f $< $@

$(OBJDIR)/%.o : %.cu
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) -I. -I$(INCDIR) -Iinclude -c $(NVCUFLAGS) --compiler-options "$(CXXFLAGS)" $< -o $@
	@$(NVCC) -I. -I$(INCDIR) -Iinclude -M $(NVCUFLAGS) --compiler-options "$(CXXFLAGS)" $< > $(@:%.o=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%.o=%.d.tmp) > $(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%.o=%.d.tmp) | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%.o=%.d)
	@rm -f $(@:%.o=%.d.tmp)

clean :
	rm -rf ${INCDIR} ${LIBDIR} ${OBJDIR}
	$(MAKE) -C collectives/device clean

install : lib
	mkdir -p $(PREFIX)/lib
	mkdir -p $(PREFIX)/include
	cp -P -v $(BUILDDIR)/lib/* $(PREFIX)/lib/
	cp -v $(BUILDDIR)/include/* $(PREFIX)/include/

FILESTOFORMAT := $(shell find . -name ".\#*" -prune -o \( -name "*.cu" -o -name "*.h" \) -print | grep -v -E 'ibvwrap.h|nvmlwrap.h|nccl.h')
# Note that formatting.mk defines a new target so in order to not overwrite the default target,
# it shouldn't be included at the top. Also, it uses the above definition of FILESTOFORMAT as well
# as the BUILDDIR variable.
include ../makefiles/formatting.mk
