FC := gfortran

LIBNAME   := libncclfor.so
LIBSONAME := $(patsubst %,%.$(NCCL_MAJOR),$(LIBNAME))
LIBTARGET := $(patsubst %,%.$(NCCL_MAJOR).$(NCCL_MINOR).$(NCCL_PATCH),$(LIBNAME))
LIBLINK   += $(patsubst lib%.so,-l%,$(LIBNAME))

ifeq ($(FC), gfortran)
FCMODFLAGS  := -J$(INCDIR)
FCPREFLAGS  := -cpp
FCCUDAFLAGS :=
FCFLAGS     := -ffree-line-length-none
CUDAFORDEP  := libcudafor
CUDALINK    := -L$(CUDA_LIB) -lcudart
CUDAFORLINK := -lcudafor
else ifeq ($(FC), ifort)
FCMODFLAGS  := -module $(INCDIR)
FCPREFLAGS  := -fpp
FCCUDAFLAGS :=
FCFLAGS     :=
CUDAFORDEP  := libcudafor
CUDALINK    := -L$(CUDA_LIB) -lcudart
CUDAFORLINK := -lcudafor
else
FCMODFLAGS  := -module $(INCDIR)
FCPREFLAGS  := -Mpreprocess
FCCUDAFLAGS := -Mcuda,cuda$(CUDA_MAJOR).$(CUDA_MINOR)
FCFLAGS     := -fast -O3
CUDAFORDEP  :=
CUDALINK    :=
CUDAFORLINK :=
endif

lib: $(CUDAFORDEP)
	$(FC) -c $(FCMODFLAGS) $(FCPREFLAGS) -fPIC $(FCCUDAFLAGS) $(FCFLAGS) src/ncclfor.f90 -o $(OBJDIR)/ncclfor.o
	$(FC) -shared -Wl,--no-as-needed -Wl,-soname,$(LIBSONAME) $(OBJDIR)/ncclfor.o -o $(LIBDIR)/$(LIBTARGET)
	ln -sf $(LIBSONAME) $(LIBDIR)/$(LIBNAME)
	ln -sf $(LIBTARGET) $(LIBDIR)/$(LIBSONAME)

export

libcudafor:
	$(MAKE) -C cudafor lib

TESTS := reduce_ptr_out allreduce_ptr_out reducescatter_ptr_out broadcast_ptr allgather_ptr_out
ifneq ($(FC), gfortran)
ifneq ($(FC), ifort)
TESTS += reduce_arr_out allreduce_arr_out reducescatter_arr_out broadcast_arr allgather_arr_out
endif
endif

TESTDIR  := $(BUILDDIR)/test/fortran
TESTBINS := $(patsubst %,$(TESTDIR)/%,$(TESTS))

test: lib $(TESTBINS)

$(TESTDIR)/%: test/%.f90
	@mkdir -p $(TESTDIR)
	$(FC) $(FCCUDAFLAGS) $(FCFLAGS) $< $(CUDALINK) -I$(INCDIR) -L$(LIBDIR) $(CUDAFORLINK) $(LIBLINK) -o $@

clean:
	rm -rf $(INCDIR)/*.mod
	rm -rf $(LIBDIR)/lib*for.so*
	rm -rf $(OBJDIR)/*for.o
	rm -rf $(TESTDIR)/

