lib:
	$(FC) -c $(FCMODFLAGS) $(FCPREFLAGS) -fPIC $(FCCUDAFLAGS) $(FCFLAGS) src/cudafor.f90 -o $(OBJDIR)/cudafor.o
	$(FC) -shared -Wl,--no-as-needed -Wl,-soname,libcudafor.so $(OBJDIR)/cudafor.o -o $(LIBDIR)/libcudafor.so

