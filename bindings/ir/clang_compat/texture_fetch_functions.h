/* Empty stub. Clang's __clang_cuda_runtime_wrapper.h includes this header
 * unconditionally, but CUDA 13 removed it. NCCL device code does not use
 * texture fetches, so an empty stub is sufficient. For CUDA 12 the real
 * header in $(CUDA_INC) is found first via -I ordering and this stub is
 * unused.
 */
