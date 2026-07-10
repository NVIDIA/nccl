*************
Data Pointers
*************

In general, NCCL will accept any CUDA pointers that are accessible from the CUDA device associated to the communicator object. This includes:

 * device memory local to the CUDA device
 * host memory registered using CUDA SDK APIs cudaHostRegister or cudaGetDevicePointer
 * managed and unified memory

The only exception is device memory located on another device but accessible from the current device using peer access. NCCL will return an error in that case to avoid programming errors (only when NCCL_CHECK_POINTERS=1 since 2.2.12).

