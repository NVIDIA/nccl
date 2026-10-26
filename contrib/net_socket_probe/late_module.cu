__device__ int module_data[1024] = {7};
extern "C" __global__ void late_kernel(int* out) { out[0] = module_data[0]; }
