# NCCL CUDA Tests CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(nccl_cuda_tests CUDA CXX)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Shared memory test
add_executable(test_shared_memory test_shared_memory.cu)
target_link_libraries(test_shared_memory PRIVATE CUDA::cuda_driver CUDA::cudart)

# Enable separable compilation for complex kernels
set_target_properties(test_shared_memory PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Set architectures (adjust as needed for your GPU)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
endif()

# Add compile definitions for debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(test_shared_memory PRIVATE DEBUG=1)
endif()

# Enable testing
enable_testing()
add_test(NAME SharedMemoryTest COMMAND test_shared_memory)
