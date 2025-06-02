# Eugo Notes
This is a CMake specific port of NVIDIA's NCCL library. We exclusively leverage Clang for both host and device compilation.

```bash
bash-5.2# ncclras --version
NCCL RAS client version 2.16.2
```

# NCCL

Optimized primitives for inter-GPU communication.

This fork enables building NCCL with CMake, Clang for both host and device, and using system nvtx instead of the bundled NVTX, and is based on the original NCCL repository. It is designed to be compatible with the original NCCL library, while providing additional build features and improvements.

## Introduction

NCCL (pronounced "Nickel") is a stand-alone library of standard communication routines for GPUs, implementing all-reduce, all-gather, reduce, broadcast, reduce-scatter, as well as any send/receive based communication pattern. It has been optimized to achieve high bandwidth on platforms using PCIe, NVLink, NVswitch, as well as networking using InfiniBand Verbs or TCP/IP sockets. NCCL supports an arbitrary number of GPUs installed in a single node or across multiple nodes, and can be used in either single- or multi-process (e.g., MPI) applications.

For more information on NCCL usage, please refer to the [NCCL documentation](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html).

More NCCL resources:
• https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html#downloadnccl
• https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
• https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#host-compiler-support-policy


## Inspiration

The inspiration for this cmake port came from (thank you @cyyever):
• https://github.com/NVIDIA/nccl/pull/664/files
• https://github.com/cyyever/nccl/tree/cmake
• https://github.com/NVIDIA/nccl/issues/1287

The xla patch for clang came from:
• https://github.dev/openxla/xla/blob/main/third_party/nccl/archive.patch


## Original NCCL repo resources:
• `export NVCC_APPEND_FLAGS='-allow-unsupported-compiler'`: https://github.com/NVIDIA/cuda-samples/issues/179
• Provenance of NVTX headers in NCCL: https://github.com/NVIDIA/nccl/issues/1270


## Clang specific CUDA resources:
• https://github.com/llvm/llvm-project/blob/main/clang/lib/Basic/Targets/NVPTX.cpp
• https://llvm.org/docs/LangRef.html#:~:text=x%3A%20Invalid.-,NVPTX,-%3A
• https://llvm.org/docs/CompileCudaWithLLVM.html


## NVCC Identification Macros
• https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#compilation-phases



~~NVTX3 finding - you can change to use find_package(CUDA) and extract that as cuda::nvtx3  https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html~~
~~Remove nvtx3 from their include folder~~
~~Update sources in src/CMakeLists.txt [CHECK W/ SLAV]~~




2. Options: (./__io__/tmp/eugo/nccl-1786e871814edc7f76f463980c283f9e04681256/)
        1. ~~Fork `nvidia/NCCL` and insert CMakeFiles introduced by fork with our changes~~
        2. ~~[DONE? -> DOUBLE CHECK W/ SLAV] In `src/device/CMakeLists.txt`, do the below (see cmake script below)~~
        3. ~~We need to manually go recursively through all CMakeLists.txt and add all new files into appropriate targets~~
            1. ~~Is it best to dynamically generate the list of files from cmake? i think we need /graph, /transport, /register, /misc and exclude /include, /device, /plugin, /ras?~~
        5. ~~Properly add options to disable/enable static and shared libraries~~
        ~~4. Make nccl.h to be generated in build and not the source tree as we did it for colldevice - but add that into target_include_directories of colldevice. @TODO: add that into other targets target_include_directories~~
        4. We need to add new CMakeLists.txt for new folders in latest NCCL
        ~~6. [DONE? -> DOUBLE CHECK W/ SLAV] Refine/revise CMake options because current ones suck (i.e., too vague and badly documented) - @TODO: highly likely, just remove most of the options.~~
        7. ~~apply selective `xla` patches?~~
        ~~8. [DONE? -> DOUBLE CHECK W/ SLAV] Dynamically remove `const` from device_table.cu - fixed by appplying xla_clang.patch~~
        ~~15. update any/all standards that need to be updated (i.e. CMAKE_CXX_STANDARD / CMAKE_CUDA_STANDARD)~~
        10. ~~[IN PROGRESS] Generate .pc and/or cmake-config files~~
            ~~1. Note: no need because we are only using nccl in torch and there are cmake variables we can pass so NCCL is found on the system.~~
        ~~11. change `add_subdirectory(collectives/device)` to `add_subdirectory(device)`~~
        ~~9. [DONE? -> DOUBLE CHECK W/ SLAV] Add ability to use system-provided nvtx.  @TODO+:Ben:add link to pytorch approach - we'll adopt (but not as-is) or use something like the below (see cmake script below) 1. Ensure that /src/CMakeLists.txt has correct `target_link_libraries(${lib_name} PRIVATE CUDA::nvtx3)` 2. Research and document the differences between our approach and the PyTorch approach to ensure clarity in implementation.~~
            ~~BEN_TODO+ add to meta.json and info.md that we are not using our system nvtx!!!~~
        ~~12. [IN PROGRESS] Introduce the build target for ncclras (`src/ras` subfolder)~~
            ~~BEN_TODO+: ncclras is built after libnccl and links to that.~~
            ~~1. Builds, but crashes at runtime~~
        ~~13. [DONE? -> DOUBLE CHECK W/ SLAV] Allow specifying of dynamic sm_* architectures~~
        ~~14. [DONE? -> DOUBLE CHECK W/ SLAV] add `-fcuda-rdc` to cuda compilation flags?~~
           ~~1. I see `-x cuda -fgpu-rdc` but not `-fcuda-rdc` so i added it to `CMAKE_CUDA_FLAGS`.~~
        ~~15. Remove `cmake/common.cmake`?~~
        ~~16. ncclras bug~~

        TODO:
        17. Write down the eugo_maintenance guide about syncing with upstream
            1. .cu, .cc, in device - add new extensions
            2. all new files, targets, and libs
        18. CMAKE_UNITY builds
        19. Ensure no ptx emission to reduce binary size and accelerate its production




# To generate .cu source files using an external Python script in CMake with a temporary build-directory outside the source tree, follow this structured approach:

# ⸻

# ✅ Step-by-step CMake Setup

# 1. Create a dedicated temporary directory

# Use file(MAKE_DIRECTORY ...) to create a temporary directory in the build tree:

# set(GENERATED_DIR "${CMAKE_BINARY_DIR}/generated_cuda")
# file(MAKE_DIRECTORY "${GENERATED_DIR}")

# 2. Invoke the Python script

# Use add_custom_command() to invoke your external script and generate .cu files:

# add_custom_command(
#   OUTPUT "${GENERATED_DIR}/stamp"
#   COMMAND ${CMAKE_COMMAND} -E echo "Generating CUDA files..."
#   COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/scripts/generate_cuda.py" "${GENERATED_DIR}"
#   COMMAND ${CMAKE_COMMAND} -E touch "${GENERATED_DIR}/stamp"
#   COMMENT "Running CUDA code generator"
#   DEPENDS "${CMAKE_SOURCE_DIR}/scripts/generate_cuda.py"
# )

# 	•	You generate a stamp file to use as a stable dependency and OUTPUT.
# 	•	${Python3_EXECUTABLE} is assumed to be available via find_package(Python3 COMPONENTS Interpreter REQUIRED)

# 3. Glob .cu files

# After generation, you can safely glob .cu files from the GENERATED_DIR:

# file(GLOB GENERATED_CUDA_SOURCES "${GENERATED_DIR}/*.cu")

# 4. Add a library or executable

# Ensure your target depends on the generation step:

# add_library(my_cuda_lib STATIC ${GENERATED_CUDA_SOURCES})
# add_dependencies(my_cuda_lib generate_cuda_code)

# Note: You should wrap the add_custom_command() inside a add_custom_target(generate_cuda_code ...) to make it an actual build step:

# add_custom_target(generate_cuda_code
#   DEPENDS "${GENERATED_DIR}/stamp"
# )


# ⸻

# 🔁 Full Example

# cmake_minimum_required(VERSION 3.18)
# project(cuda_codegen LANGUAGES CXX CUDA)

# find_package(Python3 COMPONENTS Interpreter REQUIRED)

# set(GENERATED_DIR "${CMAKE_BINARY_DIR}/generated_cuda")
# file(MAKE_DIRECTORY "${GENERATED_DIR}")

# add_custom_command(
#   OUTPUT "${GENERATED_DIR}/stamp"
#   COMMAND ${CMAKE_COMMAND} -E echo "Generating CUDA files..."
#   COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/scripts/generate_cuda.py" "${GENERATED_DIR}"
#   COMMAND ${CMAKE_COMMAND} -E touch "${GENERATED_DIR}/stamp"
#   COMMENT "Running CUDA code generator"
#   DEPENDS "${CMAKE_SOURCE_DIR}/scripts/generate_cuda.py"
# )

# add_custom_target(generate_cuda_code
#   DEPENDS "${GENERATED_DIR}/stamp"
# )

# file(GLOB GENERATED_CUDA_SOURCES "${GENERATED_DIR}/*.cu")

# add_library(my_cuda_lib STATIC ${GENERATED_CUDA_SOURCES})
# add_dependencies(my_cuda_lib generate_cuda_code)


# ⸻

# 🧪 Tips
# 	•	Always use CMAKE_BINARY_DIR or CMAKE_CURRENT_BINARY_DIR to ensure the output is outside your source tree.
# 	•	Avoid file(GLOB ...) for critical build logic in large systems unless you are confident in the directory’s contents being stable across configures. Otherwise, generate a list file from the Python script and read it via file(READ) or include().

# Let me know if you want the Python script to also emit a .cmake file with the list of generated sources.
