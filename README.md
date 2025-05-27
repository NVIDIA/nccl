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




2. Options: (./__io__/tmp/eugo/nccl-1786e871814edc7f76f463980c283f9e04681256/)
        1. ~~Fork `nvidia/NCCL` and insert CMakeFiles introduced by fork with our changes~~
        2. [DONE? -> DOUBLE CHECK W/ SLAV] In `src/device/CMakeLists.txt`, do the below (see cmake script below)
        3. We need to manually go recursively through all CMakeLists.txt and add all new files into appropriate targets
            1. Is it best to dynamically generate the list of files from cmake? i think we need /graph, /transport, /register, /misc and exclude /include, /device, /plugin, /ras?
        4. We need to add new CMakeLists.txt for new folders in latest NCCL
        5. Properly add options to disable/enable static and shared libraries
        6. [DONE? -> DOUBLE CHECK W/ SLAV] Refine/revise CMake options because current ones suck (i.e., too vague and badly documented) - @TODO: highly likely, just remove most of the options.
        7. ~~apply selective `xla` patches?~~
        8. [DONE? -> DOUBLE CHECK W/ SLAV] Dynamically remove `const` from device_table.cu
        9. [DONE? -> DOUBLE CHECK W/ SLAV] Add ability to use system-provided nvtx.  @TODO+:Ben:add link to pytorch approach - we'll adopt (but not as-is) or use something like the below (see cmake script below)
            1. Ensure that /src/CMakeLists.txt has correct `target_link_libraries(${lib_name} PRIVATE CUDA::nvtx3)`
            2. Research and document the differences between our approach and the PyTorch approach to ensure clarity in implementation.
        10. ~~[IN PROGRESS] Generate .pc and/or cmake-config files~~
            1. Note: no need because we are only using nccl in torch and there are cmake variables we can pass so NCCL is found on the system.
        11. ~~change `add_subdirectory(collectives/device)` to `add_subdirectory(device)`~~
        12. [IN PROGRESS] Introduce the build target for ncclras (`src/ras` subfolder)
        13. Allow specifying of dynamic sm_* architectures
        14. [DONE? -> DOUBLE CHECK W/ SLAV] Add `-Wnvcc-compat` and add `-fcuda-rdc` to cuda compilation flags?
        15. update any/all standards that need to be updated (i.e. CMAKE_CXX_STANDARD / CMAKE_CUDA_STANDARD)

        2. [EXTENDED].
            1. We need to run generate.py into `./__eugo_tmp`
            2. List its contents and append them into `CU_FILES` list variable
            3. Copy all files from `./__eugo_tmp` into `./` (==`src/device/`)
            4. Remove `./__eugo_tmp`
            ```
            # Step 1: Run generate.py into ./__eugo_tmp
            execute_process(
            COMMAND ${CMAKE_COMMAND} -E make_directory __eugo_tmp
            )

            execute_process(
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate.py
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/__eugo_tmp
            )

            # Step 2: Get list of generated files
            file(GLOB GENERATED_CU_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/__eugo_tmp "${CMAKE_CURRENT_SOURCE_DIR}/__eugo_tmp/*.cu")

            # Step 3: Append to CU_FILES
            set(CU_FILES "")
            foreach(file ${GENERATED_CU_FILES})
            list(APPEND CU_FILES "${CMAKE_CURRENT_SOURCE_DIR}/__eugo_tmp/${file}")
            endforeach()

            # Step 4: Copy files into src/device/
            foreach(file ${GENERATED_CU_FILES})
            file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/__eugo_tmp/${file}"
                DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}")
            endforeach()

            # Step 5: Remove __eugo_tmp
            file(REMOVE_RECURSE "${CMAKE_CURRENT_SOURCE_DIR}/__eugo_tmp")
            ```

        9. [EXTENDED].```cmake
                # Require NVTX3 from a config-based install
                find_package(nvtx3 CONFIG REQUIRED)

                add_executable(test_nvtx3 main.cpp)

                # Link against the C++ NVTX interface
                target_link_libraries(test_nvtx3 PRIVATE nvtx3::nvtx3-cpp)
            ```