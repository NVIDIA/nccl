#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

# LLVM IR generation for NCCL device APIs

# Source and output configuration
set(LLVM_SRC ${CMAKE_CURRENT_SOURCE_DIR}/nccl_device_wrapper__impl.h)
set(OBJDIR ${CMAKE_BINARY_DIR}/obj/llvm_ir)
set(LIBDIR ${CMAKE_BINARY_DIR}/lib)
set(INCLUDEDIR ${CMAKE_BINARY_DIR}/include)

# Output files (all aux files in obj/llvm_ir)
set(UNOPTIMIZED_BC ${OBJDIR}/libnccl_device.bc.unoptimized)
set(OPTIMIZED_BC ${OBJDIR}/libnccl_device.bc.optimized)
set(LLVM_IR_FILE ${OBJDIR}/libnccl_device.ll)
set(FINAL_BC ${LIBDIR}/libnccl_device.bc)
set(FINAL_LTO_IR ${LIBDIR}/libnccl_device.ltoir)

set(WRAPPER_HEADER ${INCLUDEDIR}/nccl_device_wrapper.h)

# Build configuration
# Select GPU arch and C++ standard based on CUDA Toolkit version
if(CUDAToolkit_VERSION_MAJOR LESS 12)
    set(BITCODE_LIB_ARCH sm_70 CACHE STRING "CUDA architecture for LLVM IR")
else()
    set(BITCODE_LIB_ARCH sm_90 CACHE STRING "CUDA architecture for LLVM IR")
endif()

set(LTO_IR_ARCH ${BITCODE_LIB_ARCH} CACHE STRING "CUDA architecture for LTO IR")

set(BITCODE_CXX_STD gnu++17 CACHE STRING "C++ standard for LLVM IR")
set(LTO_IR_CXX_STD c++17 CACHE STRING "C++ standard for LTO IR")

# Find required tools
find_program(CLANG_EXECUTABLE clang REQUIRED)
find_program(OPT_EXECUTABLE opt REQUIRED)
find_program(LLVM_DIS_EXECUTABLE llvm-dis REQUIRED)
find_program(LLVM_AS_EXECUTABLE llvm-as REQUIRED)

# Include paths
set(NCCL_INCLUDES
    -I${CMAKE_BINARY_DIR}/include
    -I${CMAKE_SOURCE_DIR}/src/include
    -I${CMAKE_SOURCE_DIR}/src/include/nccl_device
    -I${CMAKE_SOURCE_DIR}/src/device
)

set(CUDA_INCLUDES
    -I${CUDAToolkit_INCLUDE_DIRS}
    -I${CUDAToolkit_INCLUDE_DIRS}/cccl
)

# Empty stubs for headers clang's CUDA wrapper unconditionally includes but
# CUDA 13 removed (e.g. texture_fetch_functions.h). NCCL device code doesn't
# use the legacy texture/surface reference APIs, so it's OK to be empty.
set(CLANG_COMPAT_INCLUDES
    -I${CMAKE_CURRENT_SOURCE_DIR}/clang_compat
)

# Common clang flags used for both preprocessing and IR generation
set(COMMON_CLANG_FLAGS
    -std=${BITCODE_CXX_STD}
    -x cuda
    --cuda-path=${CUDAToolkit_ROOT_DIR}
    --cuda-device-only
    --cuda-gpu-arch=${BITCODE_LIB_ARCH}
    ${CLANG_COMPAT_INCLUDES}
    ${NCCL_INCLUDES}
    ${CUDA_INCLUDES}
    -D__clang_llvm_bitcode_lib__
    -DCUDA_MAJOR=${CUDA_MAJOR}
    -DCUDA_MINOR=${CUDA_MINOR}
    -D_NV_RSQRT_SPECIFIER=
)

# Clang flags for LLVM IR generation
set(CLANG_FLAGS
    -c -emit-llvm -O1
    ${COMMON_CLANG_FLAGS}
)

# Copy the preauthored wrapper header to the build include directory
add_custom_command(
    OUTPUT ${WRAPPER_HEADER}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${INCLUDEDIR}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/nccl_device_wrapper.h ${WRAPPER_HEADER}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/nccl_device_wrapper.h
    COMMENT "Copying nccl_device_wrapper.h to build include..."
    VERBATIM
)

# Generate unoptimized LLVM bitcode
add_custom_command(
    OUTPUT ${UNOPTIMIZED_BC}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OBJDIR}
    COMMAND ${CLANG_EXECUTABLE} ${CLANG_FLAGS} ${LLVM_SRC} -o ${UNOPTIMIZED_BC}
    DEPENDS ${LLVM_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/nccl_device_wrapper.h
    COMMENT "Generating unoptimized LLVM bitcode..."
    VERBATIM
)

# Generate optimized LLVM bitcode
add_custom_command(
    OUTPUT ${OPTIMIZED_BC}
    COMMAND ${OPT_EXECUTABLE}
        --passes=internalize,inline,globaldce
        -internalize-public-api-list=*nccl*
        ${UNOPTIMIZED_BC} -o ${OPTIMIZED_BC}
    DEPENDS ${UNOPTIMIZED_BC}
    COMMENT "Optimizing LLVM bitcode..."
    VERBATIM
)

# Generate LLVM IR text file
# Disassemble the *optimized* bitcode (after internalize/inline/globaldce);
# disassembling the unoptimized one would bypass the entire optimization stage.
add_custom_command(
    OUTPUT ${LLVM_IR_FILE}
    COMMAND ${LLVM_DIS_EXECUTABLE} ${OPTIMIZED_BC} -o ${LLVM_IR_FILE}.tmp
    COMMAND ${CMAKE_COMMAND} -E echo "Cleaning LLVM IR (removing nvvm-reflect-ftz)..."
    COMMAND bash -c "myVar=$$(cat ${LLVM_IR_FILE}.tmp | grep -E '!([0-9]+) = !\\{[^\"]*\"nvvm-reflect-ftz\"' | cut -d ' ' -f 1)\; awk '!/nvvm-reflect-ftz/' ${LLVM_IR_FILE}.tmp | sed \"/^!llvm\\.module\\.flags = /s/$$myVar, //\" > ${LLVM_IR_FILE}.tmp2"
    COMMAND ${CMAKE_COMMAND} -E echo "Cleaning LLVM IR (removing function alignment attributes)..."
    COMMAND bash -c "sed '/^define/s/ align [0-9]\\+//g' ${LLVM_IR_FILE}.tmp2 > ${LLVM_IR_FILE}"
    COMMAND ${CMAKE_COMMAND} -E remove ${LLVM_IR_FILE}.tmp ${LLVM_IR_FILE}.tmp2
    DEPENDS ${OPTIMIZED_BC}
    COMMENT "Generating LLVM IR..."
    VERBATIM
)

# Generate final bitcode from cleaned LLVM IR
add_custom_command(
    OUTPUT ${FINAL_BC}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${LIBDIR}
    COMMAND ${CMAKE_COMMAND} -E echo "Generating final bitcode from cleaned LLVM IR..."
    COMMAND ${LLVM_AS_EXECUTABLE} ${LLVM_IR_FILE} -o ${FINAL_BC}
    DEPENDS ${LLVM_IR_FILE}
    COMMENT "Generating final bitcode..."
    VERBATIM
)

# Custom target for LLVM IR generation
add_custom_target(llvm_ir
    DEPENDS ${FINAL_BC} ${OPTIMIZED_BC} ${WRAPPER_HEADER}
    COMMENT "LLVM IR and bitcode generated successfully"
)

# Print generated files on completion
add_custom_command(TARGET llvm_ir POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "LLVM IR and bitcode generated successfully:"
    COMMAND ${CMAKE_COMMAND} -E echo "  C++ Standard:      ${BITCODE_CXX_STD}"
    COMMAND ${CMAKE_COMMAND} -E echo "  GPU Architecture:  ${BITCODE_LIB_ARCH}"
    COMMAND ${CMAKE_COMMAND} -E echo "  Unoptimized:       ${UNOPTIMIZED_BC}"
    COMMAND ${CMAKE_COMMAND} -E echo "  Optimized:         ${OPTIMIZED_BC}"
    COMMAND ${CMAKE_COMMAND} -E echo "  LLVM IR:           ${LLVM_IR_FILE}"
    COMMAND ${CMAKE_COMMAND} -E echo "  Final BC:          ${FINAL_BC}"
    COMMAND ${CMAKE_COMMAND} -E echo "  Wrapper Header:    ${WRAPPER_HEADER}"
    VERBATIM
)

# LTO IR generation

# Find nvcc
find_program(CUDAToolkit_NVCC_EXECUTABLE nvcc HINTS ${CUDAToolkit_BIN_DIR} REQUIRED)

# Generate LTO IR
add_custom_command(
    OUTPUT ${FINAL_LTO_IR}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${LIBDIR}
    COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
        -std=${LTO_IR_CXX_STD}
        -arch=${LTO_IR_ARCH}
        -dlto --ltoir
        -x cu
        ${NCCL_INCLUDES}
        ${CUDA_INCLUDES}
        -DCUDA_MAJOR=${CUDA_MAJOR}
        -DCUDA_MINOR=${CUDA_MINOR}
        -D__NCCL_DEVICE_LTOIR_LIB__
        -o ${FINAL_LTO_IR}
        ${LLVM_SRC}
    DEPENDS ${LLVM_SRC} ${WRAPPER_HEADER}
    COMMENT "Generating LTO IR..."
    VERBATIM
)

# Custom target for LTO IR generation
add_custom_target(ltoir
    DEPENDS ${FINAL_LTO_IR} ${WRAPPER_HEADER}
    COMMENT "LTO IR generated successfully"
)


add_custom_command(TARGET ltoir POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "LTO IR generated successfully:"
    COMMAND ${CMAKE_COMMAND} -E echo "  C++ Standard:      ${LTO_IR_CXX_STD}"
    COMMAND ${CMAKE_COMMAND} -E echo "  GPU Architecture:  ${LTO_IR_ARCH}"
    COMMAND ${CMAKE_COMMAND} -E echo "  Final LTO IR:      ${FINAL_LTO_IR}"
    COMMAND ${CMAKE_COMMAND} -E echo "  Wrapper Header:    ${WRAPPER_HEADER}"
    VERBATIM
)
