# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

add_library(nccl_diagnostics_device OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/p2p.cu
)

set_target_properties(nccl_diagnostics_device PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

target_compile_definitions(nccl_diagnostics_device PRIVATE NCCL_OS_LINUX)

target_include_directories(nccl_diagnostics_device PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src/diagnostics
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/device
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_SOURCE_DIR}/src/include/plugin
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}/cccl
)
