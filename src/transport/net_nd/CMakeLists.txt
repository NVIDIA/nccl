# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

# Compile the provider layer without exposing an incomplete transport to NCCL.
add_library(nccl_net_nd_compile OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/ndwrap.cc
)
target_include_directories(nccl_net_nd_compile PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/device
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_SOURCE_DIR}/src/include/plugin
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}/cccl
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_BINARY_DIR}/obj/include
)
