# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
# NetworkDirect plugin sources (Windows only)
set(NET_ND_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/ndwrap.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/common.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/init.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/connect.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reg.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/p2p.cc
)

# Add to parent transport sources
set(TRANSPORT_SOURCES ${TRANSPORT_SOURCES} ${NET_ND_SOURCES} PARENT_SCOPE)
