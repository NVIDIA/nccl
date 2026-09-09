# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

# Diagnostics sources
add_subdirectory(device)

set(DIAGNOSTICS_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/p2p.cc
)

# Add diagnostics sources to parent scope
set(DIAGNOSTICS_SOURCES ${DIAGNOSTICS_SOURCES} PARENT_SCOPE)
