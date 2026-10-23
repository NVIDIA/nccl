#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(01_multiple_devices_single_process/c)
add_subdirectory(02_one_device_per_pthread/c)
if(NCCLExamples_USE_MPI)
    add_subdirectory(03_one_device_per_process_mpi/c)
endif()
