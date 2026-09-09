/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Keep this translation unit limited to NCCL's installed public surface. In
// particular, do not reach into src/include for private implementation headers.
#include <nccl.h>
#include <nccl_device.h>
