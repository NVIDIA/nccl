/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Minimal NetworkDirect status definitions needed by NCCL's NDv2 wrapper.
#pragma once

#ifndef _NDSTATUS_H_
#define _NDSTATUS_H_

#include <winerror.h>

#ifndef ND_SUCCESS
#define ND_SUCCESS S_OK
#endif

#ifndef ND_PENDING
#define ND_PENDING ((HRESULT)0x00000103L)
#endif

#ifndef ND_BUFFER_OVERFLOW
#define ND_BUFFER_OVERFLOW ((HRESULT)0x80000005L)
#endif

#ifndef ND_CANCELED
#define ND_CANCELED ((HRESULT)0xC0000120L)
#endif

#ifndef ND_TIMEOUT
#define ND_TIMEOUT ((HRESULT)0x00000102L)
#endif

#ifndef ND_IO_TIMEOUT
#define ND_IO_TIMEOUT ((HRESULT)0xC00000B5L)
#endif

#endif // _NDSTATUS_H_
