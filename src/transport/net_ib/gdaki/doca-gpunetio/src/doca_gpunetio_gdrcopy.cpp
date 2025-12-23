/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file doca_gpu_gdrcopy.h
 * @brief Implementation of the GDRCopy APIs used in doca_gpunetio
 */

#include <dlfcn.h>
#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include "doca_gpunetio_gdrcopy.h"
#include "doca_gpunetio_log.hpp"

struct gdr;
typedef struct gdr *gdr_t;
typedef struct gdr_mh_s {
    unsigned long h;
} gdr_mh_t;

#define GPU_PAGE_SHIFT 16
#define GPU_PAGE_SIZE (1UL << GPU_PAGE_SHIFT)
#define GPU_PAGE_OFFSET (GPU_PAGE_SIZE - 1)
#define GPU_PAGE_MASK (~GPU_PAGE_OFFSET)

#ifdef __GNUC__
#define TYPEOF(x) __typeof__(x)
#else
#define TYPEOF(x) decltype(x)
#endif

#define DOCA_GPUNETIO_GDRCOPY_LIB_NAME "libgdrapi.so.2"
#define DOCA_GPUNETIO_GDRCOPY_LOAD_SYM(handle, symbol, funcptr, on_error_status, on_error_out) \
    do {                                                                                       \
        funcptr = (TYPEOF(funcptr))dlsym(handle, symbol);                                      \
        if (!funcptr) {                                                                        \
            DOCA_LOG(LOG_ERR, "Failed to load symbol %s", symbol);                             \
            on_error_status = ENOENT;                                                          \
            goto on_error_out;                                                                 \
        }                                                                                      \
    } while (0)

struct doca_gpu_gdrcopy_function_table {
    void *handle;
    gdr_t (*open)();
    int (*close)(gdr_t g);
    int (*pin_buffer)(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token,
                      uint32_t va_space, gdr_mh_t *handle);
    int (*unpin_buffer)(gdr_t g, gdr_mh_t handle);
    int (*map)(gdr_t g, gdr_mh_t handle, void **va, size_t size);
    int (*unmap)(gdr_t g, gdr_mh_t handle, void *va, size_t size);
    int (*copy_from_mapping)(gdr_mh_t handle, void *h_ptr, const void *map_d_ptr, size_t size);
    int (*copy_to_mapping)(gdr_mh_t handle, const void *map_d_ptr, void *h_ptr, size_t size);
    void (*runtime_get_version)(int *major, int *minor);
    int (*driver_get_version)(gdr_t g, int *major, int *minor);
};

static struct doca_gpu_gdrcopy_function_table *doca_gpu_gdrcopy_ftable = NULL;
static gdr_t doca_gpu_gdr = NULL;

static int doca_gpu_gdrcopy_ftable_init(struct doca_gpu_gdrcopy_function_table **ftable) {
    int status = 0;
    void *handle = NULL;
    struct doca_gpu_gdrcopy_function_table *table = NULL;

    handle = dlopen(DOCA_GPUNETIO_GDRCOPY_LIB_NAME, RTLD_NOW);
    if (!handle) {
        DOCA_LOG(LOG_ERR, "Failed to open libgdrapi.so.2");
        status = ENOENT;
        goto out;
    }

    table = (struct doca_gpu_gdrcopy_function_table *)malloc(
        sizeof(struct doca_gpu_gdrcopy_function_table));
    if (!table) {
        DOCA_LOG(LOG_ERR, "Failed to allocate memory for gdrcopy function table");
        status = ENOMEM;
        goto out;
    }

    DOCA_GPUNETIO_GDRCOPY_LOAD_SYM(handle, "gdr_open", table->open, status, out);
    DOCA_GPUNETIO_GDRCOPY_LOAD_SYM(handle, "gdr_close", table->close, status, out);
    DOCA_GPUNETIO_GDRCOPY_LOAD_SYM(handle, "gdr_pin_buffer", table->pin_buffer, status, out);
    DOCA_GPUNETIO_GDRCOPY_LOAD_SYM(handle, "gdr_unpin_buffer", table->unpin_buffer, status, out);
    DOCA_GPUNETIO_GDRCOPY_LOAD_SYM(handle, "gdr_map", table->map, status, out);
    DOCA_GPUNETIO_GDRCOPY_LOAD_SYM(handle, "gdr_unmap", table->unmap, status, out);
    DOCA_GPUNETIO_GDRCOPY_LOAD_SYM(handle, "gdr_copy_from_mapping", table->copy_from_mapping,
                                   status, out);
    DOCA_GPUNETIO_GDRCOPY_LOAD_SYM(handle, "gdr_copy_to_mapping", table->copy_to_mapping, status,
                                   out);
    DOCA_GPUNETIO_GDRCOPY_LOAD_SYM(handle, "gdr_runtime_get_version", table->runtime_get_version,
                                   status, out);
    DOCA_GPUNETIO_GDRCOPY_LOAD_SYM(handle, "gdr_driver_get_version", table->driver_get_version,
                                   status, out);

    table->handle = handle;
    *ftable = table;

out:
    if (status != 0) {
        if (handle) {
            dlclose(handle);
        }
        if (table) {
            free(table);
        }
    }
    return status;
}

static int doca_gpu_init_gdrcopy() {
    int status = 0;
    if (!doca_gpu_gdr) {
        if (!doca_gpu_gdrcopy_ftable) {
            status = doca_gpu_gdrcopy_ftable_init(&doca_gpu_gdrcopy_ftable);
            if (status) {
                DOCA_LOG(LOG_ERR, "Error in doca_gpu_gdrcopy_ftable_init");
                goto out;
            }
        }

        doca_gpu_gdr = doca_gpu_gdrcopy_ftable->open();
        if (!doca_gpu_gdr) {
            DOCA_LOG(LOG_ERR, "Error in gdr_open");
            status = EIO;
            goto out;
        }
    }

out:
    return status;
}

static bool doca_gpu_enable_gdrcopy() {
    const char *env = getenv("DOCA_GPUNETIO_DISABLE_GDRCOPY");
    if (env && atoi(env) != 0) {
        DOCA_LOG(LOG_INFO, "DOCA_GPUNETIO_DISABLE_GDRCOPY is set, disabling GDRCopy");
        return false;
    }
    return true;
}

bool doca_gpu_gdrcopy_is_supported() {
    static bool is_tried_init = false;
    static bool is_supported = false;
    if (!is_tried_init) {
        bool enabled = doca_gpu_enable_gdrcopy();
        is_supported = (enabled && (doca_gpu_init_gdrcopy() == 0));
        DOCA_LOG(LOG_INFO, "GDRCopy usage is %s", is_supported ? "enabled" : "disabled");
        is_tried_init = true;
    }
    return is_supported;
}

static int priv_doca_gpu_gdrcopy_create_mapping(void *dev_aligned_ptr, size_t size,
                                                gdr_mh_t *out_mh, void **out_host_ptr) {
    int status = 0;
    gdr_mh_t mh;
    void *host_ptr;
    bool did_gdr_pin_buffer = false;

    status = doca_gpu_init_gdrcopy();
    if (status) {
        DOCA_LOG(LOG_ERR, "Error in doca_gpu_init_gdrcopy");
        goto out;
    }

    assert(((uintptr_t)dev_aligned_ptr & (GPU_PAGE_SIZE - 1ULL)) == 0);

    status = doca_gpu_gdrcopy_ftable->pin_buffer(doca_gpu_gdr, (unsigned long)dev_aligned_ptr, size,
                                                 0, 0, &mh);
    if (status) {
        DOCA_LOG(LOG_ERR, "Error in gdr_pin_buffer");
        goto out;
    }
    did_gdr_pin_buffer = true;

    status = doca_gpu_gdrcopy_ftable->map(doca_gpu_gdr, mh, &host_ptr, size);
    if (status) {
        DOCA_LOG(LOG_ERR, "Error in gdr_map");
        goto out;
    }

    *out_mh = mh;
    *out_host_ptr = host_ptr;

out:
    if (status) {
        if (did_gdr_pin_buffer) doca_gpu_gdrcopy_ftable->unpin_buffer(doca_gpu_gdr, mh);
    }
    return status;
}

int doca_gpu_gdrcopy_create_mapping(void *dev_aligned_ptr, size_t size, void **out_mh,
                                    void **out_host_ptr) {
    int status = 0;
    gdr_mh_t *mh = NULL;
    mh = (gdr_mh_t *)malloc(sizeof(gdr_mh_t));
    if (!mh) {
        DOCA_LOG(LOG_ERR, "Error in malloc for mh");
        status = ENOMEM;
        goto out;
    }

    status = priv_doca_gpu_gdrcopy_create_mapping(dev_aligned_ptr, size, mh, out_host_ptr);
    if (status) {
        DOCA_LOG(LOG_ERR, "Error in priv_doca_gpu_gdrcopy_create_mapping");
        goto out;
    }

    *out_mh = mh;

out:
    if (status) {
        if (mh) {
            free(mh);
        }
    }
    return status;
}

static void _doca_gpu_gdrcopy_destroy_mapping(gdr_mh_t *mh, void *host_ptr, size_t size) {
    assert(doca_gpu_gdr);
    doca_gpu_gdrcopy_ftable->unmap(doca_gpu_gdr, *mh, host_ptr, size);
    doca_gpu_gdrcopy_ftable->unpin_buffer(doca_gpu_gdr, *mh);
}

void doca_gpu_gdrcopy_destroy_mapping(void *mh, void *host_ptr, size_t size) {
    if (mh) {
        _doca_gpu_gdrcopy_destroy_mapping((gdr_mh_t *)mh, host_ptr, size);
        free(mh);
    }
}
