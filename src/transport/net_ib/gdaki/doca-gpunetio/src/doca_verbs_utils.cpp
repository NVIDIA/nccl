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

#include <unistd.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "common/doca_gpunetio_verbs_def.h"
#include "host/doca_gpunetio.h"
#include "host/doca_verbs.h"
#include "doca_internal.hpp"
#include "doca_verbs_net_wrapper.h"
#include "doca_verbs_dev.hpp"
#include "doca_verbs_dev_sdk_wrapper.h"

static bool parse_global_traffic_class(const char *line, uint8_t *traffic_class) {
    static constexpr char prefix[] = "Global tclass=";
    if (line == nullptr || strncmp(line, prefix, sizeof(prefix) - 1) != 0) return false;

    const char *value_begin = line + sizeof(prefix) - 1;
    char *value_end{};
    errno = 0;
    long parsed_value = strtol(value_begin, &value_end, 10);
    if (value_end == value_begin || errno == ERANGE) return false;

    while (*value_end != '\0' && isspace(static_cast<unsigned char>(*value_end))) ++value_end;

    if (*value_end != '\0' || parsed_value < 0 || parsed_value > UINT8_MAX) return false;

    *traffic_class = static_cast<uint8_t>(parsed_value);
    return true;
}

doca_error_t doca_verbs_query_global_traffic_class(doca_dev_t *net_dev, uint16_t port_num,
                                                   uint8_t *global_traffic_class) {
    doca_error_t status = DOCA_SUCCESS;
    const char *device_name;
    char path[PATH_MAX];
    int length = 0;
    FILE *file = nullptr;
    char line[256];
    bool found_global = false;
    uint8_t traffic_class = 0;
    struct ibv_context *context;

    if (net_dev == nullptr || port_num == 0 || global_traffic_class == nullptr) {
        status = DOCA_ERROR_INVALID_VALUE;
        goto out;
    }

    if (net_dev->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        auto err = doca_verbs_sdk_wrapper_dev_get_ibv_ctx(net_dev, &context);
        if (err != DOCA_SDK_WRAPPER_SUCCESS) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            status = DOCA_ERROR_UNEXPECTED;
            goto out;
        }
    } else {
        if (net_dev->open == nullptr) {
            DOCA_LOG(LOG_ERR, "Invalid input parameters.");
            status = DOCA_ERROR_INVALID_VALUE;
            goto out;
        }

        context = net_dev->open->get_ctx();
    }

    status = doca_verbs_wrapper_ibv_get_device_name(context->device, &device_name);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to get device name");
        goto out;
    }

    length = snprintf(path, sizeof(path), "/sys/class/infiniband/%s/tc/%u/traffic_class",
                      device_name, static_cast<unsigned>(port_num));
    if (length < 0 || static_cast<size_t>(length) >= sizeof(path)) {
        DOCA_LOG(LOG_ERR, "Failed to construct path to traffic class configuration");
        status = DOCA_ERROR_DRIVER;
        goto out;
    };

    file = fopen(path, "r");
    if (file == nullptr) {
        const int open_error = errno;
        if (open_error == EACCES || open_error == EPERM) {
            DOCA_LOG(LOG_DEBUG, "Permission denied to open traffic class configuration %s", path);
            status = DOCA_ERROR_NOT_PERMITTED;
        } else if (open_error == ENOENT) {
            DOCA_LOG(LOG_DEBUG, "Unable to locate traffic class configuration %s", path);
            status = DOCA_ERROR_NOT_FOUND;
        } else {
            DOCA_LOG(LOG_ERR, "Failed to open traffic class configuration %s: %s", path,
                     strerror(open_error));
            status = DOCA_ERROR_OPERATING_SYSTEM;
        }
        goto out;
    }

    while (fgets(line, sizeof(line), file) != nullptr) {
        if (parse_global_traffic_class(line, &traffic_class)) {
            found_global = true;
            break;
        }
    }

    if (found_global) {
        *global_traffic_class = traffic_class;
    } else {
        DOCA_LOG(LOG_DEBUG, "Global traffic class not found in %s", path);
        status = DOCA_ERROR_NOT_FOUND;
    }

out:
    if (file != nullptr) fclose(file);
    return status;
}
