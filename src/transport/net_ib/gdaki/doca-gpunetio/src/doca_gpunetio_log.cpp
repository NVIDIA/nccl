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

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <syslog.h>
#include <time.h>
#include <stdarg.h>

#include "doca_gpunetio_log.hpp"

static const char *doca_gpu_log_level_strings[] = {"EMERG",   "ALERT",  "CRIT", "ERR",
                                                   "WARNING", "NOTICE", "INFO", "DEBUG"};

void doca_gpu_log_print(int log_level, const char *file, int line, const char *func,
                        const char *fmt, ...) {
    static int cur_log_level = -1;
    if (cur_log_level < 0) {
        const char *debug_env = getenv("DOCA_GPUNETIO_LOG");
        if (debug_env != NULL) {
            int env_log_level = atoi(debug_env);
            if (env_log_level >= 0 &&
                env_log_level <= (int)(sizeof(doca_gpu_log_level_strings) /
                                       sizeof(doca_gpu_log_level_strings[0]))) {
                cur_log_level = env_log_level;
            }
        }
        if (cur_log_level < 0) {
            cur_log_level = 0;
        }
    }

    if (log_level <= cur_log_level) {
        time_t now = time(NULL);
        char *timestamp = ctime(&now);
        timestamp[strlen(timestamp) - 1] = '\0';
        va_list args;
        va_start(args, fmt);
        fprintf(stderr, "%s [%s] [%s]: %d: %s(): ", timestamp,
                doca_gpu_log_level_strings[log_level], file, line, func);
        vfprintf(stderr, fmt, args);
        fprintf(stderr, "\n");
        va_end(args);
    }
}
