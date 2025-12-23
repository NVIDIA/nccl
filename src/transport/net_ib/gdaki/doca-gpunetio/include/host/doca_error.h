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
 * @file doca_errors.h
 * @brief A header file for the doca_error APIs
 */

#ifndef DOCA_ERROR_H
#define DOCA_ERROR_H

#include <errno.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief DOCA API return codes
 */
typedef enum doca_error {
    DOCA_SUCCESS = 0,                      /**< Success */
    DOCA_ERROR_UNKNOWN = 1,                /**< Unknown error */
    DOCA_ERROR_NOT_PERMITTED = 2,          /**< Operation not permitted */
    DOCA_ERROR_IN_USE = 3,                 /**< Resource already in use */
    DOCA_ERROR_NOT_SUPPORTED = 4,          /**< Operation not supported */
    DOCA_ERROR_AGAIN = 5,                  /**< Resource temporarily unavailable, try again */
    DOCA_ERROR_INVALID_VALUE = 6,          /**< Invalid input */
    DOCA_ERROR_NO_MEMORY = 7,              /**< Memory allocation failure */
    DOCA_ERROR_INITIALIZATION = 8,         /**< Resource initialization failure */
    DOCA_ERROR_TIME_OUT = 9,               /**< Timer expired waiting for resource */
    DOCA_ERROR_SHUTDOWN = 10,              /**< Shut down in process or completed */
    DOCA_ERROR_CONNECTION_RESET = 11,      /**< Connection reset by peer */
    DOCA_ERROR_CONNECTION_ABORTED = 12,    /**< Connection aborted */
    DOCA_ERROR_CONNECTION_INPROGRESS = 13, /**< Connection in progress */
    DOCA_ERROR_NOT_CONNECTED = 14,         /**< Not Connected */
    DOCA_ERROR_NO_LOCK = 15,               /**< Unable to acquire required lock */
    DOCA_ERROR_NOT_FOUND = 16,             /**< Resource Not Found */
    DOCA_ERROR_IO_FAILED = 17,             /**< Input/Output Operation Failed */
    DOCA_ERROR_BAD_STATE = 18,             /**< Bad State */
    DOCA_ERROR_UNSUPPORTED_VERSION = 19,   /**< Unsupported version */
    DOCA_ERROR_OPERATING_SYSTEM = 20,      /**< Operating system call failure */
    DOCA_ERROR_DRIVER = 21,                /**< DOCA Driver call failure */
    DOCA_ERROR_UNEXPECTED = 22,            /**< An unexpected scenario was detected */
    DOCA_ERROR_ALREADY_EXIST = 23,         /**< Resource already exist */
    DOCA_ERROR_FULL = 24,                  /**< No more space in resource */
    DOCA_ERROR_EMPTY = 25,                 /**< No entry is available in resource */
    DOCA_ERROR_IN_PROGRESS = 26,           /**< Operation is in progress */
    DOCA_ERROR_TOO_BIG = 27,               /**< Requested operation too big to be contained */
    DOCA_ERROR_AUTHENTICATION = 28,        /**< Authentication failure */
    DOCA_ERROR_BAD_CONFIG = 29,            /**< Configuration is not valid */
    DOCA_ERROR_SKIPPED = 30, /**< Result is valid, but some previous output data was dropped */
    DOCA_ERROR_DEVICE_FATAL_ERROR = 31 /**< Device experienced a fatal error */
} doca_error_t;

#ifdef __cplusplus
}
#endif

/** @} */

#endif /* DOCA_ERROR_H_ */
