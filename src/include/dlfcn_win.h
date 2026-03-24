/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/* Windows shim for POSIX dlfcn.h (dlopen/dlsym/dlclose/dlerror) */

#ifndef NCCL_DLFCN_WIN_H_
#define NCCL_DLFCN_WIN_H_

#ifdef _WIN32

/* winsock2.h must be included before windows.h to prevent ws2def.h/winsock2.h
 * type redefinitions (sockaddr, fd_set, etc.).  CUDA runtime headers may pull
 * in windows.h early, so include winsock2.h unconditionally first. */
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <stdbool.h>

#define RTLD_NOW   0
#define RTLD_LOCAL 0
#define RTLD_LAZY  0

static inline void* dlopen(const char* filename, int flags) {
  (void)flags;
  if (filename == nullptr) return (void*)GetModuleHandleA(nullptr);
  return (void*)LoadLibraryA(filename);
}

static inline void* dlsym(void* handle, const char* symbol) {
  return (void*)GetProcAddress((HMODULE)handle, symbol);
}

static inline int dlclose(void* handle) {
  return FreeLibrary((HMODULE)handle) ? 0 : -1;
}

static inline const char* dlerror(void) {
  static char buf[512];
  DWORD err = GetLastError();
  if (err == 0) return nullptr;
  FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                 nullptr, err, 0, buf, sizeof(buf) - 1, nullptr);
  /* strip trailing newline */
  size_t len = strlen(buf);
  while (len > 0 && (buf[len-1] == '\n' || buf[len-1] == '\r')) buf[--len] = '\0';
  return buf;
}

/* dlinfo with RTLD_DI_LINKMAP is Linux-specific; provide stub */
#define RTLD_DI_LINKMAP 2
static inline int dlinfo(void* handle, int request, void* info) {
  (void)handle; (void)request; (void)info;
  /* Not supported on Windows; caller should use GetModuleFileNameA instead */
  SetLastError(ERROR_NOT_SUPPORTED);
  return -1;
}

#else
#include <dlfcn.h>
#endif /* _WIN32 */

#endif /* NCCL_DLFCN_WIN_H_ */
