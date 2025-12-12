/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_WIN32_DL_H_
#define NCCL_WIN32_DL_H_

/*
 * Windows Dynamic Library Loading Abstraction
 * Provides dlopen/dlsym/dlclose-like functionality using Windows API
 */

#ifdef _WIN32

#include <windows.h>
#include <stdio.h>

/* Flags for ncclDlOpen - compatible with RTLD_* on Linux */
#define NCCL_RTLD_NOW 0x00002
#define NCCL_RTLD_LOCAL 0x00000

/* Thread-local storage for error messages */
#ifdef _MSC_VER
static __declspec(thread) char ncclDlError[256] = "";
#else
static __thread char ncclDlError[256] = "";
#endif

/* Open a dynamic library */
static inline void *ncclDlOpen(const char *filename, int flags)
{
    (void)flags; /* Windows doesn't have equivalent flags */
    HMODULE handle = LoadLibraryA(filename);
    if (handle == NULL)
    {
        DWORD err = GetLastError();
        snprintf(ncclDlError, sizeof(ncclDlError), "LoadLibrary failed with error %lu", err);
        return NULL;
    }
    return (void *)handle;
}

/* Get a symbol from a dynamic library */
static inline void *ncclDlSym(void *handle, const char *symbol)
{
    if (handle == NULL)
    {
        snprintf(ncclDlError, sizeof(ncclDlError), "NULL handle passed to ncclDlSym");
        return NULL;
    }
    FARPROC proc = GetProcAddress((HMODULE)handle, symbol);
    if (proc == NULL)
    {
        DWORD err = GetLastError();
        snprintf(ncclDlError, sizeof(ncclDlError), "GetProcAddress for '%s' failed with error %lu", symbol, err);
        return NULL;
    }
    return (void *)proc;
}

/* Close a dynamic library */
static inline int ncclDlClose(void *handle)
{
    if (handle == NULL)
        return 0;
    if (!FreeLibrary((HMODULE)handle))
    {
        DWORD err = GetLastError();
        snprintf(ncclDlError, sizeof(ncclDlError), "FreeLibrary failed with error %lu", err);
        return -1;
    }
    return 0;
}

/* Get error string from last dl operation */
static inline const char *ncclDlError_get(void)
{
    return ncclDlError;
}

/* Macros to provide dlopen-like interface */
#define dlopen(name, flags) ncclDlOpen(name, flags)
#define dlsym(handle, symbol) ncclDlSym(handle, symbol)
#define dlclose(handle) ncclDlClose(handle)
#define dlerror() ncclDlError_get()

/* Map RTLD_* flags to our NCCL_RTLD_* flags */
#ifndef RTLD_NOW
#define RTLD_NOW NCCL_RTLD_NOW
#endif
#ifndef RTLD_LOCAL
#define RTLD_LOCAL NCCL_RTLD_LOCAL
#endif

#endif /* _WIN32 */

#endif /* NCCL_WIN32_DL_H_ */
