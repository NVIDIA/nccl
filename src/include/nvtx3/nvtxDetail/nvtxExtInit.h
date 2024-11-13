/*
* Copyright 2009-2023  NVIDIA Corporation.  All rights reserved.
*
* Licensed under the Apache License v2.0 with LLVM Exceptions.
* See https://llvm.org/LICENSE.txt for license information.
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

#ifndef NVTX_EXT_INIT_GUARD
#error Never include this file directly -- it is automatically included by nvToolsExt.h (except when NVTX_NO_IMPL is defined).
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* ---- Platform-independent helper definitions and functions ---- */

/* Prefer macros over inline functions to reduce symbol resolution at link time */

#if defined(_WIN32)
#define NVTX_PATHCHAR   wchar_t
#define NVTX_STR(x)     L##x
#define NVTX_GETENV     _wgetenv
#define NVTX_BUFSIZE    16384
#define NVTX_DLLHANDLE  HMODULE
#define NVTX_DLLOPEN(x) LoadLibraryW(x)
#define NVTX_DLLFUNC    GetProcAddress
#define NVTX_DLLCLOSE   FreeLibrary
#define NVTX_YIELD()    SwitchToThread()
#define NVTX_MEMBAR()   MemoryBarrier()
#define NVTX_ATOMIC_WRITE_32(address, value)                        InterlockedExchange((volatile LONG*)address, value)
#define NVTX_ATOMIC_CAS_32(old, address, exchange, comparand) old = InterlockedCompareExchange((volatile LONG*)address, exchange, comparand)
#define NVTX_ATOMIC_WRITE_PTR(address, value)                        InterlockedExchangePointer((volatile PVOID*)address, (PVOID)value)
#define NVTX_ATOMIC_CAS_PTR(old, address, exchange, comparand) old = (intptr_t)InterlockedCompareExchangePointer((volatile PVOID*)address, (PVOID)exchange, (PVOID)comparand)


#elif defined(__GNUC__)
#define NVTX_PATHCHAR   char
#define NVTX_STR(x)     x
#define NVTX_GETENV     getenv
#define NVTX_BUFSIZE    16384
#define NVTX_DLLHANDLE  void*
#define NVTX_DLLOPEN(x) dlopen(x, RTLD_LAZY)
#define NVTX_DLLFUNC    dlsym
#define NVTX_DLLCLOSE   dlclose
#define NVTX_YIELD()    sched_yield()
#define NVTX_MEMBAR()   __sync_synchronize()
/* Ensure full memory barrier for atomics, to match Windows functions. */
#define NVTX_ATOMIC_WRITE_32(address, value)                  __sync_synchronize();       __sync_lock_test_and_set(address, value)
#define NVTX_ATOMIC_CAS_32(old, address, exchange, comparand) __sync_synchronize(); old = __sync_val_compare_and_swap(address, exchange, comparand)
#define NVTX_ATOMIC_WRITE_PTR(address, value)                  __sync_synchronize();       __sync_lock_test_and_set(address, value)
#define NVTX_ATOMIC_CAS_PTR(old, address, exchange, comparand) __sync_synchronize(); old = __sync_val_compare_and_swap(address, exchange, comparand)
#else
#error The library does not support your configuration!
#endif

/* Define this to 1 for platforms that where pre-injected libraries can be discovered. */
#if defined(_WIN32)
/* TODO */
#define NVTX_SUPPORT_ALREADY_INJECTED_LIBRARY 0
#else
#define NVTX_SUPPORT_ALREADY_INJECTED_LIBRARY 0
#endif

/* Define this to 1 for platforms that support environment variables. */
/* TODO: Detect UWP, a.k.a. Windows Store app, and set this to 0. */
/* Try:  #if defined(WINAPI_FAMILY_PARTITION) && WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_APP) */
#define NVTX_SUPPORT_ENV_VARS 1

/* Define this to 1 for platforms that support dynamic/shared libraries */
#define NVTX_SUPPORT_DYNAMIC_INJECTION_LIBRARY 1

/* Injection libraries implementing InitializeInjectionNvtxExtension may be statically linked,
 * which will override any dynamic injection. This is useful for platforms, where dynamic
 * injection is not available. Since weak symbols, not explicitly marked extern, are
 * guaranteed to be initialized to zero, if no definitions are found by the linker, the
 * dynamic injection process proceeds normally, if pfnInitializeInjectionNvtx2 is 0. */
#if defined(__GNUC__) && !defined(_WIN32) && !defined(__CYGWIN__)
#define NVTX_SUPPORT_STATIC_INJECTION_LIBRARY 1
/* To statically inject an NVTX library, define InitializeInjectionNvtxExtension_fnptr as a normal
 * symbol (not weak) pointing to the implementation of InitializeInjectionNvtxExtension, which
 * does not need to be named "InitializeInjectionNvtxExtension" as it is necessary in a dynamic
 * injection library. */
__attribute__((weak)) NvtxExtInitializeInjectionFunc_t InitializeInjectionNvtxExtension_fnptr;
#else
#define NVTX_SUPPORT_STATIC_INJECTION_LIBRARY 0
#endif



/* This function tries to find or load an NVTX injection library and get the address of its
 * `InitializeInjectionExtension` function. If such a function pointer is found, it is called and
 * passed the address of this NVTX instance's `nvtxGetExportTable` function, so that the injection
 * can attach to this instance.
 * If the initialization fails for any reason, any dynamic library loaded will  be freed, and all
 * NVTX implementation functions will be set to no-ops. If the initialization succeeds, NVTX
 * functions that are not attached to the tool will be set to no-ops. This is implemented as one
 * function instead of several small functions to minimize the number of weak symbols the linker
 * must resolve. The order of search is:
 *  1) Pre-injected library exporting InitializeInjectionNvtxExtension
 *  2) Loadable library exporting InitializeInjectionNvtxExtension
 *      - Path specified by env var NVTX_INJECTION??_PATH (?? is 32 or 64)
 *      - On Android, libNvtxInjection??.so within the package (?? is 32 or 64)
 *  3) Statically-linked injection library defining InitializeInjectionNvtx2_fnptr
 */
NVTX_LINKONCE_FWDDECL_FUNCTION int NVTX_VERSIONED_IDENTIFIER(nvtxExtLoadInjectionLibrary)(
    NvtxExtInitializeInjectionFunc_t* out_init_fnptr);
NVTX_LINKONCE_DEFINE_FUNCTION int NVTX_VERSIONED_IDENTIFIER(nvtxExtLoadInjectionLibrary)(
    NvtxExtInitializeInjectionFunc_t* out_init_fnptr)
{
    const char* const initFuncName = "InitializeInjectionNvtxExtension";
    NvtxExtInitializeInjectionFunc_t init_fnptr = (NvtxExtInitializeInjectionFunc_t)0;
    NVTX_DLLHANDLE injectionLibraryHandle = (NVTX_DLLHANDLE)0;

    if (out_init_fnptr)
    {
        *out_init_fnptr = (NvtxExtInitializeInjectionFunc_t)0;
    }

#if NVTX_SUPPORT_ALREADY_INJECTED_LIBRARY
    /* Use POSIX global symbol chain to query for init function from any module. */
    init_fnptr = (NvtxExtInitializeInjectionFunc_t)NVTX_DLLFUNC(0, initFuncName);
#endif

#if NVTX_SUPPORT_DYNAMIC_INJECTION_LIBRARY
    /* Try discovering dynamic injection library to load */
    if (!init_fnptr)
    {
#if NVTX_SUPPORT_ENV_VARS
        /* If env var NVTX_INJECTION64_PATH is set, it should contain the path
           to a 64-bit dynamic NVTX injection library (and similar for 32-bit). */
        const NVTX_PATHCHAR* const nvtxEnvVarName = (sizeof(void*) == 4)
            ? NVTX_STR("NVTX_INJECTION32_PATH")
            : NVTX_STR("NVTX_INJECTION64_PATH");
#endif /* NVTX_SUPPORT_ENV_VARS */
        NVTX_PATHCHAR injectionLibraryPathBuf[NVTX_BUFSIZE];
        const NVTX_PATHCHAR* injectionLibraryPath = (const NVTX_PATHCHAR*)0;

        /* Refer to this variable explicitly in case all references to it are #if'ed out. */
        (void)injectionLibraryPathBuf;

#if NVTX_SUPPORT_ENV_VARS
        /* Disable the warning for getenv & _wgetenv -- this usage is safe because
           these functions are not called again before using the returned value. */
#if defined(_MSC_VER)
#pragma warning( push )
#pragma warning( disable : 4996 )
#endif
        injectionLibraryPath = NVTX_GETENV(nvtxEnvVarName);
#if defined(_MSC_VER)
#pragma warning( pop )
#endif
#endif

#if defined(__ANDROID__)
        if (!injectionLibraryPath)
        {
            const char *bits = (sizeof(void*) == 4) ? "32" : "64";
            char cmdlineBuf[32];
            char pkgName[PATH_MAX];
            int count;
            int pid;
            FILE *fp;
            size_t bytesRead;
            size_t pos;

            pid = (int)getpid();
            count = snprintf(cmdlineBuf, sizeof(cmdlineBuf), "/proc/%d/cmdline", pid);
            if (count <= 0 || count >= (int)sizeof(cmdlineBuf))
            {
                NVTX_ERR("Path buffer too small for: /proc/%d/cmdline\n", pid);
                return NVTX_ERR_INIT_ACCESS_LIBRARY;
            }

            fp = fopen(cmdlineBuf, "r");
            if (!fp)
            {
                NVTX_ERR("File couldn't be opened: %s\n", cmdlineBuf);
                return NVTX_ERR_INIT_ACCESS_LIBRARY;
            }

            bytesRead = fread(pkgName, 1, sizeof(pkgName) - 1, fp);
            fclose(fp);
            if (bytesRead == 0)
            {
                NVTX_ERR("Package name couldn't be read from file: %s\n", cmdlineBuf);
                return NVTX_ERR_INIT_ACCESS_LIBRARY;
            }

            pkgName[bytesRead] = 0;

            /* String can contain colon as a process separator. In this case the
               package name is before the colon. */
            pos = 0;
            while (pos < bytesRead && pkgName[pos] != ':' && pkgName[pos] != '\0')
            {
                ++pos;
            }
            pkgName[pos] = 0;

            count = snprintf(injectionLibraryPathBuf, NVTX_BUFSIZE, "/data/data/%s/files/libNvtxInjection%s.so", pkgName, bits);
            if (count <= 0 || count >= NVTX_BUFSIZE)
            {
                NVTX_ERR("Path buffer too small for: /data/data/%s/files/libNvtxInjection%s.so\n", pkgName, bits);
                return NVTX_ERR_INIT_ACCESS_LIBRARY;
            }

            /* On Android, verify path is accessible due to aggressive file access restrictions. */
            /* For dlopen, if the filename contains a leading slash, then it is interpreted as a */
            /* relative or absolute pathname; otherwise it will follow the rules in ld.so. */
            if (injectionLibraryPathBuf[0] == '/')
            {
#if (__ANDROID_API__ < 21)
                int access_err = access(injectionLibraryPathBuf, F_OK | R_OK);
#else
                int access_err = faccessat(AT_FDCWD, injectionLibraryPathBuf, F_OK | R_OK, 0);
#endif
                if (access_err != 0)
                {
                    NVTX_ERR("Injection library path wasn't accessible [code=%s] [path=%s]\n", strerror(errno), injectionLibraryPathBuf);
                    return NVTX_ERR_INIT_ACCESS_LIBRARY;
                }
            }
            injectionLibraryPath = injectionLibraryPathBuf;
        }
#endif

        /* At this point, `injectionLibraryPath` is specified if a dynamic
           injection library was specified by a tool. */
        if (injectionLibraryPath)
        {
            /* Load the injection library */
            injectionLibraryHandle = NVTX_DLLOPEN(injectionLibraryPath);
            if (!injectionLibraryHandle)
            {
                NVTX_ERR("Failed to load injection library\n");
                return NVTX_ERR_INIT_LOAD_LIBRARY;
            }
            else
            {
                /* Attempt to get the injection library's entry-point. */
                init_fnptr = (NvtxExtInitializeInjectionFunc_t)NVTX_DLLFUNC(injectionLibraryHandle, initFuncName);
                if (!init_fnptr)
                {
                    NVTX_DLLCLOSE(injectionLibraryHandle);
                    NVTX_ERR("Failed to get address of function %s from injection library\n", initFuncName);
                    return NVTX_ERR_INIT_MISSING_LIBRARY_ENTRY_POINT;
                }
            }
        }
    }
#endif

#if NVTX_SUPPORT_STATIC_INJECTION_LIBRARY
    if (!init_fnptr)
    {
        /* Check weakly-defined function pointer.  A statically-linked injection can define
           this as a normal symbol and it will take precedence over a dynamic injection. */
        if (InitializeInjectionNvtxExtension_fnptr)
        {
            init_fnptr = InitializeInjectionNvtxExtension_fnptr;
        }
    }
#endif

    if (out_init_fnptr)
    {
        *out_init_fnptr = init_fnptr;
    }

    /* At this point, if `init_fnptr` is not set, no tool has specified an NVTX injection library.
       Non-success result is returned, so that all NVTX API functions will be set to no-ops. */
    if (!init_fnptr)
    {
        return NVTX_ERR_NO_INJECTION_LIBRARY_AVAILABLE;
    }

    return NVTX_SUCCESS;
}

/* Avoid warnings about missing prototypes. */
NVTX_LINKONCE_FWDDECL_FUNCTION void NVTX_VERSIONED_IDENTIFIER(nvtxExtInitOnce) (
    nvtxExtModuleInfo_t* moduleInfo, intptr_t* moduleState);
NVTX_LINKONCE_DEFINE_FUNCTION void NVTX_VERSIONED_IDENTIFIER(nvtxExtInitOnce) (
    nvtxExtModuleInfo_t* moduleInfo, intptr_t* moduleState)
{
    intptr_t old;

    NVTX_INFO( "%s\n", __FUNCTION__ );

    if (*moduleState == NVTX_EXTENSION_LOADED)
    {
        NVTX_INFO("Module loaded\n");
        return;
    }

    NVTX_ATOMIC_CAS_PTR(
        old,
        moduleState,
        NVTX_EXTENSION_STARTING,
        NVTX_EXTENSION_FRESH);
    if (old == NVTX_EXTENSION_FRESH)
    {
        NvtxExtInitializeInjectionFunc_t init_fnptr =
            NVTX_VERSIONED_IDENTIFIER(nvtxExtGlobals1).injectionFnPtr;
        int entryPointStatus = 0;
        int forceAllToNoops = 0;
        size_t s;

        /* Load and initialize injection library, which will assign the function pointers. */
        if (init_fnptr == 0)
        {
            int result = 0;

            /* Try to load vanilla NVTX first. */
            nvtxInitialize(0);

            result = NVTX_VERSIONED_IDENTIFIER(nvtxExtLoadInjectionLibrary)(&init_fnptr);
            /* At this point `init_fnptr` will be either 0 or a real function. */

            if (result == NVTX_SUCCESS)
            {
                NVTX_VERSIONED_IDENTIFIER(nvtxExtGlobals1).injectionFnPtr = init_fnptr;
            }
            else
            {
                NVTX_ERR("Failed to load injection library\n");
            }
        }

        if (init_fnptr != 0)
        {
            /* Invoke injection library's initialization function. If it returns
               0 (failure) and a dynamic injection was loaded, unload it. */
            entryPointStatus = init_fnptr(moduleInfo);
            if (entryPointStatus == 0)
            {
                NVTX_ERR("Failed to initialize injection library -- initialization function returned 0\n");
            }
        }

        /* Clean up any functions that are still uninitialized so that they are
           skipped. Set all to null if injection init function failed as well. */
        forceAllToNoops = (init_fnptr == 0) || (entryPointStatus == 0);
        for (s = 0; s < moduleInfo->segmentsCount; ++s)
        {
            nvtxExtModuleSegment_t* segment = moduleInfo->segments + s;
            size_t i;
            for (i = 0; i < segment->slotCount; ++i)
            {
                if (forceAllToNoops || (segment->functionSlots[i] == NVTX_EXTENSION_FRESH))
                {
                    segment->functionSlots[i] = NVTX_EXTENSION_DISABLED;
                }
            }
        }

        NVTX_MEMBAR();

        /* Signal that initialization has finished and the assigned function
           pointers will be used. */
        NVTX_ATOMIC_WRITE_PTR(moduleState, NVTX_EXTENSION_LOADED);
    }
    else /* Spin-wait until initialization has finished. */
    {
        NVTX_MEMBAR();
        while (*moduleState != NVTX_EXTENSION_LOADED)
        {
            NVTX_YIELD();
            NVTX_MEMBAR();
        }
    }
}

#ifdef __cplusplus
}
#endif /* __cplusplus */
