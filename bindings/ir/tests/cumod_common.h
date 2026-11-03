#ifndef CUMOD_COMMON_H
#define CUMOD_COMMON_H

#include <cuda.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>

#define MAX_PATH_LENGTH 1024

#define CU_CHECK(call) do {                         \
    CUresult err = call;                            \
    if (err != CUDA_SUCCESS) {                      \
        const char* errStr;                         \
        cuGetErrorString(err, &errStr);             \
        printf("CUDA error at %s:%d  '%s'\n",       \
            __FILE__, __LINE__, errStr);            \
        exit(EXIT_FAILURE);                         \
    }                                               \
} while(0)

// Resolve a file sitting next to the running executable into out.
static void resolveSiblingPath(char* out, const char* fileName) {
  char exePath[MAX_PATH_LENGTH];
  ssize_t count = readlink("/proc/self/exe", exePath, MAX_PATH_LENGTH);
  if (count == -1 || count >= MAX_PATH_LENGTH) {
    fprintf(stderr, "%s:%d readlink returned error: %d\n", __FILE__, __LINE__, errno);
    exit(EXIT_FAILURE);
  }
  exePath[count] = '\0';
  char* exeDir = dirname(exePath);
  if (snprintf(out, MAX_PATH_LENGTH, "%s/%s", exeDir, fileName) >= MAX_PATH_LENGTH) {
    fprintf(stderr, "%s:%d path too long: %s/%s\n", __FILE__, __LINE__, exeDir, fileName);
    exit(EXIT_FAILURE);
  }
}

// IR utility functions for loading and managing CUDA modules
static void initCumodule(CUmodule* module, const char* cubinName) {
  char cubinPath[MAX_PATH_LENGTH];
  resolveSiblingPath(cubinPath, cubinName);
  printf("CUBIN Selected: %s\n", cubinPath);
  CU_CHECK(cuModuleLoad(module, cubinPath));
}

static void finiCumodule(CUmodule* module) {
  if (*module != NULL) {
    CU_CHECK(cuModuleUnload(*module));
    *module = NULL;
  }
}

static void initTestCaseKernel(CUmodule module, CUfunction* kernel, const char* kernelName) {
  CU_CHECK(cuModuleGetFunction(kernel, module, kernelName));
}

#ifdef USE_LTO_IR
// JIT-link the kernel LTO_IR against the device library and load the cubin.
#include <nvJitLink.h>
#include <vector>

#ifndef LTO_IR_ARCH
#define LTO_IR_ARCH "sm_90"
#endif

// NCCL device LTO_IR library, linked into every test kernel.
#ifndef NCCL_DEVICE_LTOIR
#define NCCL_DEVICE_LTOIR "libnccl_device.ltoir"
#endif

#define NVJITLINK_CHECK(handle, call) do {                                 \
    nvJitLinkResult _res = (call);                                         \
    if (_res != NVJITLINK_SUCCESS) {                                       \
        fprintf(stderr, "nvJitLink error %d at %s:%d\n",                   \
                (int)_res, __FILE__, __LINE__);                            \
        size_t _logSize = 0;                                               \
        nvJitLinkGetErrorLogSize((handle), &_logSize);                     \
        if (_logSize > 1) {                                                \
            std::vector<char> _log(_logSize);                              \
            nvJitLinkGetErrorLog((handle), _log.data());                   \
            fprintf(stderr, "%s\n", _log.data());                          \
        }                                                                  \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while(0)

static std::vector<char> readBinaryFile(const char* path) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "%s:%d cannot open %s\n", __FILE__, __LINE__, path);
    exit(EXIT_FAILURE);
  }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<char> buf(n);
  if (fread(buf.data(), 1, (size_t)n, f) != (size_t)n) {
    fprintf(stderr, "%s:%d short read on %s\n", __FILE__, __LINE__, path);
    fclose(f);
    exit(EXIT_FAILURE);
  }
  fclose(f);
  return buf;
}

// Device-only TUs emit an ELF object (with embedded LTO_IR).
// TUs with a __global__ entry emit raw LTO_IR.
// Pick the input type from the magic bytes.
static nvJitLinkInputType ltoirInputType(const std::vector<char>& b) {
  if (b.size() >= 4 && (unsigned char)b[0] == 0x7f && b[1] == 'E' && b[2] == 'L' && b[3] == 'F')
    return NVJITLINK_INPUT_OBJECT;
  return NVJITLINK_INPUT_LTOIR;
}

static void initCumoduleFromLtoir(CUmodule* module, const char* kernelLtoir) {
  const char* libLtoir = NCCL_DEVICE_LTOIR;
  char kernelPath[MAX_PATH_LENGTH];
  char libPath[MAX_PATH_LENGTH];
  resolveSiblingPath(kernelPath, kernelLtoir);
  resolveSiblingPath(libPath, libLtoir);

  std::vector<char> kernelIr = readBinaryFile(kernelPath);
  std::vector<char> libIr = readBinaryFile(libPath);

  nvJitLinkHandle linker;
  const char* arch = "-arch=" LTO_IR_ARCH;
  const char* linkOpts[] = {arch, "-lto"};
  NVJITLINK_CHECK(linker, nvJitLinkCreate(&linker, 2, linkOpts));
  NVJITLINK_CHECK(linker, nvJitLinkAddData(linker, ltoirInputType(kernelIr),
                                           kernelIr.data(), kernelIr.size(), kernelLtoir));
  NVJITLINK_CHECK(linker, nvJitLinkAddData(linker, ltoirInputType(libIr),
                                           libIr.data(), libIr.size(), libLtoir));
  NVJITLINK_CHECK(linker, nvJitLinkComplete(linker));

  size_t cubinSize = 0;
  NVJITLINK_CHECK(linker, nvJitLinkGetLinkedCubinSize(linker, &cubinSize));
  std::vector<char> cubin(cubinSize);
  NVJITLINK_CHECK(linker, nvJitLinkGetLinkedCubin(linker, cubin.data()));
  NVJITLINK_CHECK(linker, nvJitLinkDestroy(&linker));

  printf("LTO_IR JIT-linked cubin: %zu bytes (%s + %s)\n", cubinSize, kernelLtoir, libLtoir);
  CU_CHECK(cuModuleLoadData(module, cubin.data()));
}
#endif // USE_LTO_IR

#endif // CUMOD_COMMON_H
