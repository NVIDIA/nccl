#ifndef NCCL_CHECKPOINT_H_
#define NCCL_CHECKPOINT_H_

/*
 * Public helper header for applications that use the NCCL checkpoint shim via
 * LD_PRELOAD.  Including this header does not create a link dependency on the
 * shim library.  Resolve the checkpoint entry points at runtime after the
 * process has been launched with libnccl-checkpoint-shim.so preloaded.
 *
 * This header uses dlsym(3).  Applications may need to link with -ldl on
 * platforms where dlsym is provided by libdl rather than libc.
 */

#include <dlfcn.h>
#include <nccl.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef ncclResult_t (*ncclCheckpointPrepareFn)(void);
typedef ncclResult_t (*ncclCheckpointRestoreFn)(void);
typedef ncclResult_t (*ncclCheckpointGetVersionFn)(int* checkpointVersion, int* ncclVersion);

typedef struct ncclCheckpointApi {
  ncclCheckpointPrepareFn prepare;
  ncclCheckpointRestoreFn restore;
  ncclCheckpointGetVersionFn getVersion;
} ncclCheckpointApi;

static inline void ncclCheckpointApiClear(ncclCheckpointApi* api) {
  if (api == NULL) return;
  api->prepare = NULL;
  api->restore = NULL;
  api->getVersion = NULL;
}

static inline int ncclCheckpointApiIsLoaded(const ncclCheckpointApi* api) {
  return api != NULL && api->prepare != NULL && api->restore != NULL && api->getVersion != NULL;
}

static inline int ncclCheckpointApiLoad(ncclCheckpointApi* api) {
  if (api == NULL) return 1;
  api->prepare = (ncclCheckpointPrepareFn)dlsym(RTLD_DEFAULT, "ncclCheckpointPrepare");
  api->restore = (ncclCheckpointRestoreFn)dlsym(RTLD_DEFAULT, "ncclCheckpointRestore");
  api->getVersion = (ncclCheckpointGetVersionFn)dlsym(RTLD_DEFAULT, "ncclCheckpointGetVersion");
  return ncclCheckpointApiIsLoaded(api) ? 0 : 1;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // NCCL_CHECKPOINT_H_
