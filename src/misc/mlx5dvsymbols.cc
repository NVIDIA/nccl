#include <sys/types.h>
#include <unistd.h>

#include "mlx5/mlx5dvsymbols.h"

#ifdef NCCL_BUILD_MLX5DV
/* Mlx5dv linking mode. Symbols are pointers to linked MLX5 Direct Verbs */

#define ASSIGN_SYM(container, symbol, name) container->name= &symbol;

ncclResult_t buildMlx5dvSymbols(struct ncclMlx5dvSymbols* mlx5dvSymbols) {
  ASSIGN_SYM(mlx5dvSymbols, mlx5dv_is_supported, mlx5dv_internal_is_supported);
  ASSIGN_SYM(mlx5dvSymbols, mlx5dv_get_data_direct_sysfs_path, mlx5dv_internal_get_data_direct_sysfs_path);
  ASSIGN_SYM(mlx5dvSymbols, mlx5dv_reg_dmabuf_mr, mlx5dv_internal_reg_dmabuf_mr);
  return ncclSuccess;
}

#else
/* Mlx5dv dynamic loading mode. Symbols are loaded from shared objects. */

#include <dlfcn.h>
#include "core.h"

// MLX5DV Library versioning
#define MLX5DV_VERSION "MLX5_1.8"

ncclResult_t buildMlx5dvSymbols(struct ncclMlx5dvSymbols* mlx5dvSymbols) {
  static void* mlx5dvhandle = NULL;
  void* tmp;
  void** cast;

  mlx5dvhandle=dlopen("libmlx5.so", RTLD_NOW);
  if (!mlx5dvhandle) {
    mlx5dvhandle=dlopen("libmlx5.so.1", RTLD_NOW);
    if (!mlx5dvhandle) {
      INFO(NCCL_INIT, "Failed to open libmlx5.so[.1]");
      goto teardown;
    }
  }

#define LOAD_SYM(handle, symbol, funcptr) do {           \
    cast = (void**)&funcptr;                             \
    tmp = dlvsym(handle, symbol, MLX5DV_VERSION);       \
    if (tmp == NULL) {                                   \
      WARN("dlvsym failed on %s - %s version %s", symbol, dlerror(), MLX5DV_VERSION);  \
      goto teardown;                                     \
    }                                                    \
    *cast = tmp;                                         \
  } while (0)

// Attempt to load a specific symbol version - fail silently
#define LOAD_SYM_VERSION(handle, symbol, funcptr, version) do {  \
    cast = (void**)&funcptr;                                     \
    *cast = dlvsym(handle, symbol, version);                     \
    if (*cast == NULL) {                                         \
      INFO(NCCL_NET, "dlvsym failed on %s - %s version %s", symbol, dlerror(), version);  \
    }                                                            \
  } while (0)

  LOAD_SYM(mlx5dvhandle, "mlx5dv_is_supported", mlx5dvSymbols->mlx5dv_internal_is_supported);
  // Cherry-pick the mlx5dv_get_data_direct_sysfs_path API from MLX5 1.25
  LOAD_SYM_VERSION(mlx5dvhandle, "mlx5dv_get_data_direct_sysfs_path", mlx5dvSymbols->mlx5dv_internal_get_data_direct_sysfs_path, "MLX5_1.25");
  // Cherry-pick the ibv_reg_dmabuf_mr API from MLX5 1.25
  LOAD_SYM_VERSION(mlx5dvhandle, "mlx5dv_reg_dmabuf_mr", mlx5dvSymbols->mlx5dv_internal_reg_dmabuf_mr, "MLX5_1.25");

  return ncclSuccess;

teardown:
  mlx5dvSymbols->mlx5dv_internal_is_supported = NULL;
  mlx5dvSymbols->mlx5dv_internal_get_data_direct_sysfs_path = NULL;
  mlx5dvSymbols->mlx5dv_internal_reg_dmabuf_mr = NULL;

  if (mlx5dvhandle != NULL) dlclose(mlx5dvhandle);
  return ncclSystemError;
}

#endif
