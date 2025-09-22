/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "mlx5/mlx5dvwrap.h"
#include <sys/types.h>
#include <unistd.h>
#include <mutex>

#ifdef NCCL_BUILD_MLX5DV
#include <infiniband/mlx5dv.h>
#else
#include "mlx5/mlx5dvcore.h"
#endif
#include "mlx5/mlx5dvsymbols.h"

static std::once_flag initOnceFlag;
static ncclResult_t initResult;
struct ncclMlx5dvSymbols mlx5dvSymbols;

ncclResult_t wrap_mlx5dv_symbols(void) {
  std::call_once(initOnceFlag,
               [](){ initResult = buildMlx5dvSymbols(&mlx5dvSymbols); });
  return initResult;
}

/* CHECK_NOT_NULL: helper macro to check for NULL symbol */
#define CHECK_NOT_NULL(container, internal_name) \
  if (container.internal_name == NULL) { \
     WARN("NET/MLX5: lib wrapper not initialized."); \
     return ncclInternalError; \
  }

#define MLX5DV_PTR_CHECK_ERRNO(container, internal_name, call, retval, error_retval, name) \
  CHECK_NOT_NULL(container, internal_name); \
  retval = container.call; \
  if (retval == error_retval) { \
    WARN("NET/MLX5: Call to " name " failed with error %s", strerror(errno)); \
    return ncclSystemError; \
  } \
  return ncclSuccess;

bool wrap_mlx5dv_is_supported(struct ibv_device *device) {
  if (mlx5dvSymbols.mlx5dv_internal_is_supported == NULL) {
    return 0;
  }
  return mlx5dvSymbols.mlx5dv_internal_is_supported(device);
}

ncclResult_t wrap_mlx5dv_get_data_direct_sysfs_path(struct ibv_context* context, char* buf, size_t buf_len) {
  CHECK_NOT_NULL(mlx5dvSymbols, mlx5dv_internal_get_data_direct_sysfs_path);
  int ret = mlx5dvSymbols.mlx5dv_internal_get_data_direct_sysfs_path(context, buf, buf_len);
  if (ret == 0) return ncclSuccess;
  /* ENODEV can happen if the devices is not data-direct but mlx5 is used. It's not an error*/
  if (ret == ENODEV) return ncclInvalidArgument;
  INFO(NCCL_NET, "NET/MLX5: Call to mlx5dv_internal_get_data_direct_sysfs_path failed with error %s errno %d", strerror(ret), ret);
  return ncclSystemError;
}

/* DMA-BUF support */
ncclResult_t wrap_mlx5dv_reg_dmabuf_mr(struct ibv_mr **ret, struct ibv_pd *pd, uint64_t offset, size_t length, uint64_t iova, int fd, int access, int mlx5_access) {
  MLX5DV_PTR_CHECK_ERRNO(mlx5dvSymbols, mlx5dv_internal_reg_dmabuf_mr, mlx5dv_internal_reg_dmabuf_mr(pd, offset, length, iova, fd, access, mlx5_access), *ret, NULL, "mlx5dv_reg_dmabuf_mr");
}

struct ibv_mr * wrap_direct_mlx5dv_reg_dmabuf_mr(struct ibv_pd *pd, uint64_t offset, size_t length, uint64_t iova, int fd, int access, int mlx5_access) {
  if (mlx5dvSymbols.mlx5dv_internal_reg_dmabuf_mr == NULL) {
    errno = EOPNOTSUPP; // ncclIbDmaBufSupport() requires this errno being set
    return NULL;
  }
  return mlx5dvSymbols.mlx5dv_internal_reg_dmabuf_mr(pd, offset, length, iova, fd, access, mlx5_access);
}
