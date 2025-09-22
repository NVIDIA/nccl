#ifndef NCCL_MLX5DV_CORE_H_
#define NCCL_MLX5DV_CORE_H_

/* Basic MLX5 direct verbs structs. Needed to dynamically load MLX5 direct verbs functions without
 * explicit including of MLX5 direct verbs header.
 */

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
#include <unistd.h>
#include "ibvwrap.h"

enum mlx5dv_reg_dmabuf_access  {
	MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT		= (1<<0),
};

#endif  // NCCL_MLX5DV_CORE_H_
