#include <sys/types.h>
#include <unistd.h>

#include "ibvsymbols.h"

/* RDMA-core dynamic loading mode. Symbols are loaded from shared objects. */

#include <dlfcn.h>
#include "core.h"

// IBVERBS Library versioning
#define IBVERBS_VERSION "IBVERBS_1.1"

ncclResult_t buildIbvSymbols(struct ncclIbvSymbols* ibvSymbols) {
  static void* ibvhandle = NULL;
  void* tmp;
  void** cast;

  ibvhandle=dlopen("libibverbs.so", RTLD_NOW);
  if (!ibvhandle) {
    ibvhandle=dlopen("libibverbs.so.1", RTLD_NOW);
    if (!ibvhandle) {
      INFO(NCCL_INIT, "Failed to open libibverbs.so[.1]");
      goto teardown;
    }
  }

#define LOAD_SYM(handle, symbol, funcptr) do {           \
    cast = (void**)&funcptr;                             \
    tmp = dlvsym(handle, symbol, IBVERBS_VERSION);       \
    if (tmp == NULL) {                                   \
      WARN("dlvsym failed on %s - %s version %s", symbol, dlerror(), IBVERBS_VERSION);  \
      goto teardown;                                     \
    }                                                    \
    *cast = tmp;                                         \
  } while (0)

// Attempt to load a specific symbol version - fail silently
#define LOAD_SYM_VERSION(handle, symbol, funcptr, version) do {  \
    cast = (void**)&funcptr;                                     \
    *cast = dlvsym(handle, symbol, version);                     \
  } while (0)

  LOAD_SYM(ibvhandle, "ibv_get_device_list", ibvSymbols->ibv_internal_get_device_list);
  LOAD_SYM(ibvhandle, "ibv_free_device_list", ibvSymbols->ibv_internal_free_device_list);
  LOAD_SYM(ibvhandle, "ibv_get_device_name", ibvSymbols->ibv_internal_get_device_name);
  LOAD_SYM(ibvhandle, "ibv_open_device", ibvSymbols->ibv_internal_open_device);
  LOAD_SYM(ibvhandle, "ibv_close_device", ibvSymbols->ibv_internal_close_device);
  LOAD_SYM(ibvhandle, "ibv_get_async_event", ibvSymbols->ibv_internal_get_async_event);
  LOAD_SYM(ibvhandle, "ibv_ack_async_event", ibvSymbols->ibv_internal_ack_async_event);
  LOAD_SYM(ibvhandle, "ibv_query_device", ibvSymbols->ibv_internal_query_device);
  LOAD_SYM(ibvhandle, "ibv_query_port", ibvSymbols->ibv_internal_query_port);
  LOAD_SYM(ibvhandle, "ibv_query_gid", ibvSymbols->ibv_internal_query_gid);
  LOAD_SYM(ibvhandle, "ibv_query_qp", ibvSymbols->ibv_internal_query_qp);
  LOAD_SYM(ibvhandle, "ibv_alloc_pd", ibvSymbols->ibv_internal_alloc_pd);
  LOAD_SYM(ibvhandle, "ibv_dealloc_pd", ibvSymbols->ibv_internal_dealloc_pd);
  LOAD_SYM(ibvhandle, "ibv_reg_mr", ibvSymbols->ibv_internal_reg_mr);
  // Cherry-pick the ibv_reg_mr_iova2 API from IBVERBS 1.8
  LOAD_SYM_VERSION(ibvhandle, "ibv_reg_mr_iova2", ibvSymbols->ibv_internal_reg_mr_iova2, "IBVERBS_1.8");
  // Cherry-pick the ibv_reg_dmabuf_mr API from IBVERBS 1.12
  LOAD_SYM_VERSION(ibvhandle, "ibv_reg_dmabuf_mr", ibvSymbols->ibv_internal_reg_dmabuf_mr, "IBVERBS_1.12");
  LOAD_SYM(ibvhandle, "ibv_dereg_mr", ibvSymbols->ibv_internal_dereg_mr);
  LOAD_SYM(ibvhandle, "ibv_create_cq", ibvSymbols->ibv_internal_create_cq);
  LOAD_SYM(ibvhandle, "ibv_destroy_cq", ibvSymbols->ibv_internal_destroy_cq);
  LOAD_SYM(ibvhandle, "ibv_create_qp", ibvSymbols->ibv_internal_create_qp);
  LOAD_SYM(ibvhandle, "ibv_modify_qp", ibvSymbols->ibv_internal_modify_qp);
  LOAD_SYM(ibvhandle, "ibv_destroy_qp", ibvSymbols->ibv_internal_destroy_qp);
  LOAD_SYM(ibvhandle, "ibv_fork_init", ibvSymbols->ibv_internal_fork_init);
  LOAD_SYM(ibvhandle, "ibv_event_type_str", ibvSymbols->ibv_internal_event_type_str);

  return ncclSuccess;

teardown:
  ibvSymbols->ibv_internal_get_device_list = NULL;
  ibvSymbols->ibv_internal_free_device_list = NULL;
  ibvSymbols->ibv_internal_get_device_name = NULL;
  ibvSymbols->ibv_internal_open_device = NULL;
  ibvSymbols->ibv_internal_close_device = NULL;
  ibvSymbols->ibv_internal_get_async_event = NULL;
  ibvSymbols->ibv_internal_ack_async_event = NULL;
  ibvSymbols->ibv_internal_query_device = NULL;
  ibvSymbols->ibv_internal_query_port = NULL;
  ibvSymbols->ibv_internal_query_gid = NULL;
  ibvSymbols->ibv_internal_query_qp = NULL;
  ibvSymbols->ibv_internal_alloc_pd = NULL;
  ibvSymbols->ibv_internal_dealloc_pd = NULL;
  ibvSymbols->ibv_internal_reg_mr = NULL;
  ibvSymbols->ibv_internal_reg_mr_iova2 = NULL;
  ibvSymbols->ibv_internal_reg_dmabuf_mr = NULL;
  ibvSymbols->ibv_internal_dereg_mr = NULL;
  ibvSymbols->ibv_internal_create_cq = NULL;
  ibvSymbols->ibv_internal_destroy_cq = NULL;
  ibvSymbols->ibv_internal_create_qp = NULL;
  ibvSymbols->ibv_internal_modify_qp = NULL;
  ibvSymbols->ibv_internal_destroy_qp = NULL;
  ibvSymbols->ibv_internal_fork_init = NULL;
  ibvSymbols->ibv_internal_event_type_str = NULL;

  if (ibvhandle != NULL) dlclose(ibvhandle);
  return ncclSystemError;
}
