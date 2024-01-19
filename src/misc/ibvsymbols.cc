#include <sys/types.h>
#include <unistd.h>

#include "ibvsymbols.h"

#ifdef NCCL_BUILD_RDMA_CORE
/* RDMA-core linking mode. Symbols are pointers to linked IB Verbs */

#define ASSIGN_SYM(container, symbol, name) container->name= &symbol;

// Passthrough function for ibv_reg_mr macro in verbs.h
struct ibv_mr* ibv_internal_reg_mr(
      struct ibv_pd* pd,
      void* addr,
      size_t length,
      int access) {
    return ibv_reg_mr(pd, addr, length, access);
  }

// Passthrough function for ibv_internal_query_port macro in verbs.h
int ibv_internal_query_port(
      struct ibv_context* context,
      uint8_t port_num,
      struct ibv_port_attr* port_attr) {
    return ibv_query_port(context, port_num, port_attr);
}

ncclResult_t buildIbvSymbols(struct ncclIbvSymbols* ibvSymbols) {
  ASSIGN_SYM(ibvSymbols, ibv_get_device_list, ibv_internal_get_device_list);
  ASSIGN_SYM(ibvSymbols, ibv_free_device_list, ibv_internal_free_device_list);
  ASSIGN_SYM(ibvSymbols, ibv_get_device_name, ibv_internal_get_device_name);
  ASSIGN_SYM(ibvSymbols, ibv_open_device, ibv_internal_open_device);
  ASSIGN_SYM(ibvSymbols, ibv_close_device, ibv_internal_close_device);
  ASSIGN_SYM(ibvSymbols, ibv_get_async_event, ibv_internal_get_async_event);
  ASSIGN_SYM(ibvSymbols, ibv_ack_async_event, ibv_internal_ack_async_event);
  ASSIGN_SYM(ibvSymbols, ibv_query_device, ibv_internal_query_device);
  ASSIGN_SYM(ibvSymbols, ibv_query_gid, ibv_internal_query_gid);
  ASSIGN_SYM(ibvSymbols, ibv_query_qp, ibv_internal_query_qp);
  ASSIGN_SYM(ibvSymbols, ibv_alloc_pd, ibv_internal_alloc_pd);
  ASSIGN_SYM(ibvSymbols, ibv_dealloc_pd, ibv_internal_dealloc_pd);

  ASSIGN_SYM(ibvSymbols, ibv_reg_mr_iova2, ibv_internal_reg_mr_iova2);
  ASSIGN_SYM(ibvSymbols, ibv_reg_dmabuf_mr, ibv_internal_reg_dmabuf_mr);

  ASSIGN_SYM(ibvSymbols, ibv_dereg_mr, ibv_internal_dereg_mr);
  ASSIGN_SYM(ibvSymbols, ibv_create_cq, ibv_internal_create_cq);
  ASSIGN_SYM(ibvSymbols, ibv_destroy_cq, ibv_internal_destroy_cq);
  ASSIGN_SYM(ibvSymbols, ibv_create_qp, ibv_internal_create_qp);
  ASSIGN_SYM(ibvSymbols, ibv_modify_qp, ibv_internal_modify_qp);
  ASSIGN_SYM(ibvSymbols, ibv_destroy_qp, ibv_internal_destroy_qp);
  ASSIGN_SYM(ibvSymbols, ibv_fork_init, ibv_internal_fork_init);
  ASSIGN_SYM(ibvSymbols, ibv_event_type_str, ibv_internal_event_type_str);
  
  ASSIGN_SYM(ibvSymbols, ibv_query_ece, ibv_internal_query_ece);
  ASSIGN_SYM(ibvSymbols, ibv_set_ece, ibv_internal_set_ece);

  ibvSymbols->ibv_internal_reg_mr = &ibv_internal_reg_mr;
  ibvSymbols->ibv_internal_query_port = &ibv_internal_query_port;

  return ncclSuccess;
}

#else
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

  LOAD_SYM_VERSION(ibvhandle, "ibv_query_ece", ibvSymbols->ibv_internal_query_ece, "IBVERBS_1.10");
  LOAD_SYM_VERSION(ibvhandle, "ibv_set_ece",   ibvSymbols->ibv_internal_set_ece, "IBVERBS_1.10");

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
  ibvSymbols->ibv_internal_query_ece = NULL;
  ibvSymbols->ibv_internal_set_ece = NULL;

  if (ibvhandle != NULL) dlclose(ibvhandle);
  return ncclSystemError;
}

#endif
