/**
 * @file doca_gpunetio_signal.h
 * @brief A header file for the doca_gpunetio rank-local signal map
 *
 * A signal map resolves the signal id carried in the immediate of an
 * RDMA_WRITE_WITH_IMM into an 8-byte slot in GPU memory, and adds the signal value to it.
 *
 * None of these calls are thread-safe, and delivery assumes a single writer.
 */

#ifndef DOCA_GPUNETIO_SIGNAL_H
#define DOCA_GPUNETIO_SIGNAL_H

#include <stdint.h>

#include "doca_gpunetio.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DOCA_GPUNETIO_VERBS_SIGNAL_MAX_SIGNALS \
    (DOCA_GPUNETIO_VERBS_SIGNAL_IMM_ID_MASK + UINT32_C(1))
#define DOCA_GPUNETIO_VERBS_SIGNAL_SLOT_SIZE 8

/** Eight-byte GPU-resident running total updated when a signal is delivered. */
typedef uint64_t doca_gpu_verbs_signal_value_t;

struct doca_gpu_verbs_signal_map;

/**
 * Create an empty signal map.
 *
 * Allocates no GPU memory: slots arrive at registration.
 *
 * The CUDA device current on the calling host thread defines the device for this map. The library
 * does not retain or validate the cudaContext. The application must make that same context current.
 *
 * @param [out] out_map
 * The new map. Destroy it with doca_gpu_verbs_signal_map_destroy().
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 * - DOCA_ERROR_NOT_SUPPORTED - if GDRCopy is not available.
 * - DOCA_ERROR_NO_MEMORY - if the bookkeeping could not be allocated.
 */
doca_error_t doca_gpu_verbs_signal_map_create(struct doca_gpu_verbs_signal_map **out_map);

/**
 * Bind a signal id to an 8-byte slot the application allocated.
 *
 * Registration pins the 64 KiB GPU page containing the slot. It always creates a legacy CPU
 * mapping and also creates a forced-PCIe mapping when the GDRCopy runtime supports data direct.
 * Slots that share a page - which small allocations usually do - share those mappings, which stay
 * pinned until the last id in the page is unregistered.
 *
 * The application must make the CUDA context that was used to create the map current on
 * the calling host thread. gpu_addr must be device memory allocated on that device.
 *
 * Each id must use a unique slot address. Registering the same address for more than one id is
 * rejected.
 *
 * The application retains ownership of gpu_addr. It must keep every signal allocation alive until
 * doca_gpu_verbs_signal_map_destroy() succeeds, because separate CUDA allocations can share a
 * pinned 64 KiB page. The library never frees signal memory. The application must not issue CUDA
 * memory operations against the allocation while any of its slots are registered.
 *
 * Not thread-safe. Registration is rejected while the map is attached for signal delivery.
 *
 * @param [in] map
 * @param [in] sig_id
 * @param [in] gpu_addr
 * Pointer to one doca_gpu_verbs_signal_value_t in GPU memory.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received, sig_id is out of range, or
 *   gpu_addr is not device memory.
 * - DOCA_ERROR_IN_USE - if sig_id or gpu_addr is already registered, or the map is attached for
 *   signal delivery.
 * - DOCA_ERROR_NO_MEMORY - if the entry could not be recorded.
 * - DOCA_ERROR_DRIVER - if the slot could not be zeroed, or its page could not be pinned and
 *   mapped for the host.
 */
doca_error_t doca_gpu_verbs_signal_register(struct doca_gpu_verbs_signal_map *map, uint32_t sig_id,
                                            doca_gpu_verbs_signal_value_t *gpu_addr);

/**
 * Unregister a signal id.
 *
 * The page containing the slot is unpinned when its last registered signal is removed. The
 * application remains responsible for freeing GPU allocations, but must wait until
 * doca_gpu_verbs_signal_map_destroy() succeeds.
 *
 * Not thread-safe. Unregistration is rejected while the map is attached for signal delivery.
 *
 * @param [in] map
 * @param [in] sig_id
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 * - DOCA_ERROR_IN_USE - if the map is attached for signal delivery.
 * - DOCA_ERROR_NOT_FOUND - if sig_id is not registered.
 * - DOCA_ERROR_UNEXPECTED - if the signal map's internal slot bookkeeping is inconsistent.
 */
doca_error_t doca_gpu_verbs_signal_unregister(struct doca_gpu_verbs_signal_map *map,
                                              uint32_t sig_id);

/**
 * Return a registered signal's running total to zero.
 *
 * Fences the write itself, so the reset becomes visible to the GPU without waiting on the next
 * delivery batch. The caller must ensure that no signal delivery for this id is in progress;
 * otherwise reset and delivery may race.
 *
 * @param [in] map
 * @param [in] sig_id
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 * - DOCA_ERROR_NOT_FOUND - if sig_id is not registered.
 */
doca_error_t doca_gpu_verbs_signal_reset(struct doca_gpu_verbs_signal_map *map, uint32_t sig_id);

/**
 * Destroy an empty signal map.
 *
 * Every signal must be unregistered first: the map refuses to tear down while ids are still
 * registered, rather than freeing memory the application may still be holding.
 *
 * Destruction is rejected while the map is attached for signal delivery.
 *
 * @param [in] map
 * Map to destroy. Must have no registered signals left.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 * - DOCA_ERROR_IN_USE - if the map is attached for signal delivery.
 * - DOCA_ERROR_BAD_STATE - if any signal is still registered.
 */
doca_error_t doca_gpu_verbs_signal_map_destroy(struct doca_gpu_verbs_signal_map *map);

#ifdef __cplusplus
}
#endif

#endif /* DOCA_GPUNETIO_SIGNAL_H */
