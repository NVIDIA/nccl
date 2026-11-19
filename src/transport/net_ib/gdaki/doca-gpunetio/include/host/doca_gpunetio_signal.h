/**
 * @file doca_gpunetio_signal.h
 * @brief A header file for the doca_gpunetio rank-local signal map
 *
 * A signal map resolves the signal id carried in the immediate of an
 * RDMA_WRITE_WITH_IMM into an 8-byte slot in GPU memory, and adds the signal value to it.
 *
 * None of these calls are thread-safe, and delivery assumes a single writer.
 *
 * Delivery - doca_gpu_verbs_rq_progress() or a service given the map with
 * doca_gpu_verbs_service_set_signal_map() - reads the map without a lock. The caller owns the
 * map's lifecycle: it must not register, unregister, or destroy while a signal may be delivered
 * into the map.
 *
 * Delivery keeps each signal's running total (its shadow) on the host, and is its only writer. The
 * GPU resets a signal with doca_gpu_dev_verbs_signal_reset() and the queue returned by
 * doca_gpu_verbs_signal_map_get_gpu_reset_queue(): it zeroes the slot and waits for delivery to
 * zero the shadow. Either reset requires that no update to the signal is in flight. Between resets
 * a signal must be updated by only one kind of operation: immediate adds, immediate sets, or NIC
 * atomics. A signal that never receives an immediate update has no shadow to zero: the GPU may
 * reset it with an empty queue, which needs no map and never waits on the CPU.
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
 * Slots arrive at registration. The map allocates only its device reset queue: one 64 KiB GPU page,
 * mapped through GDRCopy, and a small pinned host buffer.
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
 * - DOCA_ERROR_DRIVER - if the current CUDA device or its attributes could not be queried.
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
 * the calling host thread. gpu_addr must be device memory allocated on that device. The library
 * does not validate the memory type or device of gpu_addr.
 *
 * Each id must use a unique slot address. Registering the same address for more than one id is
 * rejected.
 *
 * The application retains ownership of gpu_addr. It must keep every signal allocation alive until
 * doca_gpu_verbs_signal_map_destroy() succeeds, because separate CUDA allocations can share a
 * pinned 64 KiB page. The library never frees signal memory. The application must not issue CUDA
 * memory operations against the allocation while any of its slots are registered.
 *
 * The id can be delivered to as soon as this returns.
 *
 * Not thread-safe. Must not be called while a signal may be delivered into the map.
 *
 * @param [in] map
 * @param [in] sig_id
 * @param [in] gpu_addr
 * Pointer to one doca_gpu_verbs_signal_value_t in GPU memory.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received or sig_id is out of range.
 * - DOCA_ERROR_IN_USE - if sig_id or gpu_addr is already registered.
 * - DOCA_ERROR_NO_MEMORY - if the entry could not be allocated.
 * - DOCA_ERROR_DRIVER - if the slot could not be zeroed, or its page could not be pinned and
 *   mapped for the host.
 */
doca_error_t doca_gpu_verbs_signal_register(struct doca_gpu_verbs_signal_map *map, uint32_t sig_id,
                                            doca_gpu_verbs_signal_value_t *gpu_addr);

/**
 * Bind a contiguous range of signal ids to a contiguous array of slots in one allocation.
 *
 * Registration is all-or-nothing. If any id or slot in the range is already registered,
 * nothing is registered.
 *
 * The ownership and CUDA context rules of doca_gpu_verbs_signal_register() apply to the whole
 * array of (max_sig_id - min_sig_id + 1) slots.
 *
 * The ids can be delivered to as soon as this returns.
 *
 * Not thread-safe. Must not be called while a signal may be delivered into the map.
 *
 * @param [in] map
 * @param [in] min_sig_id
 * First signal id of the range.
 * @param [in] max_sig_id
 * Last signal id of the range, inclusive. Must be less than DOCA_GPUNETIO_VERBS_SIGNAL_MAX_SIGNALS.
 * @param [in] gpu_addr
 * Pointer to (max_sig_id - min_sig_id + 1) consecutive doca_gpu_verbs_signal_value_t in GPU memory.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received, min_sig_id is greater than
 *   max_sig_id, or max_sig_id is out of range.
 * - DOCA_ERROR_IN_USE - if any id or slot in the range is already registered.
 * - DOCA_ERROR_NO_MEMORY - if the entries could not be allocated.
 * - DOCA_ERROR_DRIVER - if the slots could not be zeroed, or a page could not be pinned and
 *   mapped for the host.
 */
doca_error_t doca_gpu_verbs_signal_register_range(struct doca_gpu_verbs_signal_map *map,
                                                  uint32_t min_sig_id, uint32_t max_sig_id,
                                                  doca_gpu_verbs_signal_value_t *gpu_addr);

/**
 * Unregister a signal id.
 *
 * The page containing the slot is unpinned when its last registered signal is removed. The
 * application remains responsible for freeing GPU allocations, but must wait until
 * doca_gpu_verbs_signal_map_destroy() succeeds.
 *
 * Signals for sig_id that arrive after this returns are dropped and counted as unregistered.
 *
 * Not thread-safe. The caller must stop all delivery into the map before calling this: delivery
 * reads the entry being freed without a lock, so a concurrent delivery is a use-after-free.
 *
 * @param [in] map
 * @param [in] sig_id
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 * - DOCA_ERROR_NOT_FOUND - if sig_id is not registered.
 * - DOCA_ERROR_UNEXPECTED - if the signal map's internal slot bookkeeping is inconsistent.
 */
doca_error_t doca_gpu_verbs_signal_unregister(struct doca_gpu_verbs_signal_map *map,
                                              uint32_t sig_id);

/**
 * Unregister a contiguous range of signal ids.
 *
 * Every id in the range must be registered. If any id is missing, nothing is unregistered.
 *
 * Not thread-safe. The delivery rules of doca_gpu_verbs_signal_unregister() apply to every id in
 * the range.
 *
 * @param [in] map
 * @param [in] min_sig_id
 * First signal id of the range.
 * @param [in] max_sig_id
 * Last signal id of the range, inclusive. Must be less than DOCA_GPUNETIO_VERBS_SIGNAL_MAX_SIGNALS.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received, min_sig_id is greater than
 *   max_sig_id, or max_sig_id is out of range.
 * - DOCA_ERROR_NOT_FOUND - if any id in the range is not registered.
 * - DOCA_ERROR_UNEXPECTED - if the signal map's internal slot bookkeeping is inconsistent. Every
 *   id in the range is still unregistered.
 */
doca_error_t doca_gpu_verbs_signal_unregister_range(struct doca_gpu_verbs_signal_map *map,
                                                    uint32_t min_sig_id, uint32_t max_sig_id);

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
 * Get the GPU view of the map's device reset queue, so the GPU can reset any of the map's signals
 * with doca_gpu_dev_verbs_signal_reset(). Its pointers are for device code only; the host must not
 * dereference them.
 *
 * The library owns the queue. It stays valid until the map is destroyed. A device reset blocks
 * until the map is progressed, with no timeout, and every device reset must have returned before
 * the map is destroyed.
 *
 * Allowed while the map is attached.
 *
 * @param [in] map
 * @param [out] out_queue
 * The queue, to pass by value to the kernel.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
doca_error_t doca_gpu_verbs_signal_map_get_gpu_reset_queue(
    struct doca_gpu_verbs_signal_map *map, struct doca_gpu_dev_verbs_signal_reset_queue *out_queue);

/**
 * Destroy an empty signal map.
 *
 * Every signal must be unregistered first: the map refuses to tear down while ids are still
 * registered, rather than freeing memory the application may still be holding.
 *
 * The caller must stop all delivery into the map and remove it from any service with
 * doca_gpu_verbs_service_set_signal_map(service, NULL) or doca_gpu_verbs_destroy_service() first.
 * Delivering into a destroyed map is a use-after-free.
 *
 * No doca_gpu_dev_verbs_signal_reset() may be waiting on the map: destroy frees the reset queue the
 * GPU polls. Every device reset must have returned, for example because the kernels that reset
 * signals have completed.
 *
 * Destroy serves any device resets already posted to the queue, and refuses to tear down if a
 * device reset has taken a ticket but not yet posted its request. This is a best-effort check, not
 * a guarantee: destroy cannot tell whether a served reset has seen its acknowledgement and
 * returned, and cannot see a reset that starts after the check. The caller must wait for every
 * kernel that may reset the map's signals to complete before calling destroy.
 *
 * @param [in] map
 * Map to destroy. Must have no registered signals left.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 * - DOCA_ERROR_BAD_STATE - if any signal is still registered, or a device reset has taken a ticket
 *   but not yet posted its request.
 */
doca_error_t doca_gpu_verbs_signal_map_destroy(struct doca_gpu_verbs_signal_map *map);

/**
 * Drain completed RDMA_WRITE_WITH_IMM receives from a QP's RQ into a signal map.
 *
 * Serves the map's pending device resets from doca_gpu_dev_verbs_signal_reset(), then polls up
 * to 64 CQEs from the QP's receive CQ and applies each immediate to the map. It neither
 * fences nor writes the RQ doorbell record: after each sweep in which any call set out_progressed,
 * call doca_gpu_verbs_rq_commit_fence(), doca_gpu_verbs_rq_commit_dbr() on each QP of the sweep,
 * then doca_gpu_verbs_rq_commit_fence() again. Until then the signal values may not be visible to
 * the GPU and the NIC does not get the consumed receive credits back, so the sender eventually
 * stalls in RNR retries.
 *
 * The QP must not be monitored by a service. The caller owns the QP's lifetime and must stop
 * calling this before unexporting, resetting, or destroying it.
 *
 * After an RQ error the RQ is stopped, and every later call returns the same error.
 * DOCA_ERROR_SKIPPED is not an RQ error: the CQEs were consumed and the RQ keeps running.
 *
 * Delivery reads the map without a lock. Only one thread at a time may call this for a given map
 * or QP, and the map must not be registered into, unregistered from, or destroyed while this
 * runs.
 *
 * @param [in] qp
 * QP whose RQ is drained.
 * @param [in] map
 * Map that receives the signals.
 * @param [out] out_progressed
 * Optional. Set to true if at least one CQE was consumed or one device reset was served.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received or the QP has no RQ.
 * - DOCA_ERROR_NOT_SUPPORTED - if the QP is data-direct and the map has no forced-PCIe mappings.
 * - DOCA_ERROR_SKIPPED - if at least one signal could not be delivered and was dropped.
 * - DOCA_ERROR_IO_FAILED - if a CQE reported a responder error.
 * - DOCA_ERROR_UNEXPECTED - if a CQE has an opcode other than RDMA_WRITE_WITH_IMM.
 */
doca_error_t doca_gpu_verbs_rq_progress(struct doca_gpu_verbs_qp *qp,
                                        struct doca_gpu_verbs_signal_map *map,
                                        bool *out_progressed);

/**
 * Give a service the map to deliver signals into.
 *
 * The map is rank-local, so it is set once on the service rather than once per QP. The service
 * delivers into the map until it is replaced or the service is destroyed. Until then the caller
 * must not register into, unregister from, or destroy the map, and must not deliver into it from
 * any other thread.
 *
 * @param [in] service
 * Service object that will deliver into the map.
 * @param [in] map
 * Map to deliver into, or NULL to clear it.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 * - DOCA_ERROR_BAD_STATE - if any RQ is already being monitored.
 */
doca_error_t doca_gpu_verbs_service_set_signal_map(doca_gpu_verbs_service_t service,
                                                   struct doca_gpu_verbs_signal_map *map);

#ifdef __cplusplus
}
#endif

#endif /* DOCA_GPUNETIO_SIGNAL_H */
