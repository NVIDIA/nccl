#ifndef DOCA_GPUNETIO_SIGNAL_MAP_HPP
#define DOCA_GPUNETIO_SIGNAL_MAP_HPP

#include <stddef.h>
#include <stdint.h>

#include <unordered_map>
#include <unordered_set>

#include "host/doca_gpunetio_signal.h"

#define DOCA_GPUNETIO_VERBS_SIGNAL_PAGE_SHIFT 16
#define DOCA_GPUNETIO_VERBS_SIGNAL_PAGE_SIZE (1UL << DOCA_GPUNETIO_VERBS_SIGNAL_PAGE_SHIFT)
#define DOCA_GPUNETIO_VERBS_SIGNAL_PAGE_OFFSET (DOCA_GPUNETIO_VERBS_SIGNAL_PAGE_SIZE - 1UL)
#define DOCA_GPUNETIO_VERBS_SIGNAL_LOOKUP_BLOCK_SHIFT 8
#define DOCA_GPUNETIO_VERBS_SIGNAL_LOOKUP_BLOCK_SIZE \
    (1U << DOCA_GPUNETIO_VERBS_SIGNAL_LOOKUP_BLOCK_SHIFT)
#define DOCA_GPUNETIO_VERBS_SIGNAL_LOOKUP_BLOCK_COUNT \
    (DOCA_GPUNETIO_VERBS_SIGNAL_MAX_SIGNALS / DOCA_GPUNETIO_VERBS_SIGNAL_LOOKUP_BLOCK_SIZE)

struct doca_gpu_verbs_signal_page {
    uintptr_t gpu_base;
    void *gdr_legacy_mh;
    void *gdr_data_direct_mh;
    void *cpu_base_legacy;
    void *cpu_base_data_direct;
    size_t mapped_size;
    uint32_t refcount;
};

/* Offset of the acks in the reset queue's GPU page. The tail sits alone in the first 128 bytes. */
#define DOCA_GPUNETIO_VERBS_SIGNAL_RESET_ACKS_OFFSET 128

struct doca_gpu_verbs_signal_entry {
    volatile doca_gpu_verbs_signal_value_t *cpu_alias_legacy;
    volatile doca_gpu_verbs_signal_value_t *cpu_alias_data_direct;
    doca_gpu_verbs_signal_value_t *gpu_addr;
    struct doca_gpu_verbs_signal_page *page;
    doca_gpu_verbs_signal_value_t shadow;
};

struct doca_gpu_verbs_signal_lookup_block {
    struct doca_gpu_verbs_signal_entry *entries[DOCA_GPUNETIO_VERBS_SIGNAL_LOOKUP_BLOCK_SIZE];
};

struct doca_gpu_verbs_signal_map {
    std::unordered_map<uint32_t, struct doca_gpu_verbs_signal_entry> *entries;
    std::unordered_map<uintptr_t, struct doca_gpu_verbs_signal_page *> *pages;
    std::unordered_set<uintptr_t> *registered_slots;
    struct doca_gpu_verbs_signal_lookup_block
        *lookup_blocks[DOCA_GPUNETIO_VERBS_SIGNAL_LOOKUP_BLOCK_COUNT];

    uint64_t dropped_signals;
    bool dropped_op_logged;
    bool dropped_unregistered_logged;
    bool dropped_mapping_logged;
    bool wrapped_logged;

    bool needs_mcst;
    bool data_direct_mode;

    /* Device reset queue. The GPU posts requests into pinned host memory and waits on the acks,
     * which the CPU writes through a GDRCopy mapping of the queue's GPU page. */
    uint64_t *reset_requests;
    void *reset_gpu_alloc;
    void *reset_gdr_mh;
    void *reset_cpu_base;
    volatile uint32_t *reset_acks_cpu;
    struct doca_gpu_dev_verbs_signal_reset_queue reset_queue;
    uint32_t reset_head;
};

/**
 * Decode a signal immediate and add its value to the addressed slot.
 *
 * Does not fence per signal write: the caller issues one fence after its
 * batch of CQEs. The fence is determined by the memory mapping using data direct
 * or legacy mode. Undeliverable signals are counted and consumed so they cannot
 * stall the receive queue.
 */
doca_error_t priv_signal_map_apply(struct doca_gpu_verbs_signal_map *map, uint32_t imm,
                                   bool use_data_direct);

/**
 * Serve the device resets posted to the map's reset queue: zero each requested signal's shadow,
 * then acknowledge the request so doca_gpu_dev_verbs_signal_reset() returns.
 *
 * Runs on the delivery thread, which is the shadow's only writer. Returns true if any reset was
 * served.
 */
bool priv_signal_reset_progress(struct doca_gpu_verbs_signal_map *map);

/* Whether the next device reset may have been posted. Delivery checks this on every progress call,
 * so it stays inline; priv_signal_reset_progress() makes the full check. */
static inline bool priv_signal_reset_pending(const struct doca_gpu_verbs_signal_map *map) {
    const uint32_t index = map->reset_head & (DOCA_GPUNETIO_VERBS_SIGNAL_RESET_QUEUE_SIZE - 1);
    const uint64_t request = __atomic_load_n(&map->reset_requests[index], __ATOMIC_RELAXED);
    return (uint32_t)(request >> 32) == map->reset_head + 1;
}

#endif /* DOCA_GPUNETIO_SIGNAL_MAP_HPP */
