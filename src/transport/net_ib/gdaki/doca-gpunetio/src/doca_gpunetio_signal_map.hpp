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
    bool attached_to_service;

    uint64_t dropped_signals;
    bool dropped_op_logged;
    bool dropped_unregistered_logged;
    bool dropped_mapping_logged;
    bool wrapped_logged;

    bool data_direct_mode;
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
                                   bool need_mcst, bool use_data_direct);

/**
 * Transfer one delivery owner's attachment from old_map to new_map.
 *
 * The caller must exclude its delivery thread while this runs. Attachment makes a map immutable
 * through the public register, unregister, and destroy APIs.
 */
doca_error_t priv_signal_map_replace_attachment(struct doca_gpu_verbs_signal_map *old_map,
                                                struct doca_gpu_verbs_signal_map *new_map);

#endif /* DOCA_GPUNETIO_SIGNAL_MAP_HPP */
