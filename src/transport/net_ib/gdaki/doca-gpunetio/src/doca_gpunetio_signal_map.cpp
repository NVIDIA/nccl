#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "doca_gpunetio_gdrcopy.h"
#include "doca_gpunetio_signal_map.hpp"
#include "doca_internal.hpp"
#include "host/doca_gpunetio.h"
#include "host/doca_gpunetio_signal.h"

/* Delivery never takes this lock: the caller must not mutate a map while signals are delivered
 * into it. */
static pthread_mutex_t signal_map_lifecycle_lock = PTHREAD_MUTEX_INITIALIZER;

class signal_map_lifecycle_guard {
   public:
    signal_map_lifecycle_guard() { pthread_mutex_lock(&signal_map_lifecycle_lock); }
    ~signal_map_lifecycle_guard() { pthread_mutex_unlock(&signal_map_lifecycle_lock); }

    signal_map_lifecycle_guard(const signal_map_lifecycle_guard &) = delete;
    signal_map_lifecycle_guard &operator=(const signal_map_lifecycle_guard &) = delete;
};

static void priv_signal_lookup_destroy(struct doca_gpu_verbs_signal_map *map) {
    for (size_t i = 0; i < DOCA_GPUNETIO_VERBS_SIGNAL_LOOKUP_BLOCK_COUNT; ++i) {
        free(map->lookup_blocks[i]);
        map->lookup_blocks[i] = nullptr;
    }
}

/* Blocks are only freed by priv_signal_lookup_destroy(), so delivery never reads a freed block. */
static doca_error_t priv_signal_lookup_slot(struct doca_gpu_verbs_signal_map *map, uint32_t sig_id,
                                            struct doca_gpu_verbs_signal_entry ***out_slot) {
    const uint32_t block_index = sig_id >> DOCA_GPUNETIO_VERBS_SIGNAL_LOOKUP_BLOCK_SHIFT;
    const uint32_t entry_index = sig_id & (DOCA_GPUNETIO_VERBS_SIGNAL_LOOKUP_BLOCK_SIZE - 1);

    if (map->lookup_blocks[block_index] == nullptr) {
        map->lookup_blocks[block_index] = (struct doca_gpu_verbs_signal_lookup_block *)calloc(
            1, sizeof(struct doca_gpu_verbs_signal_lookup_block));
        if (map->lookup_blocks[block_index] == nullptr) {
            DOCA_LOG(LOG_ERR, "Failed to allocate signal lookup block %u", block_index);
            return DOCA_ERROR_NO_MEMORY;
        }
    }
    *out_slot = &map->lookup_blocks[block_index]->entries[entry_index];
    return DOCA_SUCCESS;
}

static doca_error_t priv_signal_page_acquire(struct doca_gpu_verbs_signal_map *map,
                                             uintptr_t page_base,
                                             struct doca_gpu_verbs_signal_page **out_page) {
    auto it = map->pages->find(page_base);
    if (it != map->pages->end()) {
        it->second->refcount++;
        *out_page = it->second;
        return DOCA_SUCCESS;
    }

    auto *page =
        (struct doca_gpu_verbs_signal_page *)calloc(1, sizeof(struct doca_gpu_verbs_signal_page));
    if (page == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to allocate memory for a signal page");
        return DOCA_ERROR_NO_MEMORY;
    }

    void *gdr_legacy_mh = nullptr;
    void *cpu_base_legacy = nullptr;
    void *gdr_data_direct_mh = nullptr;
    void *cpu_base_data_direct = nullptr;
    int ret =
        doca_gpu_gdrcopy_create_mapping((void *)page_base, DOCA_GPUNETIO_VERBS_SIGNAL_PAGE_SIZE,
                                        false, &gdr_legacy_mh, &cpu_base_legacy);
    if (ret != 0) {
        DOCA_LOG(LOG_ERR, "Failed to map GPU page at %lx through the legacy path, error %d",
                 page_base, ret);
        free(page);
        return DOCA_ERROR_DRIVER;
    }

    if (map->data_direct_mode) {
        ret =
            doca_gpu_gdrcopy_create_mapping((void *)page_base, DOCA_GPUNETIO_VERBS_SIGNAL_PAGE_SIZE,
                                            true, &gdr_data_direct_mh, &cpu_base_data_direct);
        if (ret != 0) {
            DOCA_LOG(LOG_ERR,
                     "Failed to map GPU page at %lx through the forced-PCIe path, error %d",
                     page_base, ret);
            doca_gpu_gdrcopy_destroy_mapping(gdr_legacy_mh, cpu_base_legacy,
                                             DOCA_GPUNETIO_VERBS_SIGNAL_PAGE_SIZE);
            free(page);
            return DOCA_ERROR_DRIVER;
        }
    }

    page->gpu_base = page_base;
    page->gdr_legacy_mh = gdr_legacy_mh;
    page->gdr_data_direct_mh = gdr_data_direct_mh;
    page->cpu_base_legacy = cpu_base_legacy;
    page->cpu_base_data_direct = cpu_base_data_direct;
    page->mapped_size = DOCA_GPUNETIO_VERBS_SIGNAL_PAGE_SIZE;
    page->refcount = 1;

    try {
        map->pages->insert({page_base, page});
    } catch (...) {
        DOCA_LOG(LOG_ERR, "Failed to record the signal page at %lx", page_base);
        if (gdr_data_direct_mh != nullptr)
            doca_gpu_gdrcopy_destroy_mapping(gdr_data_direct_mh, cpu_base_data_direct,
                                             page->mapped_size);
        doca_gpu_gdrcopy_destroy_mapping(gdr_legacy_mh, cpu_base_legacy, page->mapped_size);
        free(page);
        return DOCA_ERROR_NO_MEMORY;
    }

    *out_page = page;
    return DOCA_SUCCESS;
}

static doca_error_t priv_signal_page_release(struct doca_gpu_verbs_signal_map *map,
                                             struct doca_gpu_verbs_signal_page *page) {
    if (--page->refcount != 0) return DOCA_SUCCESS;

    if (page->gdr_data_direct_mh != nullptr)
        doca_gpu_gdrcopy_destroy_mapping(page->gdr_data_direct_mh, page->cpu_base_data_direct,
                                         page->mapped_size);
    doca_gpu_gdrcopy_destroy_mapping(page->gdr_legacy_mh, page->cpu_base_legacy, page->mapped_size);
    map->pages->erase(page->gpu_base);
    free(page);
    return DOCA_SUCCESS;
}

static doca_error_t priv_signal_slot_unregister(struct doca_gpu_verbs_signal_map *map,
                                                uintptr_t gpu_addr) {
    if (map->registered_slots->erase(gpu_addr) == 0) {
        DOCA_LOG(LOG_ERR, "Signal slot at %lx is missing from the registration set", gpu_addr);
        return DOCA_ERROR_UNEXPECTED;
    }
    return DOCA_SUCCESS;
}

static doca_error_t priv_signal_device_needs_mcst(bool *needs_mcst) {
    int device = 0;
    cudaError_t cuda_status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaGetDevice(&device));
    if (cuda_status != cudaSuccess) {
        DOCA_LOG(LOG_ERR, "Failed to get the current CUDA device for the signal map");
        return DOCA_ERROR_DRIVER;
    }

    int order = 0;
    cuda_status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(
        cudaDeviceGetAttribute(&order, cudaDevAttrGPUDirectRDMAWritesOrdering, device));
    if (cuda_status != cudaSuccess) {
        DOCA_LOG(LOG_ERR, "Failed to check device attribute for order");
        return DOCA_ERROR_DRIVER;
    }
    *needs_mcst = order == 0;
    return DOCA_SUCCESS;
}

static void priv_signal_reset_queue_destroy(struct doca_gpu_verbs_signal_map *map) {
    if (map->reset_gdr_mh != nullptr)
        doca_gpu_gdrcopy_destroy_mapping(map->reset_gdr_mh, map->reset_cpu_base,
                                         DOCA_GPUNETIO_VERBS_SIGNAL_PAGE_SIZE);
    if (map->reset_gpu_alloc != nullptr) cudaFree(map->reset_gpu_alloc);
    if (map->reset_requests != nullptr) cudaFreeHost(map->reset_requests);
    map->reset_gdr_mh = nullptr;
    map->reset_gpu_alloc = nullptr;
    map->reset_requests = nullptr;
}

static doca_error_t priv_signal_reset_queue_create(struct doca_gpu_verbs_signal_map *map) {
    const size_t requests_size = DOCA_GPUNETIO_VERBS_SIGNAL_RESET_QUEUE_SIZE * sizeof(uint64_t);
    void *requests = nullptr;
    void *gpu_requests = nullptr;
    void *gpu_alloc = nullptr;
    uintptr_t page_base = 0;
    int ret = 0;

    cudaError_t cuda_status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(
        cudaHostAlloc(&requests, requests_size, cudaHostAllocMapped | cudaHostAllocPortable));
    if (cuda_status != cudaSuccess) {
        DOCA_LOG(LOG_ERR, "Failed to allocate the signal reset queue in pinned host memory");
        return DOCA_ERROR_NO_MEMORY;
    }
    memset(requests, 0, requests_size);
    map->reset_requests = (uint64_t *)requests;

    cuda_status =
        DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaHostGetDevicePointer(&gpu_requests, requests, 0));
    if (cuda_status != cudaSuccess) goto driver_error;

    /* GDRCopy maps whole 64 KiB GPU pages, so over-allocate and align. */
    cuda_status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(
        cudaMalloc(&gpu_alloc, DOCA_GPUNETIO_VERBS_SIGNAL_PAGE_SIZE * 2));
    if (cuda_status != cudaSuccess) {
        DOCA_LOG(LOG_ERR, "Failed to allocate GPU memory for the signal reset queue");
        priv_signal_reset_queue_destroy(map);
        return DOCA_ERROR_NO_MEMORY;
    }
    map->reset_gpu_alloc = gpu_alloc;
    page_base = ((uintptr_t)gpu_alloc + DOCA_GPUNETIO_VERBS_SIGNAL_PAGE_SIZE - 1) &
                ~(uintptr_t)DOCA_GPUNETIO_VERBS_SIGNAL_PAGE_OFFSET;

    cuda_status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(
        cudaMemset((void *)page_base, 0, DOCA_GPUNETIO_VERBS_SIGNAL_PAGE_SIZE));
    if (cuda_status != cudaSuccess) goto driver_error;
    cuda_status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaStreamSynchronize(0));
    if (cuda_status != cudaSuccess) goto driver_error;

    ret = doca_gpu_gdrcopy_create_mapping((void *)page_base, DOCA_GPUNETIO_VERBS_SIGNAL_PAGE_SIZE,
                                          false, &map->reset_gdr_mh, &map->reset_cpu_base);
    if (ret != 0) {
        DOCA_LOG(LOG_ERR, "Failed to map the signal reset queue page at %lx, error %d", page_base,
                 ret);
        map->reset_gdr_mh = nullptr;
        goto driver_error;
    }

    map->reset_acks_cpu = (volatile uint32_t *)((uint8_t *)map->reset_cpu_base +
                                                DOCA_GPUNETIO_VERBS_SIGNAL_RESET_ACKS_OFFSET);
    map->reset_queue.requests = (uint64_t *)gpu_requests;
    map->reset_queue.tail = (uint32_t *)page_base;
    map->reset_queue.acks = (uint32_t *)(page_base + DOCA_GPUNETIO_VERBS_SIGNAL_RESET_ACKS_OFFSET);
    map->reset_head = 0;
    return DOCA_SUCCESS;

driver_error:
    priv_signal_reset_queue_destroy(map);
    return DOCA_ERROR_DRIVER;
}

static struct doca_gpu_verbs_signal_entry *priv_signal_lookup(
    const struct doca_gpu_verbs_signal_map *map, uint32_t sig_id) {
    if (sig_id >= DOCA_GPUNETIO_VERBS_SIGNAL_MAX_SIGNALS) return nullptr;
    const struct doca_gpu_verbs_signal_lookup_block *lookup_block =
        map->lookup_blocks[sig_id >> DOCA_GPUNETIO_VERBS_SIGNAL_LOOKUP_BLOCK_SHIFT];
    if (lookup_block == nullptr) return nullptr;
    return lookup_block->entries[sig_id & (DOCA_GPUNETIO_VERBS_SIGNAL_LOOKUP_BLOCK_SIZE - 1)];
}

/* The caller holds the lifecycle lock and has already checked that sig_id and the slot are free. */
static doca_error_t priv_signal_entry_add(struct doca_gpu_verbs_signal_map *map, uint32_t sig_id,
                                          doca_gpu_verbs_signal_value_t *gpu_addr) {
    const uintptr_t slot_addr = (uintptr_t)gpu_addr;
    const uintptr_t page_base = slot_addr & ~(uintptr_t)DOCA_GPUNETIO_VERBS_SIGNAL_PAGE_OFFSET;
    const size_t page_offset = slot_addr & DOCA_GPUNETIO_VERBS_SIGNAL_PAGE_OFFSET;
    struct doca_gpu_verbs_signal_entry **lookup_slot = nullptr;
    doca_error_t status = priv_signal_lookup_slot(map, sig_id, &lookup_slot);
    if (status != DOCA_SUCCESS) return status;

    struct doca_gpu_verbs_signal_page *page = nullptr;
    status = priv_signal_page_acquire(map, page_base, &page);
    if (status != DOCA_SUCCESS) return status;

    struct doca_gpu_verbs_signal_entry entry = {};
    entry.cpu_alias_legacy =
        (volatile doca_gpu_verbs_signal_value_t *)((uint8_t *)page->cpu_base_legacy + page_offset);
    if (page->cpu_base_data_direct != nullptr)
        entry.cpu_alias_data_direct =
            (volatile doca_gpu_verbs_signal_value_t *)((uint8_t *)page->cpu_base_data_direct +
                                                       page_offset);
    entry.gpu_addr = gpu_addr;
    entry.page = page;

    try {
        map->registered_slots->insert(slot_addr);
    } catch (...) {
        priv_signal_page_release(map, page);
        return DOCA_ERROR_NO_MEMORY;
    }
    try {
        /* unordered_map keeps element addresses stable across rehash, so the lookup stays valid. */
        *lookup_slot = &map->entries->insert({sig_id, entry}).first->second;
    } catch (...) {
        priv_signal_slot_unregister(map, slot_addr);
        priv_signal_page_release(map, page);
        return DOCA_ERROR_NO_MEMORY;
    }
    return DOCA_SUCCESS;
}

static doca_error_t priv_signal_entry_remove(struct doca_gpu_verbs_signal_map *map,
                                             uint32_t sig_id) {
    auto it = map->entries->find(sig_id);
    if (it == map->entries->end()) return DOCA_ERROR_NOT_FOUND;

    doca_gpu_verbs_signal_value_t *gpu_addr = it->second.gpu_addr;
    struct doca_gpu_verbs_signal_page *page = it->second.page;
    doca_error_t slot_status = priv_signal_slot_unregister(map, (uintptr_t)gpu_addr);
    map->lookup_blocks[sig_id >> DOCA_GPUNETIO_VERBS_SIGNAL_LOOKUP_BLOCK_SHIFT]
        ->entries[sig_id & (DOCA_GPUNETIO_VERBS_SIGNAL_LOOKUP_BLOCK_SIZE - 1)] = nullptr;
    map->entries->erase(it);
    doca_error_t page_status = priv_signal_page_release(map, page);
    if (slot_status != DOCA_SUCCESS) return slot_status;
    return page_status;
}

static bool priv_signal_map_probe_data_direct() {
    void *allocation = nullptr;
    void *gdr_mh = nullptr;
    void *cpu_base = nullptr;
    bool supported = false;
    uintptr_t page_base = 0;
    cudaError_t cuda_status = cudaSuccess;
    int ret = 0;

    cuda_status = cudaMalloc(&allocation, DOCA_GPUNETIO_VERBS_SIGNAL_PAGE_SIZE * 2);
    if (cuda_status != cudaSuccess) {
        DOCA_LOG(LOG_INFO, "Could not allocate GPU memory for Data Direct probe: %s",
                 cudaGetErrorString(cuda_status));
        goto out;
    }

    page_base = ((uintptr_t)allocation + DOCA_GPUNETIO_VERBS_SIGNAL_PAGE_SIZE - 1) &
                ~(uintptr_t)DOCA_GPUNETIO_VERBS_SIGNAL_PAGE_OFFSET;
    ret = doca_gpu_gdrcopy_create_mapping((void *)page_base, DOCA_GPUNETIO_VERBS_SIGNAL_PAGE_SIZE,
                                          true, &gdr_mh, &cpu_base);
    if (ret != 0) {
        DOCA_LOG(LOG_INFO, "Data Direct mapping probe failed, error %d", ret);
        goto out;
    }
    supported = true;

out:
    if (gdr_mh != nullptr)
        doca_gpu_gdrcopy_destroy_mapping(gdr_mh, cpu_base, DOCA_GPUNETIO_VERBS_SIGNAL_PAGE_SIZE);
    if (allocation != nullptr) cudaFree(allocation);
    return supported;
}

doca_error_t doca_gpu_verbs_signal_map_create(struct doca_gpu_verbs_signal_map **out_map) {
    if (out_map == nullptr) return DOCA_ERROR_INVALID_VALUE;

    if (!doca_gpu_gdrcopy_is_supported()) {
        DOCA_LOG(LOG_ERR,
                 "Signal map is not supported without GDRCopy: the signal slots must stay in "
                 "GPU memory while remaining writable by the host");
        return DOCA_ERROR_NOT_SUPPORTED;
    }

    bool needs_mcst = false;
    doca_error_t status = priv_signal_device_needs_mcst(&needs_mcst);
    if (status != DOCA_SUCCESS) return status;

    auto *map =
        (struct doca_gpu_verbs_signal_map *)calloc(1, sizeof(struct doca_gpu_verbs_signal_map));
    if (map == nullptr) return DOCA_ERROR_NO_MEMORY;
    map->needs_mcst = needs_mcst;
    map->data_direct_mode = priv_signal_map_probe_data_direct();
    try {
        map->entries = new std::unordered_map<uint32_t, struct doca_gpu_verbs_signal_entry>();
        map->pages = new std::unordered_map<uintptr_t, struct doca_gpu_verbs_signal_page *>();
        map->registered_slots = new std::unordered_set<uintptr_t>();
    } catch (...) {
        delete map->entries;
        delete map->pages;
        delete map->registered_slots;
        free(map);
        return DOCA_ERROR_NO_MEMORY;
    }

    status = priv_signal_reset_queue_create(map);
    if (status != DOCA_SUCCESS) {
        delete map->entries;
        delete map->pages;
        delete map->registered_slots;
        free(map);
        return status;
    }

    *out_map = map;
    return DOCA_SUCCESS;
}

doca_error_t doca_gpu_verbs_signal_register(struct doca_gpu_verbs_signal_map *map, uint32_t sig_id,
                                            doca_gpu_verbs_signal_value_t *gpu_addr) {
    if (map == nullptr || gpu_addr == nullptr) return DOCA_ERROR_INVALID_VALUE;
    if (sig_id >= DOCA_GPUNETIO_VERBS_SIGNAL_MAX_SIGNALS) return DOCA_ERROR_INVALID_VALUE;
    if (((uintptr_t)gpu_addr & (DOCA_GPUNETIO_VERBS_SIGNAL_SLOT_SIZE - 1)) != 0)
        return DOCA_ERROR_INVALID_VALUE;

    signal_map_lifecycle_guard lifecycle_guard;
    if (map->entries->count(sig_id) != 0 || map->registered_slots->count((uintptr_t)gpu_addr) != 0)
        return DOCA_ERROR_IN_USE;

    cudaError_t cuda_status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(
        cudaMemset(gpu_addr, 0, DOCA_GPUNETIO_VERBS_SIGNAL_SLOT_SIZE));
    if (cuda_status != cudaSuccess) return DOCA_ERROR_DRIVER;
    cuda_status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaStreamSynchronize(0));
    if (cuda_status != cudaSuccess) return DOCA_ERROR_DRIVER;

    return priv_signal_entry_add(map, sig_id, gpu_addr);
}

doca_error_t doca_gpu_verbs_signal_register_range(struct doca_gpu_verbs_signal_map *map,
                                                  uint32_t min_sig_id, uint32_t max_sig_id,
                                                  doca_gpu_verbs_signal_value_t *gpu_addr) {
    if (map == nullptr || gpu_addr == nullptr) return DOCA_ERROR_INVALID_VALUE;
    if (min_sig_id > max_sig_id || max_sig_id >= DOCA_GPUNETIO_VERBS_SIGNAL_MAX_SIGNALS)
        return DOCA_ERROR_INVALID_VALUE;
    if (((uintptr_t)gpu_addr & (DOCA_GPUNETIO_VERBS_SIGNAL_SLOT_SIZE - 1)) != 0)
        return DOCA_ERROR_INVALID_VALUE;

    const uint32_t num_signals = max_sig_id - min_sig_id + 1;
    doca_error_t status = DOCA_SUCCESS;

    signal_map_lifecycle_guard lifecycle_guard;
    for (uint32_t i = 0; i < num_signals; ++i) {
        if (map->entries->count(min_sig_id + i) != 0 ||
            map->registered_slots->count((uintptr_t)(gpu_addr + i)) != 0)
            return DOCA_ERROR_IN_USE;
    }

    try {
        map->entries->reserve(map->entries->size() + num_signals);
        map->registered_slots->reserve(map->registered_slots->size() + num_signals);
    } catch (...) {
        return DOCA_ERROR_NO_MEMORY;
    }

    cudaError_t cuda_status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(
        cudaMemset(gpu_addr, 0, (size_t)num_signals * DOCA_GPUNETIO_VERBS_SIGNAL_SLOT_SIZE));
    if (cuda_status != cudaSuccess) return DOCA_ERROR_DRIVER;
    cuda_status = DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cudaStreamSynchronize(0));
    if (cuda_status != cudaSuccess) return DOCA_ERROR_DRIVER;

    for (uint32_t i = 0; i < num_signals; ++i) {
        status = priv_signal_entry_add(map, min_sig_id + i, gpu_addr + i);
        if (status != DOCA_SUCCESS) {
            for (uint32_t j = 0; j < i; ++j) (void)priv_signal_entry_remove(map, min_sig_id + j);
            break;
        }
    }
    return status;
}

doca_error_t doca_gpu_verbs_signal_unregister(struct doca_gpu_verbs_signal_map *map,
                                              uint32_t sig_id) {
    if (map == nullptr) return DOCA_ERROR_INVALID_VALUE;
    signal_map_lifecycle_guard lifecycle_guard;
    return priv_signal_entry_remove(map, sig_id);
}

doca_error_t doca_gpu_verbs_signal_unregister_range(struct doca_gpu_verbs_signal_map *map,
                                                    uint32_t min_sig_id, uint32_t max_sig_id) {
    if (map == nullptr) return DOCA_ERROR_INVALID_VALUE;
    if (min_sig_id > max_sig_id || max_sig_id >= DOCA_GPUNETIO_VERBS_SIGNAL_MAX_SIGNALS)
        return DOCA_ERROR_INVALID_VALUE;

    const uint32_t num_signals = max_sig_id - min_sig_id + 1;
    signal_map_lifecycle_guard lifecycle_guard;
    for (uint32_t i = 0; i < num_signals; ++i) {
        if (map->entries->count(min_sig_id + i) == 0) return DOCA_ERROR_NOT_FOUND;
    }

    doca_error_t first_error = DOCA_SUCCESS;
    for (uint32_t i = 0; i < num_signals; ++i) {
        doca_error_t status = priv_signal_entry_remove(map, min_sig_id + i);
        if (status != DOCA_SUCCESS && first_error == DOCA_SUCCESS) first_error = status;
    }
    return first_error;
}

doca_error_t doca_gpu_verbs_signal_reset(struct doca_gpu_verbs_signal_map *map, uint32_t sig_id) {
    if (map == nullptr) return DOCA_ERROR_INVALID_VALUE;
    signal_map_lifecycle_guard lifecycle_guard;
    auto it = map->entries->find(sig_id);
    if (it == map->entries->end()) return DOCA_ERROR_NOT_FOUND;

    struct doca_gpu_verbs_signal_entry *entry = &it->second;
    entry->shadow = 0;
    if (map->data_direct_mode && entry->cpu_alias_data_direct != nullptr)
        *entry->cpu_alias_data_direct = 0;
    else
        *entry->cpu_alias_legacy = 0;

    if (map->data_direct_mode)
        doca_internal_memory_fence();
    else
        doca_internal_wc_store_fence();
    return DOCA_SUCCESS;
}

doca_error_t doca_gpu_verbs_signal_map_get_gpu_reset_queue(
    struct doca_gpu_verbs_signal_map *map,
    struct doca_gpu_dev_verbs_signal_reset_queue *out_queue) {
    if (map == nullptr || out_queue == nullptr) return DOCA_ERROR_INVALID_VALUE;
    *out_queue = map->reset_queue;
    return DOCA_SUCCESS;
}

bool priv_signal_reset_progress(struct doca_gpu_verbs_signal_map *map) {
    uint32_t handled = 0;
    while (handled < DOCA_GPUNETIO_VERBS_SIGNAL_RESET_QUEUE_SIZE) {
        const uint32_t ticket = map->reset_head + handled;
        const uint32_t index = ticket & (DOCA_GPUNETIO_VERBS_SIGNAL_RESET_QUEUE_SIZE - 1);
        const uint64_t request = __atomic_load_n(&map->reset_requests[index], __ATOMIC_ACQUIRE);
        const uint32_t lap_check = ((ticket >> DOCA_GPUNETIO_VERBS_SIGNAL_RESET_QUEUE_SHIFT) + 1)
                                   << DOCA_GPUNETIO_VERBS_SIGNAL_RESET_CHECK_SHIFT;
        const uint32_t low = (uint32_t)request;
        if ((uint32_t)(request >> 32) != ticket + 1 ||
            (low & ~DOCA_GPUNETIO_VERBS_SIGNAL_IMM_ID_MASK) != lap_check)
            break;

        /* Acknowledge unregistered ids too, so the GPU never waits on them. */
        struct doca_gpu_verbs_signal_entry *entry =
            priv_signal_lookup(map, low & DOCA_GPUNETIO_VERBS_SIGNAL_IMM_ID_MASK);
        if (entry != nullptr) entry->shadow = 0;
        map->reset_acks_cpu[index] = ticket + 1;
        ++handled;
    }
    if (handled == 0) return false;

    map->reset_head += handled;
    doca_internal_wc_store_fence();
    return true;
}

static bool priv_signal_reset_queue_empty(struct doca_gpu_verbs_signal_map *map) {
    (void)priv_signal_reset_progress(map);
    const uint32_t outstanding = *(const volatile uint32_t *)map->reset_cpu_base - map->reset_head;
    if (outstanding == 0) return true;
    DOCA_LOG(LOG_ERR, "Signal map has %u device resets in progress", outstanding);
    return false;
}

doca_error_t doca_gpu_verbs_signal_map_destroy(struct doca_gpu_verbs_signal_map *map) {
    if (map == nullptr) return DOCA_ERROR_INVALID_VALUE;
    signal_map_lifecycle_guard lifecycle_guard;
    if (!map->entries->empty()) return DOCA_ERROR_BAD_STATE;
    if (!priv_signal_reset_queue_empty(map)) return DOCA_ERROR_BAD_STATE;

    if (map->dropped_signals != 0)
        DOCA_LOG(LOG_WARNING, "Signal map dropped %lu signals over its lifetime",
                 map->dropped_signals);
    if (!map->pages->empty()) {
        DOCA_LOG(LOG_ERR, "Signal map has no registered signals but %zu pages are still pinned",
                 map->pages->size());
        for (auto &page_entry : *map->pages) {
            struct doca_gpu_verbs_signal_page *page = page_entry.second;
            if (page->gdr_data_direct_mh != nullptr)
                doca_gpu_gdrcopy_destroy_mapping(page->gdr_data_direct_mh,
                                                 page->cpu_base_data_direct, page->mapped_size);
            doca_gpu_gdrcopy_destroy_mapping(page->gdr_legacy_mh, page->cpu_base_legacy,
                                             page->mapped_size);
            free(page);
        }
        map->pages->clear();
    }

    priv_signal_lookup_destroy(map);
    priv_signal_reset_queue_destroy(map);
    delete map->entries;
    delete map->pages;
    delete map->registered_slots;
    free(map);
    return DOCA_SUCCESS;
}

doca_error_t priv_signal_map_apply(struct doca_gpu_verbs_signal_map *map, uint32_t imm,
                                   bool use_data_direct) {
    if (map == nullptr) {
        DOCA_LOG(LOG_ERR,
                 "Signals are arriving with no signal map - see doca_gpu_verbs_rq_progress() or "
                 "doca_gpu_verbs_service_set_signal_map()");
        return DOCA_ERROR_INVALID_VALUE;
    }

    const uint8_t sig_op =
        (imm >> DOCA_GPUNETIO_VERBS_SIGNAL_IMM_OP_SHIFT) & DOCA_GPUNETIO_VERBS_SIGNAL_IMM_OP_MASK;
    const uint32_t sig_id =
        (imm >> DOCA_GPUNETIO_VERBS_SIGNAL_IMM_ID_SHIFT) & DOCA_GPUNETIO_VERBS_SIGNAL_IMM_ID_MASK;
    const uint16_t sig_val =
        (imm >> DOCA_GPUNETIO_VERBS_SIGNAL_IMM_VAL_SHIFT) & DOCA_GPUNETIO_VERBS_SIGNAL_IMM_VAL_MASK;

    if (sig_op != DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD &&
        sig_op != DOCA_GPUNETIO_VERBS_SIGNAL_OP_SET) {
        map->dropped_signals++;
        if (!map->dropped_op_logged) {
            map->dropped_op_logged = true;
            DOCA_LOG(LOG_ERR, "Unsupported signal op %u for signal id %u, dropping it", sig_op,
                     sig_id);
        }
        return DOCA_SUCCESS;
    }

    struct doca_gpu_verbs_signal_entry *entry = priv_signal_lookup(map, sig_id);
    if (entry == nullptr) {
        map->dropped_signals++;
        if (!map->dropped_unregistered_logged) {
            map->dropped_unregistered_logged = true;
            DOCA_LOG(LOG_ERR, "Signal id %u is not registered, dropping it", sig_id);
        }
        return DOCA_SUCCESS;
    }

    volatile doca_gpu_verbs_signal_value_t *cpu_alias =
        use_data_direct ? entry->cpu_alias_data_direct : entry->cpu_alias_legacy;
    if (cpu_alias == nullptr) {
        map->dropped_signals++;
        if (!map->dropped_mapping_logged) {
            map->dropped_mapping_logged = true;
            DOCA_LOG(LOG_ERR, "Signal id %u has no %s CPU mapping, dropping it", sig_id,
                     use_data_direct ? "forced-PCIe" : "legacy");
        }
        return DOCA_SUCCESS;
    }

    if (map->needs_mcst) (void)READ_ONCE(*(volatile uint8_t *)cpu_alias);

    const doca_gpu_verbs_signal_value_t shadow = entry->shadow;
    const doca_gpu_verbs_signal_value_t new_value =
        sig_op == DOCA_GPUNETIO_VERBS_SIGNAL_OP_SET
            ? (doca_gpu_verbs_signal_value_t)sig_val
            : shadow + (doca_gpu_verbs_signal_value_t)sig_val;
    if (sig_op == DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD && new_value < shadow && !map->wrapped_logged) {
        map->wrapped_logged = true;
        DOCA_LOG(LOG_WARNING, "Signal id %u wrapped its %u-bit slot", sig_id,
                 (unsigned int)(DOCA_GPUNETIO_VERBS_SIGNAL_SLOT_SIZE * 8));
    }
    entry->shadow = new_value;
    *cpu_alias = new_value;
    return DOCA_SUCCESS;
}
