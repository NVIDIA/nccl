
#include <cassert>
#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <nccl.h>
#include <optional>
#include <pybind11/functional.h>
#include <torch/python.h>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/types.h>
#include "runtime/gincomm.hpp"
#include "util/error.hpp"
#include "device/sync.cuh"
#include "device/configs.cuh"
#include "services/nccl_comm_cache.hpp"
#include "services/event.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <string>
#include <stdexcept>
#include <atomic>
#include <sched.h>

namespace pace {

namespace {

// Process-wide monotonic sequence so multiple Comms in one process that share
// the same topo_key hash don't collide on the same /pace_ipc_<hash> shm
// segment.
std::atomic<uint64_t> g_ipc_seq{0};

// The LSA team scope at construction time. NCCL is the source of truth for
// the natural scope (queried via ncclCommQueryProperties().nLsaTeams in
// Connect()), but the child *Comm ctor bodies run BEFORE Connect() and size
// buffers based on topo.nvl_local_ranks — so we need a value here. Use
// num_local_ranks as a conservative placeholder (works for single-physical-
// box; over-allocates buffer for the hypernode case — waste, not bug, and
// Connect() overwrites it with NCCL's authoritative LSA team size before
// post_connect_fn sizes the buffer).
//
// We do NOT read NCCL_LSA_TEAM_SIZE env var ourselves — that's NCCL's job
// (NCCL's computeLsaSize honors it end-to-end, and we get the result via
// ncclCommQueryProperties).

std::string ipc_shm_name_from_topo_key(const std::vector<int>& topo_key) {
    size_t h = 0;
    for (int r : topo_key) h = h * 31 + static_cast<size_t>(r);
    uint64_t seq = g_ipc_seq.fetch_add(1, std::memory_order_relaxed);
    return "/pace_ipc_" + std::to_string(h) + "_" + std::to_string(seq);
}

void ipc_shm_barrier(void* shm_ptr, int num_ranks, size_t barrier_offset) {
    auto* barrier = reinterpret_cast<std::atomic<int>*>(
        reinterpret_cast<uint8_t*>(shm_ptr) + barrier_offset);
    barrier->fetch_add(1, std::memory_order_release);
    while (barrier->load(std::memory_order_acquire) < num_ranks) {
        sched_yield();
    }
}

} // anonymous namespace

LaunchContext::LaunchContext()
    : kstream(at::cuda::getStreamFromPool(true)),
      c1_stream(at::cuda::getStreamFromPool(true)),
      c2_stream(at::cuda::getStreamFromPool(true)),
      c3_stream(at::cuda::getStreamFromPool(true)) {}

Comm::Comm(int rank, int num_ranks, int num_local_ranks, int unroll, int nvl_ring, int rdma_ring, int num_sms, std::vector<int> topo_key, bool marshals_args) {
    // Topology / launch scalars (the remaining fields keep their in-struct
    // defaults: round_n=0, active=false, use_ipc_path=false,
    // buffer_bytes/gin_sigs/gin_qps/send_slots=0 — the latter
    // three "must be set in child class"). Streams are grabbed by LaunchContext().
    topo.rank = rank;
    topo.num_ranks = num_ranks;
    topo.num_local_ranks = num_local_ranks;
    // nvl_local_ranks at ctor time = num_local_ranks (conservative; Connect()
    // overwrites with NCCL's authoritative value via ncclCommQueryProperties).
    topo.nvl_local_ranks = num_local_ranks;
    HOST_ASSERT(topo.nvl_local_ranks > 0);
    topo.topo_key = std::move(topo_key);
    launch.unroll = unroll;
    launch.nvl_ring = nvl_ring;
    launch.rdma_ring = rdma_ring;
    launch.num_sms = num_sms;
    launch.marshals_args = marshals_args;
    CUDACHECK(cudaGetDevice(&topo.cuda_device_id));
    if (!marshals_args) {
        // EP drives kernels directly; no batched-memop arg marshaling.
        launch.mparam = nullptr;
    } else if (num_sms == 0) {
        // Size the batched-memop arg buffer for the whole group. SG's
        // num_sms==0 path (scattergatherfunc) builds batches of
        // num_nodes * num_local_ranks entries (pre_c1/post_c1); num_local_ranks
        // alone overflows on multi-node (num_nodes>1), corrupting the heap.
        // num_nodes * num_local_ranks == num_ranks, so num_ranks covers it (and
        // equals num_local_ranks in the single-node case).
        launch.mparam = (CUstreamBatchMemOpParams*)malloc(sizeof(CUstreamBatchMemOpParams) * num_ranks);
    } else {
        launch.mparam = (CUstreamBatchMemOpParams*)malloc(sizeof(CUstreamBatchMemOpParams) * kMaxArgs * 8);
    }
}

pybind11::bytearray Comm::GenHeadInfo() {
    ncclUniqueId global_nccl_id;
    NCCLCHECK(ncclGetUniqueId(&global_nccl_id));
    return {reinterpret_cast<const char*>(&global_nccl_id), sizeof(global_nccl_id)};
}

std::tuple<int64_t, int64_t, int64_t> Comm::GetStream() const {
    c10::Stream s = launch.kstream.unwrap();
    return {static_cast<int64_t>(s.id()),
            static_cast<int64_t>(s.device_index()),
            static_cast<int64_t>(s.device_type())};
}

int Comm::Connect(pybind11::bytearray head_info) {
    // buffer.buffer_bytes is sized by post_connect_fn (called below, after
    // the NCCL query in the non-IPC path, or with conservative nvl_local_ranks
    // in the IPC path). Don't assert > 0 here — it's still 0 at this point.
    const int num_nodes = topo.num_ranks / topo.num_local_ranks;
    const int local_rank = topo.rank % topo.num_local_ranks;
    constexpr size_t signal_bytes = SIGNAL_BYTES;

    buffer.use_ipc_path = (num_nodes == 1);
    // Rationale: the IPC path uses cudaMalloc + cudaIpcMemHandle (no ncclComm),
    // so gdaki/ncclDevComm can't be set up. On a single physical box the whole
    // group is intranode — take the cheap IPC path. Any multi-node (GIN) case
    // takes the ncclComm + gdaki path below.

    if (buffer.use_ipc_path) {
        // Pure intranode without multicast: use cudaMalloc + cudaIpc, no ncclComm needed.
        transport.global_comm = nullptr;
        buffer.gin_win = nullptr;

        // IPC path = single physical box, default (no override). Conservative
        // topo.nvl_local_ranks (= num_local_ranks = num_ranks for single box)
        // is the authoritative value here — no NCCL query needed. Run
        // post_connect_fn to size buffer_bytes + gin_sigs accordingly.
        // (gin_sigs will be 0 since num_nodes==1 — no gdaki needed.)
        if (post_connect_fn) post_connect_fn();

        CUDACHECK(cudaMalloc(&buffer.data_buffer, buffer.buffer_bytes + signal_bytes));
        buffer.signal_buffer = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(buffer.data_buffer) + buffer.buffer_bytes);
        if (topo.rank == 0) {
            printf("[rank %d]: Pace Communicator (IPC path) allocates %lu bytes at %p\n", topo.rank, buffer.buffer_bytes, buffer.data_buffer);
        }

        // Exchange IPC handles via POSIX shared memory
        cudaIpcMemHandle_t my_handle;
        CUDACHECK(cudaIpcGetMemHandle(&my_handle, buffer.data_buffer));

        const std::string shm_name = ipc_shm_name_from_topo_key(topo.topo_key);
        const size_t handles_size = sizeof(cudaIpcMemHandle_t) * topo.num_ranks;
        const size_t barrier_offset = handles_size;
        const size_t shm_size = barrier_offset + sizeof(std::atomic<int>);

        int shm_fd;
        void* shm_ptr;
        if (local_rank == 0) {
            shm_unlink(shm_name.c_str());
            shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0666);
            HOST_ASSERT(shm_fd >= 0);
            HOST_ASSERT(ftruncate(shm_fd, shm_size) == 0);
            shm_ptr = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
            HOST_ASSERT(shm_ptr != MAP_FAILED);
        } else {
            // Wait for rank 0 to create and size the shm
            while ((shm_fd = shm_open(shm_name.c_str(), O_RDWR, 0666)) < 0) {
                sched_yield();
            }
            struct stat st;
            do { fstat(shm_fd, &st); sched_yield(); } while (st.st_size < (ssize_t)shm_size);
            shm_ptr = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
            HOST_ASSERT(shm_ptr != MAP_FAILED);
        }

        // Write own handle
        auto* all_handles = reinterpret_cast<cudaIpcMemHandle_t*>(shm_ptr);
        memcpy(&all_handles[local_rank], &my_handle, sizeof(cudaIpcMemHandle_t));

        // Barrier: wait for all ranks to write their handles
        ipc_shm_barrier(shm_ptr, topo.num_ranks, barrier_offset);

        // Import IPC handles to get p2p pointers. Same effective-group
        // indexing as the non-IPC path: p2p_ptrs[i] points to the i-th peer
        // in THIS rank's effective NVL group [(rank / nvl_local_ranks) *
        // nvl_local_ranks, ...]. Iterate the effective group (never
        // num_local_ranks, which would write past the nvl_local_ranks-sized
        // array on hypernode where the group spans more than num_local_ranks).
        buffer.p2p_ptrs = (void**)malloc(sizeof(void*) * topo.nvl_local_ranks);
        memset(buffer.p2p_ptrs, 0, sizeof(void*) * topo.nvl_local_ranks);
        const int ipc_effective_group_base = (topo.rank / topo.nvl_local_ranks) * topo.nvl_local_ranks;
        const int ipc_effective_local_rank = topo.rank - ipc_effective_group_base;
        for (int i = 0; i < topo.nvl_local_ranks; ++i) {
            if (i == ipc_effective_local_rank) {
                buffer.p2p_ptrs[i] = buffer.data_buffer;
            } else {
                CUDACHECK(cudaIpcOpenMemHandle(&buffer.p2p_ptrs[i], all_handles[ipc_effective_group_base + i], cudaIpcMemLazyEnablePeerAccess));
            }
        }

        // Clean up shared memory
        munmap(shm_ptr, shm_size);
        close(shm_fd);
        if (local_rank == 0) {
            shm_unlink(shm_name.c_str());
        }

    } else {
        // Original path: ncclComm + ncclMemAlloc + window registration
        if (NcclCommRegistry::instance().try_acquire(topo.topo_key, &transport.global_comm)) {
            // Reusing a previously-built ncclComm for this exact topology.
        } else {
            ncclUniqueId global_nccl_id;
            std::string head_str = std::string(head_info);
            HOST_ASSERT(head_str.size() == sizeof(global_nccl_id));
            std::memcpy(&global_nccl_id, head_str.c_str(), sizeof(global_nccl_id));
            ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
            config.maxCTAs = 1;
            config.minCTAs = 1;
            NCCLCHECK(ncclCommInitRankConfig(&transport.global_comm, topo.num_ranks, global_nccl_id, topo.rank, &config));
            NcclCommRegistry::instance().install(topo.topo_key, transport.global_comm);
        }

        // NCCL is authoritative for the natural LSA team scope. Query BEFORE
        // ncclMemAlloc so post_connect_fn can size buffer_bytes against the
        // authoritative nvl_local_ranks (matters for the hypernode case where
        // NCCL's lsaSize > num_local_ranks — the conservative placeholder
        // would under-size gin_sigs / over-allocate buffer in ways the kernel
        // doesn't expect, and for the override case, the override value was
        // already set in Comm::Comm but post_connect_fn reads topo.* fields).
        ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
        NCCLCHECK(ncclCommQueryProperties(transport.global_comm, &props));
        const int nccl_lsa_team_size = topo.num_ranks / props.nLsaTeams;
        HOST_ASSERT(nccl_lsa_team_size > 0);
        HOST_ASSERT(topo.num_ranks % nccl_lsa_team_size == 0);
        HOST_ASSERT(nccl_lsa_team_size % topo.num_local_ranks == 0);
        // Accept NCCL's natural LSA team scope (hypernode auto-detect — e.g.
        // NVL72 spanning multiple physical boxes). The conservative
        // num_local_ranks placeholder set in Comm::Comm is overwritten here,
        // BEFORE post_connect_fn sizes the buffer.
        topo.nvl_local_ranks = nccl_lsa_team_size;
        HOST_ASSERT(topo.nvl_local_ranks % topo.num_local_ranks == 0);
        HOST_ASSERT(topo.num_ranks % topo.nvl_local_ranks == 0);
        // Effective n_lsa_teams: num_ranks / nvl_local_ranks — equals
        // props.nLsaTeams (NCCL's natural scope). Gates the gdaki setup below.
        topo.n_lsa_teams = topo.num_ranks / topo.nvl_local_ranks;
        // Run post_connect_fn now that topo.nvl_local_ranks + n_lsa_teams
        // are authoritative. Sizes buffer_bytes + gin_sigs based on the real
        // values. (The child *Comm ctor body ran BEFORE Connect and so couldn't
        // size buffers correctly for the hypernode case — this deferral fixes
        // the buffer-vs-kernel-view mismatch.)
        if (post_connect_fn) post_connect_fn();
        NCCLCHECK(ncclMemAlloc(&buffer.data_buffer, buffer.buffer_bytes + signal_bytes));
        buffer.signal_buffer = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(buffer.data_buffer) + buffer.buffer_bytes);
        if (topo.rank == 0) {
            printf("[rank %d]: Pace Communicator allocates %lu bytes at %p (nvl_local_ranks=%d, n_lsa_teams=%d, nccl_n_lsa_teams=%d)\n",
                   topo.rank, buffer.buffer_bytes, buffer.data_buffer, topo.nvl_local_ranks, topo.n_lsa_teams, props.nLsaTeams);
        }
        NCCLCHECK(ncclCommWindowRegister(transport.global_comm, buffer.data_buffer, buffer.buffer_bytes + signal_bytes, &buffer.gin_win, NCCL_WIN_COLL_SYMMETRIC));

        // Per-peer pointers within this rank's effective NVL group
        // [(rank / nvl_local_ranks) * nvl_local_ranks, ...]. The downstream
        // kernels index p2p_ptrs[peer % nvl_local_ranks], so p2p_ptrs[i] must
        // point to the i-th peer in THIS rank's effective group, not LSA-local
        // rank i globally.
        buffer.p2p_ptrs = (void**)malloc(sizeof(void*) * topo.nvl_local_ranks);
        const int effective_group_base = (topo.rank / topo.nvl_local_ranks) * topo.nvl_local_ranks;
        const int effective_local_rank = topo.rank - effective_group_base;
        for (int i = 0; i < topo.nvl_local_ranks; ++i) {
            if (i == effective_local_rank) {
                buffer.p2p_ptrs[i] = buffer.data_buffer;
                continue;
            }
            NCCLCHECK(ncclGetLsaDevicePointer(buffer.gin_win, 0, effective_group_base + i, buffer.p2p_ptrs + i));
        }

        if (topo.n_lsa_teams > 1) {
            HOST_ASSERT(transport.gin_sigs > 0);
            const size_t signal_count = transport.gin_sigs;
            const int gin_context_count = std::max(int(transport.gin_qps), GIN_QPS);
            ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
            reqs.ginSignalCount = signal_count;
            reqs.ginContextCount = gin_context_count;
            reqs.ginExclusiveContexts = true;
            reqs.ginQueueDepth = 1024;
            reqs.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;
            NCCLCHECK(ncclDevCommCreate(transport.global_comm, &reqs, &transport.dev_comm));
            if (transport.dev_comm.ginSignalCount < signal_count) {
                printf("[rank %d]: ginSignalCount %d is less than signal_count %lu\n", topo.rank, transport.dev_comm.ginSignalCount, signal_count);
                HOST_ASSERT(transport.dev_comm.ginSignalCount >= signal_count);
            }
            if (transport.dev_comm.ginContextCount < gin_context_count) {
                printf("[rank %d]: ginContextCount %u is less than requested %d (per-SM exclusive QP may not be honored)\n",
                    topo.rank, transport.dev_comm.ginContextCount, gin_context_count);
                HOST_ASSERT(transport.dev_comm.ginContextCount >= gin_context_count);
            }
            first_sync_func(signal_count, static_cast<uint32_t>(transport.dev_comm.ginContextCount), transport.dev_comm, launch.kstream);
        }
    }

    // cudaMemsetAsync (not the synchronous cudaMemset) so the zeroing is
    // ordered on launch.kstream ahead of the p2p_ptrs memcpy below — both ops
    // target the same signal_buffer and must not race.
    CUDACHECK(cudaMemsetAsync(buffer.signal_buffer, 0, signal_bytes, launch.kstream));
    CUDACHECK(cudaMemcpyAsync(buffer.signal_buffer, buffer.p2p_ptrs, topo.nvl_local_ranks * sizeof(void*), cudaMemcpyHostToDevice, launch.kstream));

    // Kernel-arg marshaling buffers (skipped for comms that drive kernels
    // directly, e.g. EP — see LaunchContext::marshals_args).
    if (launch.marshals_args) {
        constexpr size_t arg_size = kMaxArgs * 8 * sizeof(uint64_t);
        CUDACHECK(cudaMalloc(&launch.args_gpu, arg_size));
        launch.args_host_buffer = (uint64_t*)malloc(arg_size);
        if (launch.num_sms > 0) {
            memset(launch.mparam, 0, sizeof(CUstreamBatchMemOpParams) * kMaxArgs * 8);
            for (int w = 0; w < kMaxArgs * 8; ++w) {
                launch.mparam[w].operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
                launch.mparam[w].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
                launch.mparam[w].writeValue.address = reinterpret_cast<CUdeviceptr>(launch.args_gpu + w);
                launch.mparam[w].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
            }
        }
    }
    transport.active = true;
    return 0;
}

void Comm::FillArgs(const int& u64s) {
    HOST_ASSERT(launch.num_sms > 0);

    for (int i = 0; i < u64s; ++i) {
        launch.mparam[i].writeValue.value64 = launch.args_host_buffer[i];
    }
}

void Comm::Destroy() {
    if (!transport.active) return;
    CUDACHECK(cudaFree(launch.args_gpu));

    if (buffer.use_ipc_path) {
        CUDACHECK(cudaDeviceSynchronize());

        // Mirror the Connect() IPC loop's effective-group indexing. p2p_ptrs
        // is sized nvl_local_ranks; iterate the effective group, never
        // num_local_ranks (hypernode group can span more than num_local_ranks).
        const int ipc_eff_base = (topo.rank / topo.nvl_local_ranks) * topo.nvl_local_ranks;
        const int ipc_eff_local = topo.rank - ipc_eff_base;
        for (int i = 0; i < topo.nvl_local_ranks; ++i) {
            if (i != ipc_eff_local && buffer.p2p_ptrs[i] != nullptr) {
                CUDACHECK(cudaIpcCloseMemHandle(buffer.p2p_ptrs[i]));
            }
        }
        CUDACHECK(cudaFree(buffer.data_buffer));
    } else {
        // Barrier all ranks on this (possibly shared) ncclComm so the collective
        // teardown happens in lock-step across ranks.
        NCCLCHECK(ncclAllReduce(nullptr, nullptr, 0, ncclFloat, ncclSum, transport.global_comm, launch.kstream));
        CUDACHECK(cudaStreamSynchronize(launch.kstream));

        // Destroy this Comm's GIN device-comm (created per-Comm only for the
        // inter-LSA-team path in Connect); the shared ncclComm is released below.
        // Gate on topo.n_lsa_teams (set in Connect from ncclCommQueryProperties)
        // — not num_nodes — so hypernode-intranode (single LSA team spanning
        // multiple physical boxes) skips the destroy since no gdaki was set up.
        if (topo.n_lsa_teams > 1) {
            NCCLCHECK(ncclDevCommDestroy(transport.global_comm, &transport.dev_comm));
        }

        NCCLCHECK(ncclCommWindowDeregister(transport.global_comm, buffer.gin_win));

        NCCLCHECK(ncclMemFree(buffer.data_buffer));

        NcclCommRegistry::instance().release(topo.topo_key);
        transport.global_comm = nullptr;
    }

    // Do NOT cudaStreamDestroy the launch streams: they come from torch's
    // stream pool (at::cuda::getStreamFromPool), which owns a fixed set of
    // streams per device (32 high-priority) handed out round-robin. Destroying
    // them frees handles that the pool keeps handing out, so after 8 comms
    // (8 * 4 streams) the next comm reuses a dangling stream and crashes
    // (e.g. cudaMemsetAsync). torch owns the pool streams; leave them alone.
    free(launch.mparam);
    free(buffer.p2p_ptrs);
    free(launch.args_host_buffer);
    transport.active = false;
}

Comm::~Comm() {
    if (transport.active) {
        printf("Pace Communicator is not correctly finalized\n");
        Destroy();
    }
}

}