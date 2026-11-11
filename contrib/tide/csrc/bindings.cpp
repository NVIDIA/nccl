// SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>

#include "collective/ag/plan.h"
#include "collective/rs/plan.h"
#include "runtime/comm.h"
#include "util/check.h"

namespace py = pybind11;
using tide::AgPlan;
using tide::Comm;
using tide::RsPlan;
using tide::Window;

namespace {

// uint8 tensor over [off, off+bytes) of the window. Its storage starts there (storage_offset 0,
// own version counter -- what FSDP2 expects of an all-gather output) and keeps the window, hence
// the mapping and the communicator, alive.
torch::Tensor windowTensor(std::shared_ptr<Window> w, size_t off, size_t bytes, int device) {
  TIDE_CHECK(off <= w->bytes && bytes <= w->bytes - off, "tensor range out of window");
  TIDE_CHECK(bytes <= static_cast<size_t>(INT64_MAX), "tensor range too large");
  auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device);
  return torch::from_blob(static_cast<char*>(w->base) + off, {static_cast<int64_t>(bytes)}, [w](void*) {}, opts);
}

}  // namespace

PYBIND11_MODULE(_C, m) {
  m.doc() = "TIDE: layout-aware zero-SM collectives on NCCL public APIs";
  m.def("get_unique_id", [] { return py::bytes(tide::getUniqueId()); });
  m.def("nccl_runtime_version", [] {
    int v = 0;
    NCCLCHECK(ncclGetVersion(&v));
    return v;
  });
  m.attr("nccl_compiled_version") = NCCL_VERSION_CODE;

  py::class_<Window, std::shared_ptr<Window>>(m, "Window")
      .def_property_readonly("bytes", [](const Window& w) { return w.bytes; })
      .def_property_readonly("base", [](const Window& w) { return reinterpret_cast<uintptr_t>(w.base); })
      .def_readonly("id", &Window::id)
      .def_readonly("freed", &Window::freed)
      .def_property_readonly("has_multimem", [](const Window& w) { return w.multimem != nullptr; })
      .def("tensor", [](std::shared_ptr<Window> w, int device) { return windowTensor(w, 0, w->bytes, device); },
           py::arg("device"))
      .def("tensor_range", &windowTensor, py::arg("off"), py::arg("bytes"), py::arg("device"));

  py::class_<AgPlan, std::shared_ptr<AgPlan>>(m, "AgPlan")
      .def_readonly("calls", &AgPlan::calls)
      .def_readonly("shard_bytes", &AgPlan::shardBytes)
      .def_readonly("region_bytes", &AgPlan::regionBytes)
      .def_readonly("region_off", &AgPlan::regionOff)
      .def_readonly("recv_base", &AgPlan::recvBase)
      .def_readonly("own_off", &AgPlan::ownOff)
      .def_readonly("staged", &AgPlan::staged)
      .def_readonly("direct", &AgPlan::direct)
      .def_readonly("stage_slot", &AgPlan::stageSlot)
      .def_readonly("paced", &AgPlan::paced)
      .def_property_readonly("rounds", [](const AgPlan& p) { return p.rounds.size(); })
      .def_property_readonly("pace_tails", [](const AgPlan& p) {
        std::vector<size_t> v;
        for (const auto& rd : p.rounds) v.push_back(rd.paceTail);
        return v;
      })
      .def_property_readonly("nic_chunks", [](const AgPlan& p) {
        size_t n = 0;
        for (const auto& rd : p.rounds)
          for (const auto& g : rd.groupPuts)
            for (const auto& put : g) n += put.chunks.size();
        return n;
      })
      .def_property_readonly("num_descriptors", [](const AgPlan& p) {
        size_t n = p.copies.len.size() + p.ownScatter.size();
        for (const auto& rd : p.rounds)
          for (const auto& b : rd.scatter) n += b.size();
        return n;
      });

  py::class_<RsPlan, std::shared_ptr<RsPlan>>(m, "RsPlan")
      .def_readonly("calls", &RsPlan::calls)
      .def_readonly("out_elems", &RsPlan::outElems)
      .def_readonly("out_bytes", &RsPlan::outBytes)
      .def_readonly("stage_base", &RsPlan::stageBase)
      .def_readonly("stage_bytes", &RsPlan::stageBytes)
      .def_readonly("chunks", &RsPlan::chunks)
      .def_readonly("chunk_bytes", &RsPlan::chunkBytes)
      .def_readonly("area", &RsPlan::area)
      .def_readonly("paced", &RsPlan::paced)
      .def_readonly("pace_tail", &RsPlan::paceTail)
      .def_readonly("rail_blocks", &RsPlan::railBlocks)
      .def_property_readonly("k_rounds", [](const RsPlan& p) { return p.kRounds; })
      .def_readonly("wave_nodes", &RsPlan::waveNodes)
      .def_property_readonly("waves", [](const RsPlan& p) { return p.waves.size(); })
      .def_readonly("sub_elems", &RsPlan::subElems)
      .def_readonly("parts", &RsPlan::parts)
      .def_readonly("part_elems", &RsPlan::partElems)
      .def_readonly("stg_elems", &RsPlan::stgElems)
      .def_readonly("inbox_base", &RsPlan::inboxBase)
      .def_readonly("outbox_base", &RsPlan::outboxBase)
      .def_readonly("outbox_slots", &RsPlan::outboxSlots)
      .def_readonly("pipeline", &RsPlan::pipeline)
      .def_readonly("outbox_bank", &RsPlan::outboxBank)
      .def_readonly("own_waves", &RsPlan::ownWaves)
      .def_readonly("own_spares", &RsPlan::ownSpares)
      .def_readonly("out_stride", &RsPlan::outStride)
      .def_readonly("slot_bytes", &RsPlan::slotBytes)
      .def_readonly("itemsize", &RsPlan::itemsize)
      .def_property_readonly("segments", [](const RsPlan& p) {  // (numel, pieceElems, outOff, stgOff) per segment
        std::vector<std::tuple<size_t, size_t, size_t, size_t>> v;
        for (const auto& s : p.segs) v.emplace_back(s.numel, s.pieceElems, s.outOff, s.stgOff);
        return v;
      })
      .def_property_readonly("pieces", [](const RsPlan& p) {  // (pieceElems, outOff) per segment
        std::vector<std::pair<size_t, size_t>> v;
        for (const auto& s : p.segs) v.emplace_back(s.pieceElems, s.outOff);
        return v;
      });

  py::class_<Comm>(m, "Comm")
      .def(py::init<const std::string&, int, int, int, int>(), py::arg("unique_id"), py::arg("nranks"),
           py::arg("rank"), py::arg("num_rma_ctx") = 0, py::arg("num_rma_sig") = 0)
      .def_property_readonly("rank", &Comm::rank)
      .def_property_readonly("nranks", &Comm::nranks)
      .def_property_readonly("seq", &Comm::seq)
      .def_property_readonly("node", &Comm::node)
      .def_property_readonly("init_s", [](const Comm& c) { return py::make_tuple(c.initCommS(), c.initSyncS(), c.initRailS()); })
      .def_property_readonly("n_nodes", &Comm::nNodes)
      .def_property_readonly("local_ranks", &Comm::localRanks)
      .def_property_readonly("rma_ctx", &Comm::rmaCtx)
      .def_property_readonly("rma_sig", &Comm::rmaSig)
      .def_property_readonly("nic_groups", &Comm::nicGroups)
      .def_property_readonly("rs_rail_parts", [](const Comm& c) { return c.config().rsRailParts; })
      .def_property_readonly("pace_mode", &Comm::paceMode)
      .def_property_readonly("pace_lead", &Comm::paceLead)
      .def_property_readonly("closed", &Comm::closed)
      .def("lsa_peers", [](const Comm& c) { return c.lsaPeers(); })
      .def("props",
           [](const Comm& c) {
             const auto& p = c.props();
             py::dict d;
             d["rank"] = p.rank;
             d["nranks"] = p.nranks;
             d["cuda_dev"] = p.cudaDev;
             d["device_api"] = p.deviceApi;
             d["multimem"] = p.multimem;
             d["host_rma"] = p.hostRma;
             d["n_lsa_teams"] = p.nLsaTeams;
             d["gin_type"] = p.ginType;
             d["nccl_version"] = p.ncclVersion;
             return d;
           })
      .def("alloc_window", [](Comm& c, size_t bytes) { return c.allocWindow(bytes); }, py::arg("bytes"))
      .def(
          "register_window",
          [](Comm& c, torch::Tensor buf, py::object owner) {
            TIDE_CHECK(buf.is_cuda() && buf.is_contiguous() && buf.scalar_type() == torch::kUInt8,
                       "register_window takes a contiguous uint8 CUDA tensor");
            // The tensor keeps the memory, `owner` (its MemPool) the pool: the pool goes last, under the GIL.
            struct Keep {
              torch::Tensor buf;
              py::object owner;
              ~Keep() {
                buf.reset();
                if (!Py_IsInitialized()) {
                  owner.release();
                  return;
                }
                py::gil_scoped_acquire gil;
                owner = py::object();
              }
            };
            auto keep = std::shared_ptr<void>(new Keep{buf, std::move(owner)}, [](void* p) { delete static_cast<Keep*>(p); });
            return c.registerWindow(buf.data_ptr(), static_cast<size_t>(buf.numel()), std::move(keep));
          },
          py::arg("buf"), py::arg("owner"))
      .def("free_window", &Comm::freeWindow, py::arg("window"))
      .def("close", &Comm::close)
      .def("trace_ag", &Comm::traceAgDump)
      .def("register_ag", &Comm::registerAg, py::arg("arena"), py::arg("region_off"), py::arg("sizes"),
           py::arg("offsets"), py::arg("stage_off") = -1, py::arg("own_off") = -1)
      .def("all_gather",
           [](Comm& c, AgPlan& p, int64_t stream, int64_t freeEv, int64_t doneEv) {
             c.allGather(p, reinterpret_cast<cudaStream_t>(stream), reinterpret_cast<cudaEvent_t>(freeEv),
                         reinterpret_cast<cudaEvent_t>(doneEv));
           },
           py::arg("plan"), py::arg("stream"), py::arg("free_event") = 0, py::arg("done_event") = 0)
      .def("await_stage_reads",
           [](Comm& c, int64_t stream) { c.awaitStageReads(reinterpret_cast<cudaStream_t>(stream)); },
           py::arg("stream"))
      .def("set_rs_tuning", &Comm::setRsTuning, py::arg("sum_ctas"), py::arg("mc_ctas"), py::arg("kway"),
           py::arg("chunk_bytes"), py::arg("parts") = 0, py::arg("outbox") = 0, py::arg("pipeline") = 0)
      .def_property_readonly("rs_pipeline", &Comm::rsPipeline)
      .def_property_readonly("rs_tuning", [](const Comm& c) {
        return py::make_tuple(c.rsCtas(), c.rsMcCtas(), c.rsKway(), c.rsChunkBytes());
      })
      .def_property_readonly("rs_parts", &Comm::rsParts)
      .def_property_readonly("rs_outbox", &Comm::rsOutbox)
      .def_property_readonly("rs_stage_slots", [](const Comm& c) { return c.config().rsStageSlots; })
      .def_property_readonly("rs_parts_env", [](const Comm& c) {  // TIDE_RS_PARTS if set (wins over the solver), else 0
        return (c.config().envSet & tide::Config::kEnvRsParts) ? c.config().rsParts : 0;
      })
      .def("rs_stage_bytes", &Comm::rsStageBytes, py::arg("shapes"), py::arg("itemsize"))
      .def("rs_area_bytes", &Comm::rsAreaBytes, py::arg("tables"))
      .def("register_rs", &Comm::registerRs, py::arg("arena"), py::arg("stage_off"), py::arg("shapes"),
           py::arg("itemsize"))
      .def("reduce_scatter",
           [](Comm& c, RsPlan& p, const std::vector<uintptr_t>& src, uintptr_t out, float scale, bool accumulate,
              int64_t stream, int64_t readEv, int64_t readyEv, int64_t gateEv) {
             std::vector<const void*> ptrs(src.size());
             for (size_t i = 0; i < src.size(); ++i) ptrs[i] = reinterpret_cast<const void*>(src[i]);
             c.reduceScatter(p, ptrs, reinterpret_cast<float*>(out), scale, accumulate,
                             reinterpret_cast<cudaStream_t>(stream), reinterpret_cast<cudaEvent_t>(readEv),
                             reinterpret_cast<cudaEvent_t>(readyEv), reinterpret_cast<cudaEvent_t>(gateEv));
           },
           py::arg("plan"), py::arg("src"), py::arg("out"), py::arg("scale") = 1.0f, py::arg("accumulate") = false,
           py::arg("stream") = 0, py::arg("read_event") = 0, py::arg("ready_event") = 0, py::arg("gate_event") = 0)
      .def("put_signal",
           [](Comm& c, const Window& w, size_t srcOff, size_t bytes, int peer, size_t peerOff, int sigIdx, int ctx,
              int64_t stream) { c.putSignal(w, srcOff, bytes, peer, peerOff, sigIdx, ctx, reinterpret_cast<cudaStream_t>(stream)); },
           py::arg("window"), py::arg("src_off"), py::arg("bytes"), py::arg("peer"), py::arg("peer_off"),
           py::arg("sig_idx") = 0, py::arg("ctx") = 0, py::arg("stream") = 0)
      .def("signal", [](Comm& c, int peer, int sigIdx, int ctx, int64_t stream) {
        c.signal(peer, sigIdx, ctx, reinterpret_cast<cudaStream_t>(stream));
      }, py::arg("peer"), py::arg("sig_idx") = 0, py::arg("ctx") = 0, py::arg("stream") = 0)
      .def("wait_signal", [](Comm& c, std::vector<std::array<int, 4>> descs, int64_t stream) {
        c.waitSignal(descs, reinterpret_cast<cudaStream_t>(stream));
      }, py::arg("descs"), py::arg("stream") = 0);
  m.def("group_start", [] { NCCLCHECK(ncclGroupStart()); });
  m.def("group_end", [] { NCCLCHECK(ncclGroupEnd()); });
}
