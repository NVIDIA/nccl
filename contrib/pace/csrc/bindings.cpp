
#include <cassert>
#include <cstddef>
#include <nccl.h>
#include <optional>
#include <torch/python.h>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/types.h>
#include "runtime/gincomm.hpp"
#include "services/nccl_comm_cache.hpp"
#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "PACE: Parallelism Aware Collective Engine";

    m.def("has_cached_comm", [](const std::vector<int>& key) {
        return pace::NcclCommRegistry::instance().has(key);
    });
    m.def("cached_comm_count", []() {
        return pace::NcclCommRegistry::instance().size();
    });

    pybind11::class_<pace::EventHandle>(m, "EventHandle")
        .def(pybind11::init<>())
        .def("current_stream_wait", &pace::EventHandle::current_stream_wait)
        .def("host_wait", &pace::EventHandle::host_wait)
        .def("host_query", &pace::EventHandle::host_query)
        .def("wait", &pace::EventHandle::wait);

    // Collectives HAVE-A pace::Comm (composition; see runtime/gincomm.hpp).
    // The lifecycle methods live on Comm, so each collective exposes them to
    // Python via thin forwarding lambdas to its `.comm` member.
    pybind11::class_<pace::SGComm>(m, "SGComm")
        .def(pybind11::init<int, int, int, int, int, int, int, std::vector<int>>())
        .def("get_head_info", [](pace::SGComm& s) { return s.comm.GenHeadInfo(); })
        .def("get_stream", [](pace::SGComm& s) { return s.comm.GetStream(); })
        .def("get_shared_memory_bytes", [](pace::SGComm& s) { return s.comm.GetSharedMemoryBytes(); })
        .def("connect", [](pace::SGComm& s, pybind11::bytearray h) { return s.comm.Connect(h); })
        .def("destroy", [](pace::SGComm& s) { s.comm.Destroy(); })
        .def("scatter_gather", &pace::SGComm::ScatterGather, py::call_guard<py::gil_scoped_release>())
        .def("is_ring_mode", &pace::SGComm::is_ring_mode)
        .def("num_sms", &pace::SGComm::num_sms);

    pybind11::class_<pace::AGComm>(m, "AGComm")
        .def(pybind11::init<int, int, int, int, int, int, int, bool, bool, bool, int, std::vector<int>>())
        .def("get_head_info", [](pace::AGComm& s) { return s.comm.GenHeadInfo(); })
        .def("get_stream", [](pace::AGComm& s) { return s.comm.GetStream(); })
        .def("get_shared_memory_bytes", [](pace::AGComm& s) { return s.comm.GetSharedMemoryBytes(); })
        .def("connect", [](pace::AGComm& s, pybind11::bytearray h) { return s.comm.Connect(h); })
        .def("destroy", [](pace::AGComm& s) { s.comm.Destroy(); })
        .def("all_gather", &pace::AGComm::AllGather, py::call_guard<py::gil_scoped_release>());

    pybind11::class_<pace::RSComm>(m, "RSComm")
        .def(pybind11::init<int, int, int, int, int, int, int, bool, std::vector<int>>())
        .def("get_head_info", [](pace::RSComm& s) { return s.comm.GenHeadInfo(); })
        .def("get_stream", [](pace::RSComm& s) { return s.comm.GetStream(); })
        .def("get_shared_memory_bytes", [](pace::RSComm& s) { return s.comm.GetSharedMemoryBytes(); })
        .def("connect", [](pace::RSComm& s, pybind11::bytearray h) { return s.comm.Connect(h); })
        .def("destroy", [](pace::RSComm& s) { s.comm.Destroy(); })
        .def("out_numel_alignment", &pace::RSComm::out_numel_alignment)
        .def("reduce_scatter", &pace::RSComm::ReduceScatter, py::call_guard<py::gil_scoped_release>());

}