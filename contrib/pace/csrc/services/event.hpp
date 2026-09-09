#ifndef __EVENT_HPP__
#define __EVENT_HPP__
#include <ATen/cuda/CUDAContext.h>
#include <memory>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <ATen/cuda/CUDADataType.h>
#include <torch/python.h>
#include "util/error.hpp"
namespace pace {

struct EventHandle {
    std::shared_ptr<torch::Event> event;

    EventHandle() {
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(at::cuda::getCurrentCUDAStream());
    }

    explicit EventHandle(const at::cuda::CUDAStream& stream) {
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(stream);
    }

    EventHandle(const EventHandle& other) = default;

    void current_stream_wait() const {
        at::cuda::getCurrentCUDAStream().unwrap().wait(*event);
    }

    void host_wait() const {
        event->synchronize();
    }

    bool host_query() {
        return event->query();
    }

    void wait(uint64_t stream_handle) {
        cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream_handle);
        const at::cuda::CUDAStream stream = at::cuda::getStreamFromExternal(
            cuda_stream, static_cast<int>(at::cuda::current_device()));
        stream.unwrap().wait(*event);
    }
};

torch::Event create_event(const at::cuda::CUDAStream &s);

void stream_wait(const at::cuda::CUDAStream& s_0, const at::cuda::CUDAStream& s_1);

void stream_wait(const at::cuda::CUDAStream& s, const EventHandle& event);

} // namespace pace

#endif