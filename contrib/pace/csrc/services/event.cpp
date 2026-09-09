
#include "services/event.hpp"
namespace pace {

torch::Event create_event(const at::cuda::CUDAStream &s) {
    auto event = torch::Event(torch::kCUDA);
    event.record(s);
    return event;
}

void stream_wait(const at::cuda::CUDAStream& s_0, const at::cuda::CUDAStream& s_1) {
    HOST_ASSERT(s_0.id() != s_1.id());
    s_0.unwrap().wait(create_event(s_1));
}

void stream_wait(const at::cuda::CUDAStream& s, const EventHandle& event) {
    s.unwrap().wait(*event.event);
}

}
