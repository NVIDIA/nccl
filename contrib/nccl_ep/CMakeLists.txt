# NCCL EP Example/Library CMake Configuration

# Source files
set(NCCL_EP_HOST_SRC
    nccl_ep.cc
)

set(NCCL_EP_DEVICE_SRC
    device/low_latency.cu
)

# Create a static library for NCCL EP
add_library(nccl_ep_lib STATIC
    ${NCCL_EP_HOST_SRC}
    ${NCCL_EP_DEVICE_SRC}
)

# Set target properties
set_target_properties(nccl_ep_lib PROPERTIES
    OUTPUT_NAME "nccl_ep"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "90"
    POSITION_INDEPENDENT_CODE ON
)

# Enable multinode HT support (GIN/RDMA) - always enabled for multi-node testing
target_compile_definitions(nccl_ep_lib PUBLIC NCCL_EP_ENABLE_GIN)

# Include directories - only use public NCCL headers, no internal NCCL headers
target_include_directories(nccl_ep_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/device
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_BINARY_DIR}/include/nccl_device
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}/cccl
)

# Link with NCCL
target_link_libraries(nccl_ep_lib PUBLIC
    nccl
    ${CUDAToolkit_LIBRARIES}
)

# Install headers
install(FILES
    include/nccl_ep.h
    include/common.hpp
    DESTINATION ${CMAKE_BINARY_DIR}/include
)

# Add dependency on nccl_header
add_dependencies(nccl_ep_lib nccl_header)

