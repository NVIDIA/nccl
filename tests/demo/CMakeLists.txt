add_executable(nccl_demo_test nccl_demo.cpp)

target_include_directories(nccl_demo_test PRIVATE
    ${CMAKE_BINARY_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(nccl_demo_test PRIVATE nccl CUDA::cudart)

set_target_properties(nccl_demo_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
