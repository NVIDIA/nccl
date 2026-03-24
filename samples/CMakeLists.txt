add_executable(nccl_demo nccl_demo.cu)

target_include_directories(nccl_demo PRIVATE
    ${CMAKE_BINARY_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(nccl_demo PRIVATE nccl CUDA::cudart)

set_target_properties(nccl_demo PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
