add_executable(nccl_unit_tests nccl_tests.cu)

target_include_directories(nccl_unit_tests PRIVATE
    ${CMAKE_BINARY_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(nccl_unit_tests PRIVATE nccl CUDA::cudart)

set_target_properties(nccl_unit_tests PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_test(NAME nccl_unit_tests COMMAND nccl_unit_tests)
