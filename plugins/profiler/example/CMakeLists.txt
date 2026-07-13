# Host sources plus the device-hook TU (plugin_dev.cu, profiler v7).
set(SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/print_event.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/profiler_plugin_ce.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin_dev.cu
)

# The device hook needs CUDA; enable it and default an architecture if unset.
enable_language(CUDA)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Create shared library
add_library(nccl-profiler-example SHARED ${SRC_FILES})

# Set include directories
target_include_directories(nccl-profiler-example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/nccl
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Set output name to match Makefile
set_target_properties(nccl-profiler-example PROPERTIES
    OUTPUT_NAME "nccl-profiler-example"
    PREFIX "lib"
    POSITION_INDEPENDENT_CODE ON
    # Relocatable device code: the __device__ hook must have a real, callable
    # global address for NCCL's kernel to invoke it indirectly.
    CUDA_SEPARABLE_COMPILATION ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

add_custom_command(TARGET nccl-profiler-example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/test/unit/plugins
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/lib/libnccl-profiler-example.so ${CMAKE_BINARY_DIR}/test/unit/plugins
)

# Add custom target for clean (equivalent to Makefile clean target)
add_custom_target(clean-profiler-lib
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/lib/libnccl-profiler-example.so
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/test/unit/plugins/libnccl-profiler-example.so
    COMMENT "Cleaning libnccl-profiler-example.so"
)
