# Find all C source files in current directory
set(SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/print_event.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/profiler_plugin_ce.cc
)

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
