set(SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin.c
)

# Create shared library
add_library(nccl-mixed-example SHARED ${SRC_FILES})

# Set include directories
target_include_directories(nccl-mixed-example PRIVATE
    ${CMAKE_SOURCE_DIR}/ext-net/example/nccl
    ${CMAKE_SOURCE_DIR}/ext-tuner/example/nccl
)

# Set output name to match Makefile
set_target_properties(nccl-mixed-example PROPERTIES
    OUTPUT_NAME "nccl-mixed-example"
    PREFIX "lib"
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/unit/plugins
)
