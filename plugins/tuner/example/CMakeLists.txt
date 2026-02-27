# Find all C source files in current directory
set(SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin.c
)

# Create shared library
add_library(nccl-tuner-example SHARED ${SRC_FILES})

# Set include directories
target_include_directories(nccl-tuner-example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/nccl
)

# Set output name to match Makefile
set_target_properties(nccl-tuner-example PROPERTIES
    OUTPUT_NAME "nccl-tuner-example"
    PREFIX "lib"
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/unit/plugins
)

# Add custom target for clean (equivalent to Makefile clean target)
add_custom_target(clean-tuner-lib
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/libnccl-tuner-example.so
    COMMENT "Cleaning libnccl-tuner-example.so"
)
