# Run the scripts once during configuration to get the file lists
execute_process(
    COMMAND ${CMAKE_COMMAND} -E env NCCL_USE_CMAKE=1 ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate.py ${CMAKE_CURRENT_BINARY_DIR}/gensrc "${ONLY_FUNCS}"
    OUTPUT_VARIABLE files
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
string(STRIP "${files}" files)
list(TRANSFORM files PREPEND ${CMAKE_CURRENT_BINARY_DIR}/gensrc/)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E env NCCL_USE_CMAKE=1 ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/generate.py ${CMAKE_CURRENT_BINARY_DIR}/gensrc/symmetric "${ONLY_FUNCS}"
    OUTPUT_VARIABLE symmetric_files
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
string(STRIP "${symmetric_files}" symmetric_files)
list(TRANSFORM symmetric_files PREPEND ${CMAKE_CURRENT_BINARY_DIR}/gensrc/symmetric/)

# Create custom commands to generate source files with proper dependencies
add_custom_command(
    OUTPUT  ${files}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate.py ${CMAKE_CURRENT_BINARY_DIR}/gensrc "${ONLY_FUNCS}"
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating device source files"
)

add_custom_command(
    OUTPUT  ${symmetric_files}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/generate.py ${CMAKE_CURRENT_BINARY_DIR}/gensrc/symmetric "${ONLY_FUNCS}"
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/generate.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating symmetric device source files"
)

# Add library target
add_library(nccl_device OBJECT
            ${files}
            ${symmetric_files}
            ${CMAKE_CURRENT_SOURCE_DIR}/common.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/onerank.cu
)

set_target_properties(nccl_device PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Set include directories for the target
target_include_directories(nccl_device PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_SOURCE_DIR}/src/include/plugin
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}/cccl
)

add_dependencies(nccl_device nccl_header)
