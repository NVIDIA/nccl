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

# On Windows, exclude GIN device sources (GIN backend implementation is Linux-only; would cause nvlink undefined refs).
set(device_files ${files})
set(device_symmetric_files ${symmetric_files})
if(NCCL_OS_WINDOWS)
  list(FILTER device_files EXCLUDE REGEX "gin")
  list(FILTER device_symmetric_files EXCLUDE REGEX "gin")
endif()

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
    COMMAND ${CMAKE_COMMAND} -E env ${symmetric_gen_env} ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/generate.py ${CMAKE_CURRENT_BINARY_DIR}/gensrc/symmetric "${ONLY_FUNCS}"
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/generate.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating symmetric device source files"
)

# On Linux, ncclGin_BackendMask<...> is provided by implicit instantiation in common.cu (no -G).
# On Windows, GIN device code is excluded (device_files / device_symmetric_files filtered above).

# On Windows use STATIC so CUDA device linking runs (OBJECT libs don't support it); on Linux use OBJECT.
if(NCCL_OS_WINDOWS)
    add_library(nccl_device STATIC
                ${device_files}
                ${device_symmetric_files}
                ${CMAKE_CURRENT_SOURCE_DIR}/common.cu
                ${CMAKE_CURRENT_SOURCE_DIR}/onerank.cu
    )
    set_target_properties(nccl_device PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        POSITION_INDEPENDENT_CODE ON
    )
else()
    add_library(nccl_device OBJECT
                ${files}
                ${symmetric_files}
                ${CMAKE_CURRENT_SOURCE_DIR}/common.cu
                ${CMAKE_CURRENT_SOURCE_DIR}/onerank.cu
    )
    set_target_properties(nccl_device PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    )
endif()


# On Linux, define NCCL_OS_LINUX so nccl_device.h includes gin__funcs.h (required for ncclGin_BackendMask<...> in common.cu and symmetric GIN kernels)
if(NCCL_OS_LINUX)
    target_compile_definitions(nccl_device PRIVATE NCCL_OS_LINUX)
endif()

# WIN32_LEAN_AND_MEAN prevents windows.h from pulling in winsock.h, so _WINSOCKAPI_
# is only defined once (by winsock2.h) and C4005 redefinition warnings are avoided.
# Skip GIN/DOCA (GDAKI) on Windows: doca_gpunetio headers are not available and GIN backend is Linux-only.
if(NCCL_OS_WINDOWS)
    target_compile_definitions(nccl_device PRIVATE WIN32_LEAN_AND_MEAN NCCL_GIN_GDAKI_ENABLE=0)
endif()

# Export-safe includes: only INSTALL_INTERFACE in PUBLIC; build paths in PRIVATE so export does not see source/build dirs.
target_include_directories(nccl_device PUBLIC
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(nccl_device PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_SOURCE_DIR}/src/include/plugin
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}/cccl
)

add_dependencies(nccl_device nccl_header)
