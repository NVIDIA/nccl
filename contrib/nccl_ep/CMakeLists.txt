# NCCL EP Example/Library CMake Configuration

if(DEFINED CMAKE_CUDA_ARCHITECTURES)
  foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(arch LESS 90)
      message(FATAL_ERROR "NCCL_EP requires sm_90 or higher. CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
    endif()
  endforeach()
endif()

# Parse library version from include/nccl_ep.h (single source of truth).
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/include/nccl_ep.h _NCCL_EP_HEADER_CONTENT)
string(REGEX REPLACE ".*#define NCCL_EP_MAJOR ([0-9]+).*" "\\1" NCCL_EP_MAJOR "${_NCCL_EP_HEADER_CONTENT}")
string(REGEX REPLACE ".*#define NCCL_EP_MINOR ([0-9]+).*" "\\1" NCCL_EP_MINOR "${_NCCL_EP_HEADER_CONTENT}")
string(REGEX REPLACE ".*#define NCCL_EP_PATCH ([0-9]+).*" "\\1" NCCL_EP_PATCH "${_NCCL_EP_HEADER_CONTENT}")
unset(_NCCL_EP_HEADER_CONTENT)
message(STATUS "NCCL EP version: ${NCCL_EP_MAJOR}.${NCCL_EP_MINOR}.${NCCL_EP_PATCH}")

# Source files
set(NCCL_EP_HOST_SRC
    nccl_ep.cc
    device/jit/jit_utils.cc
    device/jit/jit_cache.cc
    device/jit/nvcc_compiler.cc
    device/jit/jit_runtime.cc
)

set(NCCL_EP_DEVICE_SRC
    device/low_latency.cu
    device/hybridep_adapter.cu
)

set(NCCL_EP_SRC
    ${NCCL_EP_HOST_SRC}
    ${NCCL_EP_DEVICE_SRC}
)

add_library(nccl_ep_shared SHARED
    ${NCCL_EP_SRC}
)
add_library(nccl_ep_static STATIC
    ${NCCL_EP_SRC}
)
add_custom_target(nccl_ep_lib DEPENDS nccl_ep_shared nccl_ep_static)

set(NCCL_EP_TARGETS nccl_ep_shared nccl_ep_static)

set_target_properties(nccl_ep_shared nccl_ep_static PROPERTIES
    OUTPUT_NAME "nccl_ep"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "90"
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

# Version the shared library:
#   libnccl_ep.so.MAJOR.MINOR.PATCH (actual file)
#   libnccl_ep.so.MAJOR             (SONAME; linker symlink)
#   libnccl_ep.so                   (unversioned symlink for build/runtime loading)
set_target_properties(nccl_ep_shared PROPERTIES
    VERSION   ${NCCL_EP_MAJOR}.${NCCL_EP_MINOR}.${NCCL_EP_PATCH}
    SOVERSION ${NCCL_EP_MAJOR}
)

set_target_properties(nccl_ep_shared PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
    INSTALL_RPATH "$ORIGIN"
)

set(NCCL_EP_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/device
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_BINARY_DIR}/include/nccl_device
    ${CMAKE_SOURCE_DIR}/src/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}/cccl
)

set(NCCL_EP_JIT_SOURCE_DIR "${CMAKE_BINARY_DIR}/include/nccl_ep")
set(NCCL_EP_JIT_BUILD_INCLUDE_DIR "${CMAKE_BINARY_DIR}/include")
set(NCCL_EP_JIT_CUDA_INCLUDE_DIR "")
if(CUDAToolkit_INCLUDE_DIRS)
    list(GET CUDAToolkit_INCLUDE_DIRS 0 NCCL_EP_JIT_CUDA_INCLUDE_DIR)
endif()
set(NCCL_EP_JIT_CONFIG_HEADER "${NCCL_EP_JIT_SOURCE_DIR}/config.h")
set(NCCL_EP_JIT_HEADERS
    device/hybrid_ep.cuh
    device/device_primitives.cuh
    device/hybridep_configs.cuh
    device/cuda_compat_shims.cuh
    device/scan_kernel.cuh
    device/macros.cuh
)

file(MAKE_DIRECTORY "${NCCL_EP_JIT_SOURCE_DIR}")
file(WRITE "${NCCL_EP_JIT_CONFIG_HEADER}"
    "#pragma once\n"
    "#define NCCL_EP_JIT_SOURCE_DIR \"${NCCL_EP_JIT_SOURCE_DIR}\"\n"
    "#define NCCL_EP_JIT_BUILD_INCLUDE_DIR \"${NCCL_EP_JIT_BUILD_INCLUDE_DIR}\"\n"
    "#define NCCL_EP_JIT_CUDA_INCLUDE_DIR \"${NCCL_EP_JIT_CUDA_INCLUDE_DIR}\"\n"
)

foreach(header IN LISTS NCCL_EP_JIT_HEADERS)
    get_filename_component(header_dir "${header}" DIRECTORY)
    file(MAKE_DIRECTORY "${NCCL_EP_JIT_SOURCE_DIR}/${header_dir}")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${header}" "${NCCL_EP_JIT_SOURCE_DIR}/${header}" COPYONLY)
endforeach()

# Allow device code to call host constexpr with --expt-relaxed-constexpr
# Hides all symbols by default in the host compiled object files with -fvisibility=hidden
foreach(t IN LISTS NCCL_EP_TARGETS)
    target_compile_options(${t} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-fvisibility=hidden>
        $<$<COMPILE_LANGUAGE:CXX>:-include>
        $<$<COMPILE_LANGUAGE:CXX>:${NCCL_EP_JIT_CONFIG_HEADER}>
    )
    if(TARGET CUDA::cuda_driver)
        target_link_libraries(${t} PUBLIC CUDA::cuda_driver)
    else()
        target_link_libraries(${t} PUBLIC cuda)
    endif()
    target_include_directories(${t} PUBLIC ${NCCL_EP_INCLUDE_DIRS})
    target_link_libraries(${t} PUBLIC nccl ${CUDAToolkit_LIBRARIES})
endforeach()

# Copy headers to build/include (runs at configure)
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include")
configure_file(include/nccl_ep.h    "${CMAKE_BINARY_DIR}/include/nccl_ep.h"    COPYONLY)
configure_file(include/ep_enums.h   "${CMAKE_BINARY_DIR}/include/ep_enums.h"   COPYONLY)
configure_file(include/common.hpp   "${CMAKE_BINARY_DIR}/include/common.hpp"   COPYONLY)

# Install to custom prefix
install(FILES
    include/nccl_ep.h
    include/ep_enums.h
    include/common.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

foreach(header IN LISTS NCCL_EP_JIT_HEADERS)
    get_filename_component(header_dir "${header}" DIRECTORY)
    install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/${header}"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/nccl_ep/${header_dir}"
    )
endforeach()

install(TARGETS ${NCCL_EP_TARGETS}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
