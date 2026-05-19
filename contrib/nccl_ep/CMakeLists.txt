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

# Allow device code to call host constexpr with --expt-relaxed-constexpr
# Hides all symbols by default in the host compiled object files with -fvisibility=hidden
foreach(t IN LISTS NCCL_EP_TARGETS)
    target_compile_options(${t} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-fvisibility=hidden>
    )
    target_compile_definitions(${t} PRIVATE
        NCCL_EP_JIT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        NCCL_EP_JIT_BUILD_INCLUDE_DIR="${CMAKE_BINARY_DIR}/include"
        NCCL_EP_JIT_CUDA_INCLUDE_DIR="${CUDAToolkit_INCLUDE_DIRS}")
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

# Install to custom prefix
install(FILES
    include/nccl_ep.h
    include/ep_enums.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(TARGETS ${NCCL_EP_TARGETS}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
