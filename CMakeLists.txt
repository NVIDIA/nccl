cmake_minimum_required(VERSION 3.22)


# === @begin: General Configuration ===
project(nccl LANGUAGES CUDA CXX VERSION 2.16.2.1)

# @TODO+:check: do we still need that?
option(PRINT_VERBOSE "Print verbose build information" OFF)
option(
    USE_SYSTEM_NVTX
    "Use system-installed NVTX3 (via find_package). If OFF, disables NVTX. Disabling NVTX is only for compatibility with NCCL"
    ON
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 23)

set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} \
    -Wnvcc-compat \
    -Xcuda-ptxas -maxrregcount=96 \
    -Xcuda-fatbinary -compress-all \
    -fvisibility=hidden \
    -fPIC"
)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} \
    -fPIC \
    -fvisibility=hidden \
    -Wall \
    -Wno-unused-function \
    -Wno-sign-compare \
    -Wvla"
)

# @TODO+: check and remove
# -fcuda-rdc
# === @end: General Configuration ===



# Find system NVTX3
if(USE_SYSTEM_NVTX)
    # Fail immediately if not found
    find_package(nvtx3 CONFIG REQUIRED)
else()
    # Disable NVTX-related features
    add_compile_definitions(NVTX_DISABLE)
endif()

# === @end: General Configuration ===


# === @begin: src/ ===
add_subdirectory(src)
# === @end: src/ ===


# === @begin: Install Targets ===
include(GNUInstallDirs)
install(FILES ${INCEXPORTS} TYPE INCLUDE)

target_include_directories(ncclras PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

install(
    TARGETS nccl nccl_static
    EXPORT NCCLConfig
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    EXPORT NCCLConfig
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nccl
    NAMESPACE NCCL::
)
# === @end: Install Targets ===