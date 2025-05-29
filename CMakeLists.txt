cmake_minimum_required(VERSION 3.22)


# === @begin: General Configuration ===
project(nccl LANGUAGES CUDA CXX VERSION 2.16.2.1)

# @TODO+:check: do we still need that?
option(PRINT_VERBOSE "Print verbose build information" OFF)

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

# @TODO+: check and remove #@HELP
# -fcuda-rdc
# === @end: General Configuration ===


# === @begin: Dependencies ===
# Find system NVTX3
find_package(CUDAToolkit REQUIRED)

if (TARGET CUDA::nvtx3)
    message(STATUS "Found CUDA::nvtx3")
else()
    message(FATAL_ERROR "CUDA::nvtx3 not found")
endif()
# === @end: Dependencies ===


# === @begin: src/ ===
add_subdirectory(src)
# === @end: src/ ===


# # === @begin: Install Targets ===
# include(GNUInstallDirs)
# install(FILES ${INCEXPORTS} TYPE INCLUDE)

# target_include_directories(ncclras PUBLIC
#   ${CMAKE_CURRENT_SOURCE_DIR}
#   ${CMAKE_CURRENT_SOURCE_DIR}/../include
# )

# install(
#     TARGETS nccl nccl_static
#     EXPORT NCCLConfig
#     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
#     INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
# )

# install(
#     EXPORT NCCLConfig
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/nccl
#     NAMESPACE NCCL::
# )
# # === @end: Install Targets ===