cmake_minimum_required(VERSION 3.22)

project(nccl LANGUAGES CUDA CXX VERSION 2.16.2.1)

option(PRINT_VERBOSE "Print verbose build information" OFF)
option(
    USE_SYSTEM_NVTX
    "Use system-installed NVTX3 (via find_package). If OFF, disables NVTX. Disabling NVTX is only for compatibility with NCCL"
    ON
)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_STATIC_LIBS "Build static libraries" OFF)

include_directories(src/include)

add_subdirectory(src)

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
