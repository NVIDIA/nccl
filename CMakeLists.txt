cmake_minimum_required(VERSION 3.22)


# === @begin: General Configuration ===
project(nccl LANGUAGES CUDA CXX VERSION 2.16.2.1)

# @TODO+:return to 23, see other issue blah-blah
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

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
    -Wall \
    -Wno-unused-function \
    -Wno-sign-compare \
    -fvisibility=hidden \
    -Wvla"
)
# === @end: General Configuration ===


# === @begin: src/ ===
add_subdirectory(src)
# === @end: src/ ===


# === @begin: Install Targets ===
include(GNUInstallDirs)


# NOTE: This is the install target output!
# -- Installing: /usr/local/include/nccl.h
# -- Up-to-date: /usr/local/include/nccl_net.h
# -- Installing: /usr/local/lib64/libnccl.so.2.16.2.1
# -- Installing: /usr/local/lib64/libnccl.so.2
# -- Installing: /usr/local/lib64/libnccl.so
# -- Installing: /usr/local/bin/ncclras
# -- Installing: /usr/local/lib64/cmake/nccl/NCCLConfig.cmake
# -- Installing: /usr/local/lib64/cmake/nccl/NCCLConfig-release.cmake

# Install the headers into the include directory
install(FILES ${INCEXPORTS} TYPE INCLUDE)

install(
    TARGETS nccl ncclras
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