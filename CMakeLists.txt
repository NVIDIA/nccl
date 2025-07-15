cmake_minimum_required(VERSION 4.0)

project(
  nccl
  LANGUAGES CUDA CXX
  VERSION 2.27.3.1)

option(VERBOSE "VERBOSE" OFF)
option(KEEP "KEEP" OFF)
option(TRACE "TRACE" OFF)
option(PROFAPI "PROFAPI" OFF)
option(NVTX "NVTX" ON)
option(NET_PROFILER "NET_PROFILER" OFF)

add_subdirectory(src)

install(
  TARGETS nccl nccl_static
  EXPORT NCCLConfig
  FILE_SET public_headers
  DESTINATION include)

install(
  EXPORT NCCLConfig
  DESTINATION lib/cmake/nccl
  NAMESPACE NCCL::)
