include(../cmake/common.cmake)

find_package(CUDAToolkit REQUIRED)

set(nccl_Major ${nccl_VERSION_MAJOR})
set(nccl_Minor ${nccl_VERSION_MINOR})
set(nccl_Patch ${nccl_VERSION_PATCH})
# NCCL_VERSION(X,Y,Z) ((X) * 10000 + (Y) * 100 + (Z))
math(
  EXPR
  nccl_Version
  "${nccl_VERSION_MAJOR} * 10000 + ${nccl_VERSION_MINOR} * 100 + ${nccl_VERSION_PATCH}"
)
set(nccl_Suffix)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/nccl.h.in
               ${CMAKE_CURRENT_SOURCE_DIR}/include/nccl.h)

file(
  GLOB
  SRC_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/misc/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/transport/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/collectives/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/graph/*.cc")

set(HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/nccl.h")

set(NCCL_LIBS nccl;nccl_static)

add_library(nccl SHARED ${SRC_FILES})
add_library(nccl_static STATIC ${SRC_FILES})

foreach(lib_name IN LISTS NCCL_LIBS)
  set_property(TARGET ${lib_name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  nccl_add_target_options(${lib_name})
  target_include_directories(
    ${lib_name}
    PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/plugin>)
  target_include_directories(${lib_name} PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
  target_sources(
    ${lib_name}
    PUBLIC FILE_SET
           public_headers
           TYPE
           HEADERS
           BASE_DIRS
           "${CMAKE_CURRENT_SOURCE_DIR}"
           FILES
           ${HEADER_FILES})
endforeach()
