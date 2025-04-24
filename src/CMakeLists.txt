include(../cmake/common.cmake)

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

add_subdirectory(device)

set(SRC_FILES
    init.cc
    init_nvtx.cc
    channel.cc
    bootstrap.cc
    transport.cc
    enqueue.cc
    group.cc
    debug.cc
    proxy.cc
    enhcompat.cc
    net.cc
    misc/cudawrap.cc
    misc/nvmlwrap.cc
    misc/ibvwrap.cc
    misc/gdrwrap.cc
    misc/utils.cc
    misc/argcheck.cc
    misc/socket.cc
    misc/shmutils.cc
    misc/profiler.cc
    misc/param.cc
    misc/strongstream.cc
    transport/p2p.cc
    transport/shm.cc
    transport/net.cc
    transport/net_socket.cc
    transport/net_ib.cc
    transport/coll_net.cc
    # collectives/sendrecv.cc
    # collectives/all_reduce.cc
    # collectives/all_gather.cc
    # collectives/broadcast.cc
    # collectives/reduce.cc
    # collectives/reduce_scatter.cc
    graph/topo.cc
    graph/paths.cc
    graph/search.cc
    graph/connect.cc
    graph/rings.cc
    graph/trees.cc
    graph/tuning.cc
    graph/xml.cc)

set(INCEXPORTS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/nccl.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/nccl_net.h
    PARENT_SCOPE)

set(NCCL_LIBS nccl;nccl_static)

add_library(nccl SHARED ${SRC_FILES} $<TARGET_OBJECTS:colldevice>)
add_library(nccl_static STATIC ${SRC_FILES} $<TARGET_OBJECTS:colldevice>)

foreach(lib_name IN LISTS NCCL_LIBS)
  set_property(TARGET ${lib_name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET ${lib_name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_include_directories(
    ${lib_name} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                       $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_include_directories(${lib_name}
                             PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endforeach()
