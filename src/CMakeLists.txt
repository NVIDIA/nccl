# === @begin: Headers Configuration ===
set(nccl_Major ${nccl_VERSION_MAJOR})
set(nccl_Minor ${nccl_VERSION_MINOR})
set(nccl_Patch ${nccl_VERSION_PATCH})
# NCCL_VERSION(X,Y,Z) ((X) * 10000 + (Y) * 100 + (Z))
math(
    EXPR
    nccl_Version
    "${nccl_VERSION_MAJOR} * 10000 + ${nccl_VERSION_MINOR} * 100 + ${nccl_VERSION_PATCH}"
)

# By default, it's an empty string, but we still must define it otherwise `configure_file` will fail.
# In theory, one who builds this package may define that to unique value to be used in a `nccl` version string.
# We, however, don't do that in Eugo environment and keep it mostly for compatability with original Makefile builds.
set(nccl_Suffix "")

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/nccl.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/include/nccl.h
)
# === @end: Headers Configuration ===


# === @begin: device/ ===
add_subdirectory(device)
# === @end: device/ ===


# === @begin: ras/ ===
add_subdirectory(ras)
# === @end: ras/ ===

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

    # === @begin: misc/ ===
    misc/argcheck.cc
    misc/cudawrap.cc
    misc/gdrwrap.cc
    misc/ibvsymbols.cc
    misc/ibvwrap.cc
    misc/ipcsockert.cc
    misc/nvmlwrap.cc
    misc/param.cc
    misc/shmutils.cc
    misc/socket.cc
    misc/strongstream.cc
    misc/utils.cc
    # === @end: misc/ ===

    # === @begin: plugin/ ===
    plugin/net/net_v6.cc
    plugin/net/net_v7.cc
    plugin/net/net_v8.cc
    plugin/net/net_v9.cc
    plugin/net/net_v10.cc
    plugin/profiler/profiler_v1.cc
    plugin/profiler/profiler_v2.cc
    plugin/profiler/profiler_v3.cc
    plugin/tuner/tuner_v1.cc
    plugin/tuner/tuner_v2.cc
    plugin/tuner/tuner_v3.cc
    plugin/net.cc
    plugin/plugin_open.cc
    plugin/profiler.cc
    plugin/tuner.cc
    # === @end: plugin/ ===

    # === @begin: transport/ ===
    transport/coll_net.cc
    transport/generic.cc
    transport/net_ib.cc
    transport/net_socket.cc
    transport/net.cc
    transport/nvls.cc
    transport/p2p.cc
    transport/profiler.cc
    transport/shm.cc
    # === @end: transport/ ===

    # === @being: graph/ ===

    # === @@being: Header Files ===
    graph/rings.h
    graph/topo.h
    graph/xml.h
    # === @@end: Header Files ===

    # === @@begin: Source Files ===
    graph/connect.cc
    graph/paths.cc
    graph/rings.cc
    graph/search.cc
    graph/topo.cc
    graph/trees.cc
    graph/tuning.cc
    graph/xml.cc
    # === @@end: Source Files ===
    # === @end: graph/ ===

    # === @begin: register/ ===
    coll_reg.cc
    register.cc
    sendrecv_reg.cc
    # === @end: register/ ===
)

# set(INCEXPORTS
#     ${CMAKE_CURRENT_SOURCE_DIR}/include/nccl.h
#     ${CMAKE_CURRENT_SOURCE_DIR}/include/nccl_net.h
#     PARENT_SCOPE
# )

# set(NCCL_LIBS nccl;nccl_static)

# add_library(nccl SHARED ${SRC_FILES} $<TARGET_OBJECTS:colldevice>)
# add_library(nccl_static STATIC ${SRC_FILES} $<TARGET_OBJECTS:colldevice>)

# foreach(lib_name IN LISTS NCCL_LIBS)
#     set_property(TARGET ${lib_name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#     set_property(TARGET ${lib_name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
#     target_include_directories(
#         ${lib_name}
#         PUBLIC
#             $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#             $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
#     )
#     target_include_directories(
#         ${lib_name}
#         PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
#     )
#     target_link_libraries(${lib_name} PRIVATE ncclras)
# endforeach()

# # Link nvtx
# if(USE_SYSTEM_NVTX)
#     target_link_libraries(${lib_name} PRIVATE CUDA::nvtx3)
# else()
#     # Disable NVTX-related features
#     add_compile_definitions(NVTX_DISABLE)
#     message(STATUS "NVTX features are disabled.")
# endif()