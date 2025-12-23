# Transport sources

set(TRANSPORT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/nvls.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/profiler.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/net_socket.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/p2p.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/net.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_net.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/shm.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/generic.cc
)

add_subdirectory(net_ib)

# Add transport sources to parent scope
set(TRANSPORT_SOURCES ${TRANSPORT_SOURCES} PARENT_SCOPE)

# Add DOCA device headers to parent scope
set(DEVICE_DOCA_HEADERS ${DEVICE_DOCA_HEADERS} PARENT_SCOPE)

# Add DOCA sources to parent scope
set(DOCA_SOURCES ${DOCA_SOURCES} PARENT_SCOPE)

# Add DOCA_HOME to parent scope
set(DOCA_HOME ${DOCA_HOME} PARENT_SCOPE)
