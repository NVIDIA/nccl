# NOTE: Make sure that platform supports IBV extension(s), do not assume it.
if(RDMA_CORE)
    include(CheckCSourceCompiles)

    check_c_source_compiles("
    #include <infiniband/verbs.h>
    int main(void) {
        struct ibv_port_attr a;
        a.active_speed_ex = 0;
        return 0;
    }
    " IBV_HAS_ACTIVE_SPEED_EX)
else()
    # Stubs do have support for it.
    set(IBV_HAS_ACTIVE_SPEED_EX ON)
endif ()
if(IBV_HAS_ACTIVE_SPEED_EX)
    add_compile_definitions(IBV_HAS_ACTIVE_SPEED_EX=1)
else()
    add_compile_definitions(IBV_HAS_ACTIVE_SPEED_EX=0)
endif()


# Transport sources
set(NET_IB_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/common.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/connect.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/gdr.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/init.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/p2p.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/p2p_resiliency.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/p2p_resiliency_recovery.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/reg.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/gin.cc
)

add_subdirectory(gdaki)

# Add transport sources to parent scope
set(TRANSPORT_SOURCES ${TRANSPORT_SOURCES} ${NET_IB_SOURCES} PARENT_SCOPE)

# Add DOCA device headers to parent scope
set(DEVICE_DOCA_HEADERS ${DEVICE_DOCA_HEADERS} PARENT_SCOPE)

# Add DOCA sources to parent scope
set(DOCA_SOURCES ${DOCA_SOURCES} PARENT_SCOPE)

# Add DOCA_HOME to parent scope
set(DOCA_HOME ${DOCA_HOME} PARENT_SCOPE)
