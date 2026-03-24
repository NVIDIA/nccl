# Register sources (GIN sources are Linux-only)
set(SYM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/core.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ll_a2a.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/lsa_barrier.cc
)
if(NCCL_OS_LINUX)
    list(APPEND SYM_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/gin_barrier.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/gin_scratch.cc
    )
endif()

# Add register sources to parent scope
set(SYM_SOURCES ${SYM_SOURCES} PARENT_SCOPE)
