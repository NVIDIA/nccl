# Register sources
set(SYM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/core.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ll_a2a.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/lsa_barrier.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/gin_barrier.cc
)

# Add register sources to parent scope
set(SYM_SOURCES ${SYM_SOURCES} PARENT_SCOPE)
