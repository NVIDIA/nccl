# Proxy GPU context sources
set(GIN_SOURCES ${GIN_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/gpucontext.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/gpucontext_v1.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/gpucontext_v2.cc
    PARENT_SCOPE)
