# Gin sources
set(GIN_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/gin_host.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/gin_host_proxy.cc
)

# Add gin sources to parent scope
set(GIN_SOURCES ${GIN_SOURCES} PARENT_SCOPE)
