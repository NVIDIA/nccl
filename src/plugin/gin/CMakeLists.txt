# Gin plugin sources
set(PLUGIN_GIN_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/gin_v11.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/gin_v12.cc
)

# Add net plugin sources to parent scope
set(PLUGIN_GIN_SOURCES ${PLUGIN_GIN_SOURCES} PARENT_SCOPE)
