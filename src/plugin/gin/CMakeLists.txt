# Plugin sources
set(PLUGIN_GIN_SOURCES
${CMAKE_CURRENT_SOURCE_DIR}/gin_v13.cc
${CMAKE_CURRENT_SOURCE_DIR}/gin_v14.cc
)

# Add plugin sources to parent scope
set(PLUGIN_GIN_SOURCES ${PLUGIN_GIN_SOURCES} PARENT_SCOPE)
