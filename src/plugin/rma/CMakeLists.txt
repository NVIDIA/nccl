# Plugin sources
set(PLUGIN_RMA_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/rma_v14.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/rma_v13.cc
)

# Add plugin sources to parent scope
set(PLUGIN_RMA_SOURCES ${PLUGIN_RMA_SOURCES} PARENT_SCOPE)
