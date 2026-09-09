# Config sources
set(CONFIG_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/collconfig.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/algorithm_registry.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/algorithm_parser.cc
)

# Add config sources to parent scope
set(CONFIG_SOURCES ${CONFIG_SOURCES} PARENT_SCOPE)
