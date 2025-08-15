# Net plugin sources
set(PLUGIN_NET_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/net_v9.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/net_v6.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/net_v7.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/net_v8.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/net_v10.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/net_v11.cc
)

# Add net plugin sources to parent scope
set(PLUGIN_NET_SOURCES ${PLUGIN_NET_SOURCES} PARENT_SCOPE)
