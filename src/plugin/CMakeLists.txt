# Add plugin subdirectories
add_subdirectory(profiler)
add_subdirectory(net)
add_subdirectory(tuner)
add_subdirectory(env)

# Plugin sources
set(PLUGIN_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/net.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/profiler.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin_open.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tuner.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/env.cc
    ${PLUGIN_NET_SOURCES}
    ${PLUGIN_PROFILER_SOURCES}
    ${PLUGIN_TUNER_SOURCES}
    ${PLUGIN_ENV_SOURCES}
)

# Add plugin sources to parent scope
set(PLUGIN_SOURCES ${PLUGIN_SOURCES} PARENT_SCOPE)
