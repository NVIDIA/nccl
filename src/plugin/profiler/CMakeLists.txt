# Profiler plugin sources
set(PLUGIN_PROFILER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/profiler_v3.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/profiler_v4.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/profiler_v1.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/profiler_v2.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/profiler_v5.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/profiler_v6.cc
)

# Add profiler plugin sources to parent scope
set(PLUGIN_PROFILER_SOURCES ${PLUGIN_PROFILER_SOURCES} PARENT_SCOPE)
