# Env plugin sources
set(PLUGIN_ENV_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/env_v1.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/env_v2.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/default_env.cc
)

# Env plugin sources to parent scope
set(PLUGIN_ENV_SOURCES ${PLUGIN_ENV_SOURCES} PARENT_SCOPE)
