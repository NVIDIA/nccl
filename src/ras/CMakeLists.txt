# RAS sources
set(RAS_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/collectives.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/diagnostics.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/diagnostics_checks_common.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/diagnostics_env.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/diagnostics_gpu.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/rasnet.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/peers.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ras.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ras_param.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/client_support.cc
)

# Add RAS sources to parent scope
set(RAS_SOURCES ${RAS_SOURCES} PARENT_SCOPE)
