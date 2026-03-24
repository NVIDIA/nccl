# RAS is a Linux-specific subsystem; exclude from Windows builds
if(NCCL_OS_LINUX)
set(RAS_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/collectives.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/rasnet.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/peers.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ras.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/client_support.cc
)
endif()

# Add RAS sources to parent scope
set(RAS_SOURCES ${RAS_SOURCES} PARENT_SCOPE)
