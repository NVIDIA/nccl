# RAS sources - Linux-only (uses POSIX sockets and signals)
if(NOT WIN32)
    set(RAS_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/collectives.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/rasnet.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/peers.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ras.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/client_support.cc
    )
else()
    # Windows stub implementation
    set(RAS_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/ras_win32.cc
    )
endif()

# Add RAS sources to parent scope
set(RAS_SOURCES ${RAS_SOURCES} PARENT_SCOPE)
