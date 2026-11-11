if(NCCL_OS_LINUX)
    set(OS_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/linux.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/linux_ipcsocket.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/linux_socket_pair.cc
    )
elseif(NCCL_OS_WINDOWS)
    set(OS_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/windows.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/windows_stubs.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/windows_ipcsocket.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/windows_socket_pair.cc
    )
endif()

set(OS_LINK_OPTIONS)
# Add compatibility shim if using static cudart
if(CUDARTLIB STREQUAL "cudart_static")
    if(NCCL_OS_LINUX)
        list(APPEND OS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/linux_enhcompat.cc)
    elseif(NCCL_OS_WINDOWS)
        list(APPEND OS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/windows_enhcompat.cc)
        # Load the shim object so the linker sees its fallback aliases.
        set(OS_LINK_OPTIONS "$<HOST_LINK:/INCLUDE:ncclCudaEnhcompatAnchor>")
    endif()
endif()

set(OS_SOURCES ${OS_SOURCES} PARENT_SCOPE)
set(OS_LINK_OPTIONS ${OS_LINK_OPTIONS} PARENT_SCOPE)
