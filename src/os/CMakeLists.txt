if(NCCL_OS_LINUX)
    set(OS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/linux.cc PARENT_SCOPE)
elseif(NCCL_OS_WINDOWS)
    set(OS_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/windows.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/windows_stubs.cc
        PARENT_SCOPE)
endif()
