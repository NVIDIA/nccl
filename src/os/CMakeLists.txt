# if(NCCL_OS_LINUX) # @EUGO_CHANGE: We moved all sources to `src/transport/CMakeLists.txt` for better modularity and maintainability. --- IGNORE ---
#     set(OS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/linux.cc PARENT_SCOPE)
# elseif(NCCL_OS_WINDOWS)
#     set(OS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/windows.cc PARENT_SCOPE)
# endif()
