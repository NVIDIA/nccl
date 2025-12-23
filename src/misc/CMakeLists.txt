# Misc sources
set(MISC_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/strongstream.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/socket.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ibvwrap.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/mlx5dvsymbols.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/mlx5dvwrap.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/cudawrap.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/param.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ipcsocket.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/shmutils.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/nvmlwrap.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/argcheck.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/gdrwrap.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/ibvsymbols.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/git_version.cc
)

# Add misc sources to parent scope
set(MISC_SOURCES ${MISC_SOURCES} PARENT_SCOPE)
