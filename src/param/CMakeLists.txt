# Param sources
set(PARAM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/c_api.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/param.cc
)

# Add param sources to parent scope
set(PARAM_SOURCES ${PARAM_SOURCES} PARENT_SCOPE)
