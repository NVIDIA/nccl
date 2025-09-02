# Tuner plugin sources
set(PLUGIN_TUNER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/tuner_v2.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tuner_v3.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tuner_v4.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tuner_v5.cc
)

# Add tuner plugin sources to parent scope
set(PLUGIN_TUNER_SOURCES ${PLUGIN_TUNER_SOURCES} PARENT_SCOPE)
