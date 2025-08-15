# Register sources
set(REGISTER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/register.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/coll_reg.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/sendrecv_reg.cc
)

# Add register sources to parent scope
set(REGISTER_SOURCES ${REGISTER_SOURCES} PARENT_SCOPE)
