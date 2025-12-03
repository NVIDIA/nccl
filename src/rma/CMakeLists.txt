# RMA sources
set(RMA_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/rma.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/rma_proxy.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/rma_ce.cc
)

# Add RMA sources to parent scope
set(RMA_SOURCES ${RMA_SOURCES} PARENT_SCOPE)
