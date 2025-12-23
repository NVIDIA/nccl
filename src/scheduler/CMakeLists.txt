# Scheduler sources
set(SCHEDULER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/symmetric_sched.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/allgatherv_sched.cc
)

# Add scheduler sources to parent scope
set(SCHEDULER_SOURCES ${SCHEDULER_SOURCES} PARENT_SCOPE)
