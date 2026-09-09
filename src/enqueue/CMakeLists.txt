# Enqueue sources
set(ENQUEUE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/enqueue.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/mgmt_task_enq.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/task_prep/task_prep.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/task_prep/task_pretuning.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/task_prep/task_classify.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/task_prep/task_posttuning.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/task_sched/task_sched.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/task_sched/sym_sched.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/task_sched/legacy_sched.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/task_sched/allgatherv_sched.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/task_sched/p2p_sched.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/task_sched/rma_sched.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/task_sched/ce_sched.cc
)

# Add enqueue sources to parent scope
set(ENQUEUE_SOURCES ${ENQUEUE_SOURCES} PARENT_SCOPE)
