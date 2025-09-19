# === @begin: Non-Generated Sources ===
set(COLLDEVICE_SRC_FILES
   common.cu
   onerank.cu
)
# === @end: Non-Generated Sources ===


# === @begin: Generated Sources ===
# @NVIDIA_ORIGINAL: @begin:
# @EUGO_CHANGE:
# In the original implementation, they first execute scripts (`execute_process`) at configuration-time to get the files list AND then use this list for buildtime where via `add_custom_command` they generate these files again.
# This doesn't make sense as the latter is being executed with the same arguments. More correctly for them will be to use `GLOB` or whatever, but anyway, we already have our approach. Torch does it at buildtime w/o configuration time tricks.
# The issue w/ their approach is that files are stored in exactly the same directory in both cases. As a result, CMake and Ninja always detect that files changed since configuration time and rerun it to only realize that files changed since the buildtime and rerun it to only realize ... (recursion).
#
# @EUGO_CHANGE:
# 1. `files` -> `COLLDEVICE_GENSRC_FILES`
# 2. `symmetric_files` -> `COLLDEVICE_GENSRC_SYMMETRIC_FILES`
# set(ONLY_FUNCS "SendRecv") # @HELP # TODO+ remove me!!!
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate.py ${CMAKE_CURRENT_BINARY_DIR}/gensrc
    OUTPUT_VARIABLE COLLDEVICE_GENSRC_FILES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
string(STRIP "${COLLDEVICE_GENSRC_FILES}" COLLDEVICE_GENSRC_FILES)
list(TRANSFORM COLLDEVICE_GENSRC_FILES PREPEND ${CMAKE_CURRENT_BINARY_DIR}/gensrc/)
list(APPEND COLLDEVICE_SRC_FILES ${COLLDEVICE_GENSRC_FILES})
# @HELP: Clean unused/dangerous paths! .mk and entire dirs

execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/generate.py ${CMAKE_CURRENT_BINARY_DIR}/gensrc/symmetric
    OUTPUT_VARIABLE COLLDEVICE_GENSRC_SYMMETRIC_FILES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
string(STRIP "${COLLDEVICE_GENSRC_SYMMETRIC_FILES}" COLLDEVICE_GENSRC_SYMMETRIC_FILES)
list(TRANSFORM COLLDEVICE_GENSRC_SYMMETRIC_FILES PREPEND ${CMAKE_CURRENT_BINARY_DIR}/gensrc/symmetric/)
list(APPEND COLLDEVICE_SRC_FILES ${COLLDEVICE_GENSRC_SYMMETRIC_FILES})
# @HELP: Clean unused/dangerous paths! .mk and entire dirs
# @NVIDIA_ORIGINAL: @end

# @HELP+: remove after tests
#file(GLOB_RECURSE GENERATED_COLLDEVICE_CU_FILES RELATIVE ${CMAKE_CURRENT_BINARY_DIR} "${CMAKE_CURRENT_BINARY_DIR}/*.cu")
# file(GLOB GENERATED_COLLDEVICE_CU_FILES RELATIVE ${CMAKE_CURRENT_BINARY_DIR} "${CMAKE_CURRENT_BINARY_DIR}/**/*.cu")
#list(APPEND COLLDEVICE_SRC_FILES ${GENERATED_COLLDEVICE_CU_FILES})
# list(APPEND COLLDEVICE_SRC_FILES ${COLLDEVICE_GENSRC_SYMMETRIC_FILES})
# === @end: Generated Sources ===


# === @begin: nccl_colldevice library ===
# @EUGO_CHANGE:
# `nccl_device` -> `nccl_colldevice` - we renamed this library for clarity as they already have `@/src/nccl_device` folder which is not used for this library at all.
add_library(nccl_colldevice STATIC ${COLLDEVICE_SRC_FILES})
message(WARNING "[EUGO]: ${COLLDEVICE_SRC_FILES}")

target_include_directories(
    nccl_colldevice
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_SOURCE_DIR}/src/include
        ${CMAKE_SOURCE_DIR}/src/include/plugin
        ${CONFIGURED_HEADERS_DIR}
)


# @EUGO_CHANGE:
# 1. `CUDAToolkit_INCLUDE_DIRS/cccl` (this folder even doesn't exist modern days) -> `target_link_libraries(nccl_colldevice PRIVATE ${NCCL_CUDA_DEPENDENCIES})`
# 2. `CUDAToolkit_INCLUDE_DIRS` -> `target_link_libraries(nccl_colldevice PRIVATE ${NCCL_CUDA_DEPENDENCIES})`
target_link_libraries(nccl_colldevice PRIVATE ${NCCL_CUDA_DEPENDENCIES})
# === @end: nccl_colldevice library ===


