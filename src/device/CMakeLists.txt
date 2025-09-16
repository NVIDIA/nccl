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
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate.py ${CMAKE_CURRENT_BINARY_DIR}/gensrc "${ONLY_FUNCS}"
    OUTPUT_VARIABLE COLLDEVICE_GENSRC_FILES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
string(STRIP "${COLLDEVICE_GENSRC_FILES}" COLLDEVICE_GENSRC_FILES)
list(TRANSFORM COLLDEVICE_GENSRC_FILES PREPEND ${CMAKE_CURRENT_BINARY_DIR}/gensrc/)
list(APPEND COLLDEVICE_SRC_FILES ${COLLDEVICE_GENSRC_FILES})

execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/generate.py ${CMAKE_CURRENT_BINARY_DIR}/gensrc/symmetric "${ONLY_FUNCS}"
    OUTPUT_VARIABLE COLLDEVICE_GENSRC_SYMMETRIC_FILES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
string(STRIP "${COLLDEVICE_GENSRC_SYMMETRIC_FILES}" COLLDEVICE_GENSRC_SYMMETRIC_FILES)
list(TRANSFORM COLLDEVICE_GENSRC_SYMMETRIC_FILES PREPEND ${CMAKE_CURRENT_BINARY_DIR}/gensrc/symmetric/)
list(APPEND COLLDEVICE_SRC_FILES ${COLLDEVICE_GENSRC_SYMMETRIC_FILES})
# @NVIDIA_ORIGINAL: @end
# === @end: Generated Sources ===


# === @begin: nccl_colldevice library ===
# @EUGO_CHANGE:
# `nccl_device` -> `nccl_colldevice` - we renamed this library for clarity as they already have `@/src/nccl_device` folder which is not used for this library at all.
#message(FATAL_ERROR "[EUGO]: ${COLLDEVICE_SRC_FILES}")
add_library(nccl_colldevice OBJECT ${COLLDEVICE_SRC_FILES})

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
# 1. `CUDAToolkit_INCLUDE_DIRS/cccl` (this folder even doesn't exist modern days) -> `target_link_libraries(nccl_colldevice PRIVATE CCCL::CCCL)`
# 2. `CUDAToolkit_INCLUDE_DIRS` -> @HELP(today) does it really use CUDA headers other than CCCL? We'll need to check - one question if it relies on low-level CUDA headers which are injected by the compiler and another case if it depends - e.g., on `libcupti` headers which are considered standalone headers and aren't injected automatically.
target_link_libraries(nccl_colldevice PRIVATE CCCL::CCCL)
# === @end: nccl_colldevice library ===
