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
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate.py ${CMAKE_CURRENT_BINARY_DIR}/gensrc ${ONLY_FUNCS}
    OUTPUT_VARIABLE COLLDEVICE_GENSRC_FILES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
string(STRIP "${COLLDEVICE_GENSRC_FILES}" COLLDEVICE_GENSRC_FILES)
list(TRANSFORM COLLDEVICE_GENSRC_FILES PREPEND ${CMAKE_CURRENT_BINARY_DIR}/gensrc/)
list(FILTER COLLDEVICE_GENSRC_FILES INCLUDE REGEX "\\.cu$") # We need to keep only .cu **files** (there are generated .mk files and redundant/duplicative folders w/ .cu files we filter out)
list(APPEND COLLDEVICE_SRC_FILES ${COLLDEVICE_GENSRC_FILES})

execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/generate.py ${CMAKE_CURRENT_BINARY_DIR}/gensrc/symmetric
    OUTPUT_VARIABLE COLLDEVICE_GENSRC_SYMMETRIC_FILES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
string(STRIP "${COLLDEVICE_GENSRC_SYMMETRIC_FILES}" COLLDEVICE_GENSRC_SYMMETRIC_FILES)
list(TRANSFORM COLLDEVICE_GENSRC_SYMMETRIC_FILES PREPEND ${CMAKE_CURRENT_BINARY_DIR}/gensrc/symmetric/)
list(FILTER COLLDEVICE_GENSRC_SYMMETRIC_FILES INCLUDE REGEX "\\.cu$") # We need to keep only .cu **files** (there are generated .mk files and redundant/duplicative folders w/ .cu files we filter out)
list(APPEND COLLDEVICE_SRC_FILES ${COLLDEVICE_GENSRC_SYMMETRIC_FILES})
# @NVIDIA_ORIGINAL: @end
# === @end: Generated Sources ===


# === @begin: `libnccl_colldevice` library ===
# @EUGO_CHANGE:
# 1. `nccl_device` -> `nccl_colldevice` - we renamed this library for clarity as they already have `@/src/nccl_device` folder which is not used for this library at all.
# 2. `OBJECT` -> `STATIC` - we changed the type of this sublibrary, as otherwise `CMake` uses `nvlink` for non-CUDA C++ sources, too, disabling most our optimizations like `LTO` as unsupported.
add_library(nccl_colldevice STATIC ${COLLDEVICE_SRC_FILES})
# message(WARNING "[EUGO]: ${COLLDEVICE_SRC_FILES}")

target_include_directories(
    nccl_colldevice
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_SOURCE_DIR}/src/include
        ${CMAKE_SOURCE_DIR}/src/include/plugin
        ${CONFIGURED_HEADERS_DIR}
)

target_compile_definitions(nccl_colldevice PRIVATE ${NCCL_COMMON_COMPILE_DEFINITIONS})

# @EUGO_CHANGE:
# 1. `CUDAToolkit_INCLUDE_DIRS/cccl` (this folder even doesn't exist modern days) -> `target_link_libraries(nccl_colldevice PRIVATE ${NCCL_DEPENDENCIES})`
# 2. `CUDAToolkit_INCLUDE_DIRS` -> `target_link_libraries(nccl_colldevice PRIVATE ${NCCL_DEPENDENCIES})`
target_link_libraries(nccl_colldevice PRIVATE ${NCCL_DEPENDENCIES})
# === @end: `libnccl_colldevice` library ===
