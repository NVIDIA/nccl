# === @begin: Non-Generated Sources ===
set(
   COLLDEVICE_SRC_FILES
   common.cu
   onerank.cu
)
# === @end: Non-Generated Sources ===


# === @begin: Generated Sources ===
execute_process(
    COMMAND
    ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/generate.py
    ${CMAKE_CURRENT_BINARY_DIR}
)

file(GLOB GENERATED_COLLDEVICE_SRC_FILES RELATIVE ${CMAKE_CURRENT_BINARY_DIR} "${CMAKE_CURRENT_BINARY_DIR}/*.cu")

list(APPEND COLLDEVICE_SRC_FILES ${GENERATED_COLLDEVICE_SRC_FILES})
# === @end: Generated Sources ===


# === @begin: colldevice library ===
add_library(colldevice OBJECT ${COLLDEVICE_SRC_FILES})

target_include_directories(
    colldevice
    PRIVATE
         ${CMAKE_CURRENT_SOURCE_DIR}/../include
         ${CMAKE_CURRENT_SOURCE_DIR}
         ${CMAKE_CURRENT_BINARY_DIR}
         ${CONFIGURED_HEADERS_DIR}
)

# Link colldevice privately to nvtx3
target_link_libraries(colldevice PRIVATE CUDA::nvtx3)

set_property(TARGET colldevice PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET colldevice PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
# === @end: colldevice library ===
