# === @begin: Non-Generated Sources ===
set(
   CU_FILES
   common.cu
   onerank.cu
)
# === @end: Non-Generated Sources ===


# === @begin: Generated Sources ===
set(GENERATED_CU_SOURCES_DIR "${CMAKE_BINARY_DIR}/eugo/src/device")
file(MAKE_DIRECTORY "${GENERATED_DIR}")

execute_process(
    COMMAND
    ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/generate.py
    ${GENERATED_CU_SOURCES_DIR}
)

file(GLOB GENERATED_CU_FILES RELATIVE ${GENERATED_CU_SOURCES_DIR} "${GENERATED_CU_SOURCES_DIR}/*.cu")

list(APPEND CU_FILES ${GENERATED_CU_FILES})
# === @end: Generated Sources ===


# === @begin: colldevice library ===
add_library(colldevice OBJECT ${CU_FILES})

target_include_directories(
    colldevice
    PRIVATE
         ${CMAKE_CURRENT_SOURCE_DIR}/../include
         ${CMAKE_CURRENT_SOURCE_DIR}
         ${GENERATED_DIR}
)

set_property(TARGET colldevice PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET colldevice PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
# === @end: colldevice library ===
