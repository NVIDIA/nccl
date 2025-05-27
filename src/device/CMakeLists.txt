# === @begin: Non-Generated Sources ===
 set(
   CU_FILES
   common.cu
   onerank.cu
)
# === @end: Non-Generated Sources ===


# === @begin: Generated Sources ===
set(GENERATED_CU_SOURCES_DIR "${CMAKE_BINARY_DIR}/src/device")
file(MAKE_DIRECTORY "${GENERATED_DIR}")

execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate.py ${GENERATED_CU_SOURCES_DIR}
    WORKING_DIRECTORY ${GENERATED_CU_SOURCES_DIR}
)

file(GLOB GENERATED_CU_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")


# To generate .cu source files using an external Python script in CMake with a temporary build-directory outside the source tree, follow this structured approach:

# ⸻

# ✅ Step-by-step CMake Setup

# 1. Create a dedicated temporary directory

# Use file(MAKE_DIRECTORY ...) to create a temporary directory in the build tree:

# set(GENERATED_DIR "${CMAKE_BINARY_DIR}/generated_cuda")
# file(MAKE_DIRECTORY "${GENERATED_DIR}")

# 2. Invoke the Python script

# Use add_custom_command() to invoke your external script and generate .cu files:

# add_custom_command(
#   OUTPUT "${GENERATED_DIR}/stamp"
#   COMMAND ${CMAKE_COMMAND} -E echo "Generating CUDA files..."
#   COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/scripts/generate_cuda.py" "${GENERATED_DIR}"
#   COMMAND ${CMAKE_COMMAND} -E touch "${GENERATED_DIR}/stamp"
#   COMMENT "Running CUDA code generator"
#   DEPENDS "${CMAKE_SOURCE_DIR}/scripts/generate_cuda.py"
# )

# 	•	You generate a stamp file to use as a stable dependency and OUTPUT.
# 	•	${Python3_EXECUTABLE} is assumed to be available via find_package(Python3 COMPONENTS Interpreter REQUIRED)

# 3. Glob .cu files

# After generation, you can safely glob .cu files from the GENERATED_DIR:

# file(GLOB GENERATED_CUDA_SOURCES "${GENERATED_DIR}/*.cu")

# 4. Add a library or executable

# Ensure your target depends on the generation step:

# add_library(my_cuda_lib STATIC ${GENERATED_CUDA_SOURCES})
# add_dependencies(my_cuda_lib generate_cuda_code)

# Note: You should wrap the add_custom_command() inside a add_custom_target(generate_cuda_code ...) to make it an actual build step:

# add_custom_target(generate_cuda_code
#   DEPENDS "${GENERATED_DIR}/stamp"
# )


# ⸻

# 🔁 Full Example

# cmake_minimum_required(VERSION 3.18)
# project(cuda_codegen LANGUAGES CXX CUDA)

# find_package(Python3 COMPONENTS Interpreter REQUIRED)

# set(GENERATED_DIR "${CMAKE_BINARY_DIR}/generated_cuda")
# file(MAKE_DIRECTORY "${GENERATED_DIR}")

# add_custom_command(
#   OUTPUT "${GENERATED_DIR}/stamp"
#   COMMAND ${CMAKE_COMMAND} -E echo "Generating CUDA files..."
#   COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/scripts/generate_cuda.py" "${GENERATED_DIR}"
#   COMMAND ${CMAKE_COMMAND} -E touch "${GENERATED_DIR}/stamp"
#   COMMENT "Running CUDA code generator"
#   DEPENDS "${CMAKE_SOURCE_DIR}/scripts/generate_cuda.py"
# )

# add_custom_target(generate_cuda_code
#   DEPENDS "${GENERATED_DIR}/stamp"
# )

# file(GLOB GENERATED_CUDA_SOURCES "${GENERATED_DIR}/*.cu")

# add_library(my_cuda_lib STATIC ${GENERATED_CUDA_SOURCES})
# add_dependencies(my_cuda_lib generate_cuda_code)


# ⸻

# 🧪 Tips
# 	•	Always use CMAKE_BINARY_DIR or CMAKE_CURRENT_BINARY_DIR to ensure the output is outside your source tree.
# 	•	Avoid file(GLOB ...) for critical build logic in large systems unless you are confident in the directory’s contents being stable across configures. Otherwise, generate a list file from the Python script and read it via file(READ) or include().

# Let me know if you want the Python script to also emit a .cmake file with the list of generated sources.
# === @end: Generated Sources ===



# set(
#   CU_FILES
#   device_table.cu
#   all_gather.cu
#   broadcast.cu
#   sendrecv.cu

#   all_reduce_minmax_bf16.cu
#   all_reduce_minmax_f16.cu
#   all_reduce_minmax_f32.cu
#   all_reduce_minmax_f64.cu
#   all_reduce_minmax_i32.cu
#   all_reduce_minmax_i64.cu
#   all_reduce_minmax_u32.cu
#   all_reduce_minmax_u64.cu
#   all_reduce_minmax_u8.cu
#   all_reduce_premulsum_bf16.cu
#   all_reduce_premulsum_f16.cu
#   all_reduce_premulsum_f32.cu
#   all_reduce_premulsum_f64.cu
#   all_reduce_premulsum_u32.cu
#   all_reduce_premulsum_u64.cu
#   all_reduce_premulsum_u8.cu
#   all_reduce_prod_bf16.cu
#   all_reduce_prod_f16.cu
#   all_reduce_prod_f32.cu
#   all_reduce_prod_f64.cu
#   all_reduce_prod_u32.cu
#   all_reduce_prod_u64.cu
#   all_reduce_prod_u8.cu
#   all_reduce_sum_bf16.cu
#   all_reduce_sum_f16.cu
#   all_reduce_sum_f32.cu
#   all_reduce_sum_f64.cu
#   all_reduce_sum_u32.cu
#   all_reduce_sum_u64.cu
#   all_reduce_sum_u8.cu
#   all_reduce_sumpostdiv_i32.cu
#   all_reduce_sumpostdiv_i64.cu
#   all_reduce_sumpostdiv_i8.cu
#   all_reduce_sumpostdiv_u32.cu
#   all_reduce_sumpostdiv_u64.cu
#   all_reduce_sumpostdiv_u8.cu
#   reduce_minmax_bf16.cu
#   reduce_minmax_f16.cu
#   reduce_minmax_f32.cu
#   reduce_minmax_f64.cu
#   reduce_minmax_u32.cu
#   reduce_minmax_u64.cu
#   reduce_minmax_u8.cu
#   reduce_premulsum_bf16.cu
#   reduce_premulsum_f16.cu
#   reduce_premulsum_f32.cu
#   reduce_premulsum_f64.cu
#   reduce_premulsum_u32.cu
#   reduce_premulsum_u64.cu
#   reduce_premulsum_u8.cu
#   reduce_prod_bf16.cu
#   reduce_prod_f16.cu
#   reduce_prod_f32.cu
#   reduce_prod_f64.cu
#   reduce_prod_u32.cu
#   reduce_prod_u64.cu
#   reduce_prod_u8.cu
#   reduce_scatter_minmax_bf16.cu
#   reduce_scatter_minmax_f16.cu
#   reduce_scatter_minmax_f32.cu
#   reduce_scatter_minmax_f64.cu
#   reduce_scatter_minmax_i32.cu
#   reduce_scatter_minmax_i64.cu
#   reduce_scatter_minmax_u32.cu
#   reduce_scatter_minmax_u64.cu
#   reduce_scatter_minmax_u8.cu
#   reduce_scatter_premulsum_bf16.cu
#   reduce_scatter_premulsum_f16.cu
#   reduce_scatter_premulsum_f32.cu
#   reduce_scatter_premulsum_f64.cu
#   reduce_scatter_premulsum_u32.cu
#   reduce_scatter_premulsum_u64.cu
#   reduce_scatter_premulsum_u8.cu
#   reduce_scatter_prod_bf16.cu
#   reduce_scatter_prod_f16.cu
#   reduce_scatter_prod_f32.cu
#   reduce_scatter_prod_f64.cu
#   reduce_scatter_prod_u32.cu
#   reduce_scatter_prod_u64.cu
#   reduce_scatter_prod_u8.cu
#   reduce_scatter_sum_bf16.cu
#   reduce_scatter_sum_f16.cu
#   reduce_scatter_sum_f32.cu
#   reduce_scatter_sum_f64.cu
#   reduce_scatter_sum_u32.cu
#   reduce_scatter_sum_u64.cu
#   reduce_scatter_sum_u8.cu
#   reduce_scatter_sumpostdiv_i32.cu
#   reduce_scatter_sumpostdiv_i64.cu
#   reduce_scatter_sumpostdiv_i8.cu
#   reduce_scatter_sumpostdiv_u32.cu
#   reduce_scatter_sumpostdiv_u64.cu
#   reduce_scatter_sumpostdiv_u8.cu
#   reduce_sum_bf16.cu
#   reduce_sum_f16.cu
#   reduce_sum_f32.cu
#   reduce_sum_f64.cu
#   reduce_sum_u32.cu
#   reduce_sum_u64.cu
#   reduce_sum_u8.cu
#   reduce_sumpostdiv_i32.cu
#   reduce_sumpostdiv_i64.cu
#   reduce_sumpostdiv_i8.cu
#   reduce_sumpostdiv_u32.cu
#   reduce_sumpostdiv_u64.cu
#   reduce_sumpostdiv_u8.cu
# )

# # Create an empty list of CU_FILES to store generated files
# set(CU_FILES "")

# # Define reusable temp directory
# set(EUGO_TMP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/__eugo_tmp")

# # Step 1: Create temp dir and run generate.py
# execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${EUGO_TMP_DIR})

# execute_process(
#     COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate.py
#     WORKING_DIRECTORY ${EUGO_TMP_DIR}
# )

# # Step 2: Get list of generated .cu files
# file(GLOB GENERATED_CU_FILES RELATIVE ${EUGO_TMP_DIR} "${EUGO_TMP_DIR}/*.cu")

# # Step 3: Append to CU_FILES
# foreach(file ${GENERATED_CU_FILES})
#     list(APPEND CU_FILES "${EUGO_TMP_DIR}/${file}")
# endforeach()

# # Step 3.5: Patch device_table.cu to remove all instances of 'const'
# set(GENERATED_DEVICE_TABLE "${EUGO_TMP_DIR}/device_table.cu")

# if(EXISTS "${GENERATED_DEVICE_TABLE}")
#     execute_process(
#         COMMAND sed -i.bak "s/\\bconst\\b//g" "${GENERATED_DEVICE_TABLE}"
#     )
# endif()

# # Remove `.bak` file
# file(REMOVE "${GENERATED_DEVICE_TABLE}.bak")

# # Step 4: Copy files to src/device/
# foreach(file ${GENERATED_CU_FILES})
#     file(
#         COPY "${EUGO_TMP_DIR}/${file}"
#         DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/src/device/"
#     )
# endforeach()

# # Step 5: Remove temp dir
# file(REMOVE_RECURSE "${EUGO_TMP_DIR}")

# # Step 6: Add generated files to CU_FILES
# add_library(colldevice OBJECT ${CU_FILES})

# # Step 7: Add include directories for colldevice. This is needed for the generated files to find the headers and for the generated files to be able to include each other
# target_include_directories(
#     colldevice
#     PRIVATE
#         ${CMAKE_CURRENT_SOURCE_DIR}/../../include
#         ${CMAKE_CURRENT_SOURCE_DIR}
# )

# # Step 8: Set properties for colldevice to enable CUDA separable compilation and resolve device symbols
# # Compiled kernels and collectives with relocatable device code ...
# set_property(TARGET colldevice PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# set_property(TARGET colldevice PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
