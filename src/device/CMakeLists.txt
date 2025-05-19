# set(
#   CU_FILES
#   common.cu
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

# Create an empty list of CU_FILES to store generated files
set(CU_FILES "")

# Define reusable temp directory
set(EUGO_TMP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/__eugo_tmp")

# Step 1: Create temp dir and run generate.py
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${EUGO_TMP_DIR})

execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate.py
    WORKING_DIRECTORY ${EUGO_TMP_DIR}
)

# Step 2: Get list of generated .cu files
file(GLOB GENERATED_CU_FILES RELATIVE ${EUGO_TMP_DIR} "${EUGO_TMP_DIR}/*.cu")

# Step 3: Append to CU_FILES
foreach(file ${GENERATED_CU_FILES})
    list(APPEND CU_FILES "${EUGO_TMP_DIR}/${file}")
endforeach()

# Step 3.5: Patch device_table.cu to remove all instances of 'const'
set(GENERATED_DEVICE_TABLE "${EUGO_TMP_DIR}/device_table.cu")

if(EXISTS "${GENERATED_DEVICE_TABLE}")
    execute_process(
        COMMAND sed -i.bak "s/\\bconst\\b//g" "${GENERATED_DEVICE_TABLE}"
    )
endif()

# Remove `.bak` file
file(REMOVE "${GENERATED_DEVICE_TABLE}.bak")

# Step 4: Copy files to src/device/
foreach(file ${GENERATED_CU_FILES})
    file(
        COPY "${EUGO_TMP_DIR}/${file}"
        DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/src/device/"
    )
endforeach()

# Step 5: Remove temp dir
file(REMOVE_RECURSE "${EUGO_TMP_DIR}")

# Step 6: Add generated files to CU_FILES
add_library(colldevice OBJECT ${CU_FILES})

# Step 7: Add include directories for colldevice. This is needed for the generated files to find the headers and for the generated files to be able to include each other
target_include_directories(
    colldevice
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Step 8: Set properties for colldevice to enable CUDA separable compilation and resolve device symbols
# Compiled kernels and collectives with relocatable device code ...
set_property(TARGET colldevice PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET colldevice PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
