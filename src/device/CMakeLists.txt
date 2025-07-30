set(CU_FILES onerank_reduce.cu functions.cu)

add_library(colldevice OBJECT ${CU_FILES})

set(datatypes "i8;u8;i32;u32;i64;u64;f16;f32;f64")
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11")
  list(APPEND datatypes bf16)
endif()
set(ops "sum;prod;min;max;premulsum;sumpostdiv")
list(LENGTH ops op_num)
math(EXPR op_num "${op_num} - 1")
list(LENGTH datatypes datatype_num)
math(EXPR datatype_num "${datatype_num} - 1")
set(base_files "sendrecv;all_reduce;all_gather;broadcast;reduce;reduce_scatter")
foreach(base IN LISTS base_files)
  foreach(opn RANGE ${op_num})
    list(GET ops ${opn} op)
    foreach(dtn RANGE ${datatype_num})
      list(GET datatypes ${dtn} dt)
      set(new_file ${CMAKE_CURRENT_BINARY_DIR}/${base}_${op}_${dt}.cu)
      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${base}.cu ${new_file}
                     COPYONLY)
      set_property(SOURCE ${new_file} PROPERTY COMPILE_DEFINITIONS
                                               NCCL_OP=${opn} NCCL_TYPE=${dtn})
      target_sources(colldevice PRIVATE ${new_file})
    endforeach()
  endforeach()
endforeach()

target_include_directories(
  colldevice PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include
                     ${CMAKE_CURRENT_SOURCE_DIR})

# Compiled kernels and collectives with relocatable device code ...
set_property(TARGET colldevice PROPERTY CUDA_SEPARABLE_COMPILATION ON)
