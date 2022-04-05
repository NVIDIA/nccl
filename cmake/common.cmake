set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_CUDA_STANDARD 11)

set(CMAKE_CUDA_STANDARD_REQUIRED YES)
set(CMAKE_CUDA_EXTENSIONS NO)
set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -Xptxas -maxrregcount=96 -Xfatbin -compress-all -fPIC -fvisibility=hidden"
)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -fPIC -fvisibility=hidden -Wall -Wno-unused-function -Wno-sign-compare  -Wvla"
)
add_compile_options($<$<CONFIG:Debug>:-ggdb3>)
add_compile_options($<$<NOT:$<CONFIG:Debug>>:-O3>)
if(VERBOSE)
  set(CMAKE_CUDA_FLAGS
      "${CMAKE_CUDA_FLAGS} -Xptxas -v -Xcompiler -Wall,-Wextra,-Wno-unused-parameter"
  )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()

if(TRACE)
  add_compile_definitions(ENABLE_TRACE)
endif()

if(NOT NVTX)
  add_compile_definitions(NVTX_DISABLE)
endif()

if(KEEP)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -keep")
endif()

if(PROFAPI)
  add_compile_definitions(PROFAPI)
endif()
