set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_CUDA_STANDARD 11)

set(CMAKE_CUDA_STANDARD_REQUIRED YES)
set(CMAKE_CUDA_EXTENSIONS NO)
set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} -fcuda-rdc -Wnvcc-compat -Xcuda-ptxas -maxrregcount=96 -Xcuda-fatbinary -compress-all -fPIC -fvisibility=hidden"
)

set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -fPIC -fvisibility=hidden -Wall -Wno-unused-function -Wno-sign-compare -Wvla"
)
add_compile_options($<$<CONFIG:Debug>:-ggdb3>)
add_compile_options($<$<NOT:$<CONFIG:Debug>>:-O3>)

if(PRINT_VERBOSE)
    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} -fcuda-rdc -Wnvcc-compat -Xcuda-ptxas -v -Wall,-Wextra,-Wno-unused-parameter"
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()

# Find system NVTX3
if(USE_SYSTEM_NVTX)
    # Fail immediately if not found
    find_package(nvtx3 CONFIG REQUIRED)
else()
    # Disable NVTX-related features
    add_compile_definitions(NVTX_DISABLE)
endif()
