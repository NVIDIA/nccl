cmake_minimum_required(VERSION 3.18)
project(nccl_benchmark CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

# Find NCCL
if(WIN32)
    set(NCCL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../../build/include" CACHE PATH "NCCL include directory")
    set(NCCL_LIBRARY "${CMAKE_SOURCE_DIR}/../../build/lib/nccl.lib" CACHE PATH "NCCL library")
else()
    find_library(NCCL_LIBRARY nccl HINTS /usr/local/lib /usr/lib/x86_64-linux-gnu)
    set(NCCL_INCLUDE_DIR "/usr/include" CACHE PATH "NCCL include directory")
endif()

add_executable(nccl_stress_test nccl_stress_test.cu)

target_include_directories(nccl_stress_test PRIVATE 
    ${NCCL_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(nccl_stress_test PRIVATE 
    ${NCCL_LIBRARY}
    CUDA::cudart
)

if(WIN32)
    target_compile_definitions(nccl_stress_test PRIVATE WIN32 _WIN32)
    # Copy NCCL DLL to output directory
    add_custom_command(TARGET nccl_stress_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/../../build/lib/nccl.dll"
        $<TARGET_FILE_DIR:nccl_stress_test>
    )
endif()
