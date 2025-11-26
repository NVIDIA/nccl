# NCCL Tests CMakeLists.txt
cmake_minimum_required(VERSION 3.18)

project(nccl_tests LANGUAGES CXX C)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find NCCL (from parent build)
if(NOT TARGET nccl)
    message(FATAL_ERROR "NCCL target not found. Build tests from the main CMake project.")
endif()

# Platform-specific settings
if(WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    add_compile_definitions(NOMINMAX)
    add_compile_definitions(WIN32_LEAN_AND_MEAN)
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_SOURCE_DIR}/src/include/platform
    ${CMAKE_BINARY_DIR}/include
)

# Test executable for platform abstraction
add_executable(test_platform
    platform/test_platform_detection.cpp
    platform/test_time_functions.cpp
    platform/test_socket_abstraction.cpp
    platform/test_thread_abstraction.cpp
    platform/test_cpu_affinity.cpp
    platform/test_misc_functions.cpp
    platform/test_main.cpp
)

target_link_libraries(test_platform PRIVATE nccl)

if(WIN32)
    target_link_libraries(test_platform PRIVATE ws2_32 iphlpapi)
endif()

# Add test
enable_testing()
add_test(NAME platform_tests COMMAND test_platform)

# Optional: Add verbose output
set_tests_properties(platform_tests PROPERTIES
    ENVIRONMENT "NCCL_DEBUG=INFO"
)
