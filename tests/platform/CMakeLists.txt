# Standalone Platform Tests CMakeLists.txt
# Can be built independently without CUDA or the main NCCL library

cmake_minimum_required(VERSION 3.18)

project(nccl_platform_tests LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Platform-specific settings
if(WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    add_compile_definitions(NOMINMAX)
    add_compile_definitions(WIN32_LEAN_AND_MEAN)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/include
)

# Standalone test executable (single file)
add_executable(test_platform_standalone
    test_platform_standalone.cpp
)

if(WIN32)
    target_link_libraries(test_platform_standalone PRIVATE ws2_32 iphlpapi bcrypt)
endif()

# Full test suite (multiple files)
add_executable(test_platform_full
    test_platform_detection.cpp
    test_time_functions.cpp
    test_socket_abstraction.cpp
    test_thread_abstraction.cpp
    test_cpu_affinity.cpp
    test_misc_functions.cpp
    test_main.cpp
)

if(WIN32)
    target_link_libraries(test_platform_full PRIVATE ws2_32 iphlpapi bcrypt)
else()
    target_link_libraries(test_platform_full PRIVATE pthread dl)
endif()

# Add tests
enable_testing()
add_test(NAME platform_standalone COMMAND test_platform_standalone)
add_test(NAME platform_full COMMAND test_platform_full)
