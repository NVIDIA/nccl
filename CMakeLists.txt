cmake_minimum_required(VERSION 3.25)

# Version information
# Read makefiles/version.mk file
file(READ ${CMAKE_SOURCE_DIR}/makefiles/version.mk VERSION_CONTENT)
string(REGEX REPLACE ".*NCCL_MAJOR[ ]*:=[ ]*([0-9]+).*" "\\1" NCCL_MAJOR "${VERSION_CONTENT}")
string(REGEX REPLACE ".*NCCL_MINOR[ ]*:=[ ]*([0-9]+).*" "\\1" NCCL_MINOR "${VERSION_CONTENT}")
string(REGEX REPLACE ".*NCCL_PATCH[ ]*:=[ ]*([0-9]+).*" "\\1" NCCL_PATCH "${VERSION_CONTENT}")
string(REGEX REPLACE ".*NCCL_SUFFIX[ ]*:=[ ]*([a-zA-Z0-9]*).*" "\\1" NCCL_SUFFIX "${VERSION_CONTENT}")
string(REGEX REPLACE ".*PKG_REVISION[ ]*:=[ ]*([0-9]+).*" "\\1" PKG_REVISION "${VERSION_CONTENT}")
math(EXPR NCCL_VERSION_CODE "(${NCCL_MAJOR} * 10000) + (${NCCL_MINOR} * 100) + ${NCCL_PATCH}")

# Make version information available to C++ source files
add_compile_definitions(
    NCCL_USE_CMAKE
    NCCL_MAJOR=${NCCL_MAJOR}
    NCCL_MINOR=${NCCL_MINOR}
    NCCL_PATCH=${NCCL_PATCH}
    NCCL_VERSION_CODE=${NCCL_VERSION_CODE}
)

set(ENV{NCCL_USE_CMAKE} "1")

project(NCCL VERSION ${NCCL_MAJOR}.${NCCL_MINOR}.${NCCL_PATCH}
        LANGUAGES CUDA CXX C)

# Make CMAKE_BUILD_TYPE to release by default if not set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Package building options
option(BUILD_DEBIAN_PACKAGE "Build Debian package" OFF)
option(BUILD_REDHAT_PACKAGE "Build Redhat package" OFF)
option(BUILD_TXZ_PACKAGE "Build TXZ package" OFF)
option(BUILD_CONDA_PACKAGE "Build Conda package" OFF)
option(BUILD_SRCTXZ_PACKAGE "Build source tarball package" OFF)
option(BUILD_DOC_PACKAGE "Build documentation package" OFF)
option(BUILD_PACKAGES "Build all packages" OFF)
option(VERBOSE "Enable verbose output" OFF)
option(KEEP "Keep intermediate files" OFF)
option(DEBUG "Enable debug build" OFF)
option(ASAN "Enable Address Sanitizer" OFF)
option(UBSAN "Enable Undefined Behavior Sanitizer" OFF)
option(TRACE "Enable tracing" OFF)
option(WERROR "Treat warnings as errors" OFF)
option(PROFAPI "Enable profiling API" ON)
option(NVTX "Enable NVTX" ON)
option(RDMA_CORE "Enable RDMA core" OFF)
option(NET_PROFILER "Enable network profiler" OFF)
option(MLX5DV "Enable MLX5DV" OFF)
option(MAX_EXT_NET_PLUGINS "Maximum external network plugins" 0)
option(EMIT_LLVM_IR "Generate LLVM IR for device APIs" OFF)

# Detect OS
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(NCCL_OS_WINDOWS ON)
    add_definitions(-DNCCL_OS_WINDOWS)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(NCCL_OS_LINUX ON)
    add_definitions(-DNCCL_OS_LINUX)
else()
    message(FATAL_ERROR "Unsupported OS: ${CMAKE_SYSTEM_NAME}")
endif()

find_package(CUDAToolkit REQUIRED)

# CUDA version detection
string(REGEX MATCH "([0-9]+\\.[0-9]+)" CUDA_VERSION "${CUDAToolkit_VERSION}")

# Extract major and minor version numbers
string(REGEX MATCH "([0-9]+)" CUDA_MAJOR "${CUDA_VERSION}")
string(REGEX MATCH "([0-9]+)$" CUDA_MINOR "${CUDA_VERSION}")
string(REGEX REPLACE ".*\\.([0-9]+)$" "\\1" CUDA_MINOR "${CUDA_VERSION}")

# Add CUDA version definitions after find_package
add_compile_definitions(
    CUDA_MAJOR=${CUDA_MAJOR}
    CUDA_MINOR=${CUDA_MINOR}
)

# CUDA architecture flags
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    message(STATUS "CMAKE_CUDA_ARCHITECTURES not defined or empty, setting default values based on CUDA version")

    if(${CUDA_MAJOR} LESS 9)
        set(CMAKE_CUDA_ARCHITECTURES "35;50;60;61")
    elseif(${CUDA_MAJOR} EQUAL 9)
        set(CMAKE_CUDA_ARCHITECTURES "35;50;60;61;70")
    elseif(${CUDA_MAJOR} EQUAL 10)
        set(CMAKE_CUDA_ARCHITECTURES "35;50;60;61;70")
    elseif(${CUDA_MAJOR} EQUAL 11)
        if(${CUDA_MINOR} LESS 8)
            set(CMAKE_CUDA_ARCHITECTURES "35;50;60;61;70;80")
        else()
            set(CMAKE_CUDA_ARCHITECTURES "35;50;60;61;70;80;90")
        endif()
    elseif(${CUDA_MAJOR} EQUAL 12)
        if(${CUDA_MINOR} LESS 8)
            set(CMAKE_CUDA_ARCHITECTURES "50;60;61;70;80;90")
        else()
            set(CMAKE_CUDA_ARCHITECTURES "50;60;61;70;80;90;100;120")
        endif()
    elseif(${CUDA_MAJOR} EQUAL 13)
        set(CMAKE_CUDA_ARCHITECTURES "50;60;61;70;80;90;100;110;120")
    else()
        # For future CUDA versions, include all architectures up to the latest known
        set(CMAKE_CUDA_ARCHITECTURES "50;60;61;70;80;90;100;110;120")
    endif()
endif()
message(STATUS "Using CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fvisibility=hidden -Wall -Wno-unused-function -Wno-sign-compare -Wvla -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -Xptxas -maxrregcount=96 -Xfatbin -compress-all -fPIC")

# Sanitizer options
if(ASAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -static-libasan")
endif()

if(UBSAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=undefined -static-libubsan")
endif()

# Additional options
if(TRACE)
    add_definitions(-DENABLE_TRACE)
endif()

if(NOT NVTX)
    add_definitions(-DNVTX_DISABLE)
endif()

if(WERROR)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
endif()

if(PROFAPI)
    add_definitions(-DPROFAPI)
endif()

set(EXTRA_LIBS)

# RDMA and MLX5DV are Linux-specific features
if(RDMA_CORE)
    add_definitions(-DNCCL_BUILD_RDMA_CORE=1)
    find_library(VERBS_LIBRARY NAMES verbs)
    if(VERBS_LIBRARY)
        list(APPEND EXTRA_LIBS ${VERBS_LIBRARY})
    endif()
endif()

if(MLX5DV)
    add_definitions(-DNCCL_BUILD_MLX5DV=1)
    find_library(MLX5_LIBRARY NAMES mlx5)
    if(MLX5_LIBRARY)
        list(APPEND EXTRA_LIBS ${MLX5_LIBRARY})
    endif()
endif()

if(NET_PROFILER)
    add_definitions(-DNCCL_ENABLE_NET_PROFILING=1)
endif()

if(MAX_EXT_NET_PLUGINS GREATER 0)
    add_definitions(-DNCCL_NET_MAX_PLUGINS=${MAX_EXT_NET_PLUGINS})
endif()

add_definitions(-DDOCA_VERBS_USE_CUDA_WRAPPER)
add_definitions(-DDOCA_VERBS_USE_NET_WRAPPER)
add_definitions(-DNCCL_GIN_PROXY_ENABLE=1)
if(EMIT_LLVM_IR)
    add_definitions(-DEMIT_LLVM_IR=1)
endif()

# Library dependencies
find_library(RT_LIBRARY NAMES rt)
if(RT_LIBRARY)
    list(APPEND EXTRA_LIBS ${RT_LIBRARY})
endif()

# Debug/Release specific flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS} -O0 -G -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS} -O3")

add_subdirectory(ext-mixed/example)
add_subdirectory(ext-net)
add_subdirectory(ext-profiler/example)
add_subdirectory(ext-tuner/example)
add_subdirectory(ext-env/example)
add_subdirectory(src)

# Add package building subdirectories
if(BUILD_DEBIAN_PACKAGE OR BUILD_PACKAGES)
  add_subdirectory(pkg/debian)
endif()

if(BUILD_REDHAT_PACKAGE OR BUILD_PACKAGES)
  add_subdirectory(pkg/redhat)
endif()

if(BUILD_TXZ_PACKAGE OR BUILD_PACKAGES)
  add_subdirectory(pkg/txz)
endif()

if(BUILD_CONDA_PACKAGE OR BUILD_PACKAGES)
  add_subdirectory(pkg/conda)
endif()

if(BUILD_SRCTXZ_PACKAGE OR BUILD_PACKAGES)
  add_subdirectory(pkg/srctxz)
endif()

if(BUILD_DOC_PACKAGE OR BUILD_PACKAGES)
  add_subdirectory(pkg/doc)
endif()

if(EMIT_LLVM_IR)
    add_subdirectory(ir)
    add_dependencies(llvm_ir nccl_header)
    add_custom_target(nccl_with_ir ALL DEPENDS nccl llvm_ir)
    message(STATUS "LLVM IR generation will be included in default build")
endif()
