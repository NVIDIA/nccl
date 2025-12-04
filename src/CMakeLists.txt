# Source files
set(LIBSRCFILES
    bootstrap.cc
    channel.cc
    ce_coll.cc
    collectives.cc
    debug.cc
    enqueue.cc
    group.cc
    init.cc
    init_nvtx.cc
    proxy.cc
    transport.cc
    mnnvl.cc
    allocator.cc
    sym_kernels.cc
    dev_runtime.cc
)

# Add compatibility shim if using static cudart
if(CUDARTLIB STREQUAL "cudart_static")
    list(APPEND LIBSRCFILES enhcompat.cc)
endif()

# Configure pkg-config file
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/nccl.pc.in
    ${CMAKE_BINARY_DIR}/lib/pkgconfig/nccl.pc
    @ONLY
)

# Add files from subdirectories
add_subdirectory(transport)
add_subdirectory(misc)
add_subdirectory(register)
add_subdirectory(graph)
add_subdirectory(plugin)
add_subdirectory(device)
add_subdirectory(nccl_device)
add_subdirectory(ras)
add_subdirectory(scheduler)
add_subdirectory(gin)

# Strip source path prefix from debug info (GCC/Clang only)
if(NOT MSVC)
    add_compile_options(-fmacro-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}/=)
endif()

# Add all source files
list(APPEND LIBSRCFILES
    ${TRANSPORT_SOURCES}
    ${MISC_SOURCES}
    ${REGISTER_SOURCES}
    ${GRAPH_SOURCES}
    ${PLUGIN_SOURCES}
    ${RAS_SOURCES}
    ${SYM_SOURCES}
    ${SCHEDULER_SOURCES}
    ${GIN_SOURCES}
    ${DOCA_SOURCES}
)

###################### Create a shared NCCL library ############################
add_library(nccl SHARED)

target_sources(nccl PRIVATE ${LIBSRCFILES})

# Include directories
target_include_directories(nccl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/device
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/plugin
    ${CUDAToolkit_INCLUDE_DIRS}
    ${DOCA_HOME}/include
    ${CUDAToolkit_INCLUDE_DIRS}/cccl
)

# Read nccl.h.in and perform substitutions
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/nccl.h.in NCCL_H_CONTENT)
string(REPLACE "\${nccl:Major}" "${NCCL_MAJOR}" NCCL_H_CONTENT "${NCCL_H_CONTENT}")
string(REPLACE "\${nccl:Minor}" "${NCCL_MINOR}" NCCL_H_CONTENT "${NCCL_H_CONTENT}")
string(REPLACE "\${nccl:Patch}" "${NCCL_PATCH}" NCCL_H_CONTENT "${NCCL_H_CONTENT}")
string(REPLACE "\${nccl:Suffix}" "${NCCL_SUFFIX}" NCCL_H_CONTENT "${NCCL_H_CONTENT}")
string(REPLACE "\${nccl:Version}" "${NCCL_VERSION_CODE}" NCCL_H_CONTENT "${NCCL_H_CONTENT}")
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include)
file(WRITE ${CMAKE_BINARY_DIR}/include/nccl.h "${NCCL_H_CONTENT}")

# Create a custom target for the header (no longer needs custom_command)
add_custom_target(nccl_header_gen ALL DEPENDS ${CMAKE_BINARY_DIR}/include/nccl.h)

file(GLOB_RECURSE SRC_DEVICE_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include/nccl_device/*.h)

# Copy all device header files to the destination
foreach(HEADER_FILE ${SRC_DEVICE_HEADERS})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${HEADER_FILE} ${CMAKE_BINARY_DIR}/${HEADER_FILE} COPYONLY)
    list(APPEND DEVICE_HEADERS ${CMAKE_BINARY_DIR}/${HEADER_FILE})
endforeach()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/nccl_device.h ${CMAKE_BINARY_DIR}/include/nccl_device.h COPYONLY)

add_custom_target(nccl_header DEPENDS
    ${CMAKE_BINARY_DIR}/include/nccl_device.h
    ${DEVICE_HEADERS}
    ${DEVICE_DOCA_HEADERS}
)
add_dependencies(nccl_header nccl_header_gen)

add_dependencies(nccl nccl_header)
add_dependencies(nccl_device nccl_header)

# Set version and output name
set_target_properties(nccl PROPERTIES
    VERSION ${NCCL_MAJOR}.${NCCL_MINOR}.${NCCL_PATCH}
    SOVERSION ${NCCL_MAJOR}
    OUTPUT_NAME "nccl"
)

# Platform-specific prefix
if(NOT WIN32)
    set_target_properties(nccl PROPERTIES PREFIX "lib")
endif()

# Set CUDA specific flags
set_target_properties(nccl PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON
)

# Platform-specific link libraries
if(WIN32)
    target_link_libraries(nccl
        PRIVATE
        nccl_device
        ws2_32
        iphlpapi
        bcrypt
        ${CUDAToolkit_LIBRARIES}
        ${EXTRA_LIBS}
    )
    # Add Windows-specific compile definitions
    target_compile_definitions(nccl PRIVATE
        NCCL_EXPORTS
        _CRT_SECURE_NO_WARNINGS
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
else()
    target_link_libraries(nccl
        PRIVATE
        nccl_device
        pthread
        rt
        dl
        ${CUDAToolkit_LIBRARIES}
        ${EXTRA_LIBS}
    )
    # Add version script for symbol visibility control (Linux only)
    target_link_options(nccl PRIVATE
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/libnccl.map"
    )
endif()

# Set output directories for nccl shared library
set_target_properties(nccl PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"  # For Windows DLLs
)

###################### Create a ras binary executable ############################
# ncclras is Linux-only (requires POSIX sockets and signals)
if(NOT WIN32)
    set(RAS_BINSRCFILES ras/client.cc)

    add_executable(ncclras ${RAS_BINSRCFILES})

    target_include_directories(ncclras PUBLIC
        ${CMAKE_BINARY_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    add_dependencies(ncclras nccl_header)

    target_link_libraries(ncclras
        PRIVATE
        pthread
        rt
        dl
    )

    # Set output directory for ncclras executable
    set_target_properties(ncclras PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()

###################### Create a static NCCL library ############################
add_library(nccl_static STATIC ${LIBSRCFILES})

# Include directories
target_include_directories(nccl_static PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/device
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/plugin
    ${CUDAToolkit_INCLUDE_DIRS}
    transport/gdaki/doca-gpunetio/include
    ${CUDAToolkit_INCLUDE_DIRS}/cccl
)

# Add dependency on nccl_header
add_dependencies(nccl_static nccl_header)

# Platform-specific link libraries for static library
if(WIN32)
    target_link_libraries(nccl_static
        PRIVATE
        nccl_device
        ws2_32
        iphlpapi
        bcrypt
        ${CUDAToolkit_LIBRARIES}
        ${EXTRA_LIBS}
    )
    target_compile_definitions(nccl_static PRIVATE
        _CRT_SECURE_NO_WARNINGS
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
else()
    target_link_libraries(nccl_static
        PRIVATE
        nccl_device
        pthread
        rt
        dl
        ${CUDAToolkit_LIBRARIES}
        ${EXTRA_LIBS}
    )
endif()

# Set CUDA specific flags
set_target_properties(nccl_static PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON
)

# Set output directory for nccl_static library
set_target_properties(nccl_static PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
