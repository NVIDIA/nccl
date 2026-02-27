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
    proxy.cc
    transport.cc
    mnnvl.cc
    allocator.cc
    sym_kernels.cc
    dev_runtime.cc
    mem_manager.cc
)

# Add NVTX support if enabled
if(NVTX)
    list(APPEND LIBSRCFILES init_nvtx.cc)
endif()

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
add_subdirectory(os)
add_subdirectory(gin)
add_subdirectory(rma)

if(NCCL_OS_LINUX)
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
    ${OS_SOURCES}
    ${GIN_SOURCES}
    ${DOCA_SOURCES}
    ${RMA_SOURCES}
)

###################### Create a shared NCCL library ############################
add_library(nccl SHARED)

target_sources(nccl PRIVATE ${LIBSRCFILES})
target_sources(nccl PRIVATE $<TARGET_OBJECTS:nccl_device>)

# Include directories
target_include_directories(nccl
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/device
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/plugin
    ${DOCA_HOME}/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}/cccl
    ${CMAKE_BINARY_DIR}/obj/include
)

add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/include/nccl.h
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/include
    COMMAND sed -e "s/\\\$$\\{nccl:Major\\}/${NCCL_MAJOR}/g"
                -e "s/\\\$$\\{nccl:Minor\\}/${NCCL_MINOR}/g"
                -e "s/\\\$$\\{nccl:Patch\\}/${NCCL_PATCH}/g"
                -e "s/\\\$$\\{nccl:Suffix\\}/${NCCL_SUFFIX}/g"
                -e "s/\\\$$\\{nccl:Version\\}/${NCCL_VERSION_CODE}/g"
                ${CMAKE_CURRENT_SOURCE_DIR}/nccl.h.in > ${CMAKE_BINARY_DIR}/include/nccl.h
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/nccl.h.in
)

file(GLOB_RECURSE SRC_DEVICE_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include/nccl_device/*.h)

# Copy all device header files to the destination
foreach(HEADER_FILE ${SRC_DEVICE_HEADERS})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${HEADER_FILE} ${CMAKE_BINARY_DIR}/${HEADER_FILE} COPYONLY)
    list(APPEND DEVICE_HEADERS ${CMAKE_BINARY_DIR}/${HEADER_FILE})
endforeach()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/nccl_device.h ${CMAKE_BINARY_DIR}/include/nccl_device.h COPYONLY)

add_custom_target(nccl_header DEPENDS
    ${CMAKE_BINARY_DIR}/include/nccl.h
    ${CMAKE_BINARY_DIR}/include/nccl_device.h
    ${DEVICE_HEADERS}
    ${DEVICE_DOCA_HEADERS}
)

add_dependencies(nccl nccl_header)
add_dependencies(nccl_device nccl_header)

# Set version and output name
set_target_properties(nccl PROPERTIES
    VERSION ${NCCL_MAJOR}.${NCCL_MINOR}.${NCCL_PATCH}
    SOVERSION ${NCCL_MAJOR}
    OUTPUT_NAME "nccl"
    PREFIX "lib"
)

#
# Generate nccl_git_version.h
#
set(GIT_VERSION_FILE "${CMAKE_BINARY_DIR}/obj/include/nccl_git_version.h")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/obj/include")
add_custom_target(generate_git_version ALL
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/src/misc/generate_git_version.py ${GIT_VERSION_FILE}
  COMMENT "Checking git version"
  BYPRODUCTS ${GIT_VERSION_FILE}
)
add_dependencies(nccl generate_git_version)

# Git version overrides (set via cmake -DNCCL_GIT_BRANCH=xxx -DNCCL_GIT_COMMIT_HASH=yyy)
set(NCCL_GIT_BRANCH "" CACHE STRING "Override git branch name")
set(NCCL_GIT_COMMIT_HASH "" CACHE STRING "Override git commit hash")
if(NCCL_GIT_BRANCH)
    target_compile_definitions(nccl PRIVATE NCCL_GIT_BRANCH="${NCCL_GIT_BRANCH}")
endif()
if(NCCL_GIT_COMMIT_HASH)
    target_compile_definitions(nccl PRIVATE NCCL_GIT_COMMIT_HASH="${NCCL_GIT_COMMIT_HASH}")
endif()

# Set CUDA specific flags
set_target_properties(nccl PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON
)

# Link libraries
target_link_libraries(nccl
    PRIVATE
    Threads::Threads
    ${CMAKE_DL_LIBS}
    CUDA::cudart
    ${EXTRA_LIBS}
)

# Add version script for symbol visibility control
target_link_options(nccl PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/libnccl.map"
)

# Set output directories for nccl shared library
set_target_properties(nccl PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

###################### Create a ras binary executable ############################
set(RAS_BINSRCFILES ras/client.cc)

add_executable(ncclras ${RAS_BINSRCFILES})

target_include_directories(ncclras
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

add_dependencies(ncclras nccl_header)

target_link_libraries(ncclras
    PRIVATE
    pthread
    ${CMAKE_DL_LIBS}
)

# Set output directory for ncclras executable
set_target_properties(ncclras PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

###################### Create a static NCCL library ############################
add_library(nccl_static STATIC ${LIBSRCFILES})
target_sources(nccl_static PRIVATE $<TARGET_OBJECTS:nccl_device>)

# Include directories
target_include_directories(nccl_static
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/device
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/plugin
    transport/net_ib/gdaki/doca-gpunetio/include
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}/cccl
    ${CMAKE_BINARY_DIR}/obj/include
)

# Add dependency on nccl_header
add_dependencies(nccl_static nccl_header)
add_dependencies(nccl_static generate_git_version)

# Git version overrides for static lib
if(NCCL_GIT_BRANCH)
    target_compile_definitions(nccl_static PRIVATE NCCL_GIT_BRANCH="${NCCL_GIT_BRANCH}")
endif()
if(NCCL_GIT_COMMIT_HASH)
    target_compile_definitions(nccl_static PRIVATE NCCL_GIT_COMMIT_HASH="${NCCL_GIT_COMMIT_HASH}")
endif()

# Link libraries
target_link_libraries(nccl_static
    PUBLIC
    Threads::Threads
    ${CMAKE_DL_LIBS}
    CUDA::cudart
    ${EXTRA_LIBS}
)

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

###################### Install targets / headers / pkg-config ############################
install(TARGETS nccl nccl_static
  EXPORT NCCLTargets
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)

# Install tools, but do not export them as linkable imported targets.
install(TARGETS ncclras
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

# Install generated + copied public headers from the build tree.
install(DIRECTORY "${CMAKE_BINARY_DIR}/include/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

# Install pkg-config metadata generated during configure.
install(FILES "${CMAKE_BINARY_DIR}/lib/pkgconfig/nccl.pc"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
)
