# DOCA
# Allow users to specify DOCA_HOME via cmake -DDOCA_HOME=<path>
# Default to transport/net_ib/gdaki/doca-gpunetio if not specified
if(NOT DEFINED DOCA_HOME)
    set(DOCA_HOME ${CMAKE_CURRENT_SOURCE_DIR}/doca-gpunetio)
endif()

# Copy DOCA GPUNetIO headers to build directory
set(DOCA_INCLUDE_SOURCE_DIR ${DOCA_HOME}/include)
set(DOCA_INCLUDE_DEST_DIR ${CMAKE_BINARY_DIR}/include/nccl_device/gin/gdaki/doca_gpunetio)

# Get all header files from the source directory, including subfolders
file(GLOB DOCA_HEADER_TOP ${DOCA_INCLUDE_SOURCE_DIR}/doca_gpunetio_device.h)
file(GLOB DOCA_HEADER_COMMON ${DOCA_INCLUDE_SOURCE_DIR}/common/*.h)
file(GLOB DOCA_HEADER_DEVICE ${DOCA_INCLUDE_SOURCE_DIR}/device/*.cuh)

# Copy top-level header
foreach(HEADER_FILE ${DOCA_HEADER_TOP})
    get_filename_component(HEADER_NAME ${HEADER_FILE} NAME)
    configure_file(${HEADER_FILE} ${DOCA_INCLUDE_DEST_DIR}/${HEADER_NAME} COPYONLY)
    list(APPEND DEVICE_DOCA_HEADERS ${DOCA_INCLUDE_DEST_DIR}/${HEADER_NAME})
endforeach()

# Copy common/ headers
foreach(HEADER_FILE ${DOCA_HEADER_COMMON})
    get_filename_component(HEADER_NAME ${HEADER_FILE} NAME)
    configure_file(${HEADER_FILE} ${DOCA_INCLUDE_DEST_DIR}/common/${HEADER_NAME} COPYONLY)
    list(APPEND DEVICE_DOCA_HEADERS ${DOCA_INCLUDE_DEST_DIR}/common/${HEADER_NAME})
endforeach()

# Copy device/ headers
foreach(HEADER_FILE ${DOCA_HEADER_DEVICE})
    get_filename_component(HEADER_NAME ${HEADER_FILE} NAME)
    configure_file(${HEADER_FILE} ${DOCA_INCLUDE_DEST_DIR}/device/${HEADER_NAME} COPYONLY)
    list(APPEND DEVICE_DOCA_HEADERS ${DOCA_INCLUDE_DEST_DIR}/device/${HEADER_NAME})
endforeach()

# Add DOCA device headers to parent scope
set(DEVICE_DOCA_HEADERS ${DEVICE_DOCA_HEADERS} PARENT_SCOPE)

# DOCA sources
set(DOCA_SOURCES
    ${DOCA_HOME}/src/doca_verbs_qp.cpp
    ${DOCA_HOME}/src/doca_verbs_cq.cpp
    ${DOCA_HOME}/src/doca_verbs_device_attr.cpp
    ${DOCA_HOME}/src/doca_verbs_umem.cpp
    ${DOCA_HOME}/src/doca_verbs_srq.cpp
    ${DOCA_HOME}/src/doca_verbs_uar.cpp
    ${DOCA_HOME}/src/doca_gpunetio.cpp
    ${DOCA_HOME}/src/doca_gpunetio_log.cpp
    ${DOCA_HOME}/src/doca_gpunetio_high_level.cpp
    ${DOCA_HOME}/src/doca_verbs_cuda_wrapper.cpp
    ${DOCA_HOME}/src/doca_verbs_mlx5dv_wrapper.cpp
    ${DOCA_HOME}/src/doca_verbs_ibv_wrapper.cpp
    ${DOCA_HOME}/src/doca_gpunetio_gdrcopy.cpp
)

# Add DOCA sources to parent scope
set(DOCA_SOURCES ${DOCA_SOURCES} PARENT_SCOPE)

# Add DOCA_HOME to parent scope
set(DOCA_HOME ${DOCA_HOME} PARENT_SCOPE)

# Add gin_host_gdaki.cc to TRANSPORT_SOURCES in parent scope
set(TRANSPORT_SOURCES ${TRANSPORT_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/gin_host_gdaki.cc PARENT_SCOPE)
