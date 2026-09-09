# EFA-DP-Direct
# EFA CUDA data-path direct device headers (mirrors the DOCA GPUNetIO pattern
# in transport/net_ib/gdaki/CMakeLists.txt).
# Allow users to specify EFA_DP_DIRECT_HOME via cmake -DEFA_DP_DIRECT_HOME=<path>
# Default to this directory if not specified.
if(NOT DEFINED EFA_DP_DIRECT_HOME)
    set(EFA_DP_DIRECT_HOME ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# Copy EFA-DP-Direct headers to the build directory
set(EFA_DP_DIRECT_INCLUDE_SOURCE_DIR ${EFA_DP_DIRECT_HOME}/include)
set(EFA_DP_DIRECT_INCLUDE_DEST_DIR ${CMAKE_BINARY_DIR}/include/nccl_device/transport/net_efa_gda/efa-dp-direct/include)

# Gather header files from the source directory
file(GLOB EFA_DP_DIRECT_HEADER_DEVICE_CUH ${EFA_DP_DIRECT_INCLUDE_SOURCE_DIR}/device/*.cuh)
file(GLOB EFA_DP_DIRECT_HEADER_DEVICE_H   ${EFA_DP_DIRECT_INCLUDE_SOURCE_DIR}/device/*.h)
file(GLOB EFA_DP_DIRECT_HEADER_COMMON     ${EFA_DP_DIRECT_INCLUDE_SOURCE_DIR}/common/*.h)

# Copy device/ headers (.cuh and .h)
foreach(HEADER_FILE ${EFA_DP_DIRECT_HEADER_DEVICE_CUH} ${EFA_DP_DIRECT_HEADER_DEVICE_H})
    get_filename_component(HEADER_NAME ${HEADER_FILE} NAME)
    configure_file(${HEADER_FILE} ${EFA_DP_DIRECT_INCLUDE_DEST_DIR}/device/${HEADER_NAME} COPYONLY)
    list(APPEND DEVICE_EFA_DP_HEADERS ${EFA_DP_DIRECT_INCLUDE_DEST_DIR}/device/${HEADER_NAME})
endforeach()

# Copy common/ headers
foreach(HEADER_FILE ${EFA_DP_DIRECT_HEADER_COMMON})
    get_filename_component(HEADER_NAME ${HEADER_FILE} NAME)
    configure_file(${HEADER_FILE} ${EFA_DP_DIRECT_INCLUDE_DEST_DIR}/common/${HEADER_NAME} COPYONLY)
    list(APPEND DEVICE_EFA_DP_HEADERS ${EFA_DP_DIRECT_INCLUDE_DEST_DIR}/common/${HEADER_NAME})
endforeach()

# Add EFA-DP-Direct device headers to parent scope
set(DEVICE_EFA_DP_HEADERS ${DEVICE_EFA_DP_HEADERS} PARENT_SCOPE)
